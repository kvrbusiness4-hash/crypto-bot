# bot_paid.py
# -*- coding: utf-8 -*-

import os
import hmac
import time
import json
import math
import hashlib
import logging
import asyncio
from typing import Any, Dict, Optional, List

import requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    AIORateLimiter,
)

# ---------- Логи ----------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
L = logging.getLogger("bot")

# ---------- ENV ----------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_ID = os.getenv("ADMIN_ID", "").strip()

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "").strip()
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "").strip()
BYBIT_BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").rstrip("/")
BYBIT_PROXY = os.getenv("BYBIT_PROXY", "").strip()  # напр.: http://user:pass@ip:port

DEFAULT_SCAN_MIN = int(os.getenv("DEFAULT_SCAN_MIN", os.getenv("HEARTBEAT_MIN", "60")))
SIZE_USDT = float(os.getenv("SIZE_USDT", "5"))
LEVERAGE = int(os.getenv("LEVERAGE", "3"))
SL_PCT = float(os.getenv("SL_PCT", "3"))
TP_PCT = float(os.getenv("TP_PCT", "5"))
TRADE_ENABLED = os.getenv("TRADE_ENABLED", "ON").upper() == "ON"

UTC_FMT = "%Y-%m-%d %H:%M:%SZ"

# ---------- Глобальні ----------
app: Optional[Application] = None
scheduler: Optional[AsyncIOScheduler] = None
auto_scan_job = None

# локальний облік відкритих угод
open_trades: List[Dict[str, Any]] = []
OPEN_LIMIT = 2  # максимум одночасних позицій

# ---------- Утиліти ----------
def utc_now_str() -> str:
    import datetime as dt
    return dt.datetime.utcnow().strftime(UTC_FMT)

def _requests_proxies() -> Optional[Dict[str, str]]:
    if not BYBIT_PROXY:
        return None
    return {"http": BYBIT_PROXY, "https": BYBIT_PROXY}

async def http_get_json(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    proxies = _requests_proxies()
    def _do():
        r = requests.get(url, params=params, headers=headers or {}, timeout=20, proxies=proxies)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "application/json" not in ct.lower():
            raise RuntimeError(f"Non-JSON response: {r.text[:200]}")
        return r.json()
    return await asyncio.to_thread(_do)

# ---------- Bybit private ----------
def _bybit_auth_headers(params: Dict[str, Any]) -> Dict[str, str]:
    ts = str(int(time.time() * 1000))
    params["api_key"] = BYBIT_API_KEY
    params["timestamp"] = ts
    params["recv_window"] = "5000"
    signed_str = "&".join([f"{k}={params[k]}" for k in sorted(params)])
    sign = hmac.new(BYBIT_API_SECRET.encode(), signed_str.encode(), hashlib.sha256).hexdigest()
    params["sign"] = sign
    return {"Content-Type": "application/x-www-form-urlencoded"}

async def bybit_private_post(path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise RuntimeError("Bybit API keys are not set")
    url = f"{BYBIT_BASE_URL}{path}"
    proxies = _requests_proxies()
    payload = dict(data)
    headers = _bybit_auth_headers(payload)
    def _do():
        r = requests.post(url, data=payload, headers=headers, timeout=20, proxies=proxies)
        r.raise_for_status()
        j = r.json()
        if str(j.get("retCode")) != "0":
            raise RuntimeError(f"Bybit error: {j.get('retMsg')} ({j.get('retCode')})")
        return j
    return await asyncio.to_thread(_do)

async def bybit_private_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise RuntimeError("Bybit API keys are not set")
    url = f"{BYBIT_BASE_URL}{path}"
    proxies = _requests_proxies()
    payload = dict(params)
    headers = _bybit_auth_headers(payload)
    def _do():
        r = requests.get(url, params=payload, headers=headers, timeout=20, proxies=proxies)
        r.raise_for_status()
        j = r.json()
        if str(j.get("retCode")) != "0":
            raise RuntimeError(f"Bybit error: {j.get('retMsg')} ({j.get('retCode')})")
        return j
    return await asyncio.to_thread(_do)

# ---------- Публічні Bybit ----------
async def get_tickers_linear() -> Dict[str, Any]:
    return await http_get_json(f"{BYBIT_BASE_URL}/v5/market/tickers", {"category": "linear"}, headers={"Accept": "application/json"})

async def get_last_price(symbol: str) -> float:
    j = await http_get_json(f"{BYBIT_BASE_URL}/v5/market/tickers", {"category": "linear", "symbol": symbol}, headers={"Accept":"application/json"})
    arr = ((j or {}).get("result", {}) or {}).get("list", []) or []
    if not arr:
        raise RuntimeError("Price not found")
    return float(arr[0]["lastPrice"])

# ---------- Торгові функції ----------
def _qty_from_usdt(size_usdt: float, price: float, step: float = 0.0001) -> str:
    raw = size_usdt / price
    q = math.floor(raw / step) * step
    if q <= 0:
        q = step
    prec = max(0, str(step)[::-1].find('.'))
    return f"{q:.{prec}f}"

async def ensure_leverage(symbol: str, leverage: int):
    try:
        await bybit_private_post("/v5/position/set-leverage", {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        })
    except Exception as e:
        L.warning("set-leverage failed: %s", e)

async def place_market_with_sl_tp(symbol: str, side: str, size_usdt: float, sl_pct: float, tp_pct: float) -> Dict[str, Any]:
    price = await get_last_price(symbol)
    qty = _qty_from_usdt(size_usdt, price)
    await ensure_leverage(symbol, LEVERAGE)

    if side.upper() == "Buy":
        tp_price = price * (1 + tp_pct / 100.0)
        sl_price = price * (1 - sl_pct / 100.0)
    else:
        tp_price = price * (1 - tp_pct / 100.0)
        sl_price = price * (1 + sl_pct / 100.0)

    def _fmt(v: float) -> str:
        return f"{v:.6f}"

    payload = {
        "category": "linear",
        "symbol": symbol,
        "side": side,                   # Buy / Sell
        "orderType": "Market",
        "qty": qty,
        "timeInForce": "IOC",
        "reduceOnly": "false",
        "takeProfit": _fmt(tp_price),
        "stopLoss": _fmt(sl_price),
        "tpTriggerBy": "LastPrice",
        "slTriggerBy": "LastPrice",
        "positionIdx": "0",
    }
    j = await bybit_private_post("/v5/order/create", payload)
    oid = ((j.get("result") or {}).get("orderId")) or ""
    return {"price": price, "qty": qty, "orderId": oid}

async def fetch_open_positions() -> List[Dict[str, Any]]:
    j = await bybit_private_get("/v5/position/list", {"category": "linear"})
    arr = (j.get("result") or {}).get("list", []) or []
    res = []
    for p in arr:
        if float(p.get("size", "0") or 0) > 0:
            res.append(p)
    return res

async def refresh_open_trades():
    global open_trades
    pos = await fetch_open_positions()
    new_list = []
    for p in pos:
        symbol = p["symbol"]
        side = p.get("side", "Buy")
        entry = float(p.get("avgPrice","0") or 0)
        size = float(p.get("size","0") or 0)
        if size <= 0:
            continue
        new_list.append({"symbol": symbol, "side": side, "entryPrice": entry, "qty": size})
    open_trades = new_list

# ---------- Вибір сигналів ----------
def pick_strong_symbols(tickers: List[Dict[str, Any]], limit: int = 2) -> List[str]:
    rows = []
    for t in tickers:
        s = t.get("symbol","")
        if not s.endswith("USDT"):
            continue
        try:
            ch = float(t.get("price24hPcnt","0"))
        except:
            continue
        rows.append((abs(ch), s))
    rows.sort(reverse=True, key=lambda x: x[0])
    return [s for _, s in rows[:limit]]

# ---------- Команди ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Привіт! Бот готовий.\n"
        "/status — статус\n"
        "/signals — скан сильних (top30)\n"
        "/trade_on | /trade_off — автотрейд\n"
        f"/auto_on {DEFAULT_SCAN_MIN} | /auto_off — автоскан\n"
        f"/set_size {int(SIZE_USDT)} — розмір угоди в USDT\n"
        f"/set_lev {LEVERAGE} — плече\n"
        f"/set_risk {int(SL_PCT)} {int(TP_PCT)} — SL/TP у %"
    )
    await update.message.reply_text(text)

def get_interval_min_safe(job) -> int:
    try:
        if not job:
            return int(DEFAULT_SCAN_MIN)
        trig = getattr(job, "trigger", None)
        if trig is not None:
            iv = getattr(trig, "interval", None)
            if iv is not None:
                ts = getattr(iv, "total_seconds", None)
                sec = int(ts()) if callable(ts) else int(iv)
                return max(1, sec // 60) if sec >= 60 else max(1, sec)
        iv = getattr(job, "interval", None)
        if iv is not None:
            ts = getattr(iv, "total_seconds", None)
            sec = int(ts()) if callable(ts) else int(iv)
            return max(1, sec // 60) if sec >= 60 else max(1, sec)
    except Exception:
        pass
    return int(DEFAULT_SCAN_MIN)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await refresh_open_trades()
    opened = len(open_trades)
    interval_min = get_interval_min_safe(auto_scan_job)
    proxy_state = "використовується" if BYBIT_PROXY else "не використовується"
    text = (
        f"Статус автоскану: {'ON' if auto_scan_job else 'OFF'} · кожні {interval_min} хв.\n"
        f"SL={SL_PCT:.2f}% · TP={TP_PCT:.2f}%\n"
        f"TRADE_ENABLED={'ON' if TRADE_ENABLED else 'OFF'} · SIZE={SIZE_USDT:.2f} USDT · LEV={LEVERAGE}\n"
        f"Фільтр: TOP30 · Проксі: {proxy_state}\n"
        f"Відкриті угоди ({opened}/{OPEN_LIMIT}): " + ("—" if opened == 0 else ', '.join([t['symbol'] for t in open_trades])) + "\n"
        f"UTC: {utc_now_str()}"
    )
    await update.message.reply_text(text)

async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔎 Сканую ринок…")
    try:
        data = await get_tickers_linear()
        rows = data.get("result", {}).get("list", []) or []
        syms = pick_strong_symbols(rows, 2)
        if not syms:
            await msg.edit_text("Порожньо.")
            return
        out = ["📈 Сильні сигнали:"]
        for s in syms:
            try:
                p = await get_last_price(s)
                sl = p * (1 - SL_PCT/100.0)
                tp = p * (1 + TP_PCT/100.0)
                out.append(f"• {s} Buy @ {p:.6f} · SL {sl:.6f} · TP {tp:.6f}")
            except:
                out.append(f"• {s}")
        await msg.edit_text("\n".join(out))
    except Exception as e:
        await msg.edit_text(f"❌ Помилка: {e}")

async def cmd_trade_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TRADE_ENABLED
    TRADE_ENABLED = True
    await update.message.reply_text("Автотрейд: УВІМКНЕНО ✓")

async def cmd_trade_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TRADE_ENABLED
    TRADE_ENABLED = False
    await update.message.reply_text("Автотрейд: ВИМКНЕНО ✕")

async def cmd_set_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SIZE_USDT
    try:
        v = float(context.args[0])
        if v <= 0:
            raise ValueError
        SIZE_USDT = v
        await update.message.reply_text(f"OK. SIZE_USDT={SIZE_USDT:.2f}")
    except Exception:
        await update.message.reply_text("Формат: /set_size 5")

async def cmd_set_lev(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LEVERAGE
    try:
        v = int(context.args[0])
        if v < 1:
            raise ValueError
        LEVERAGE = v
        await update.message.reply_text(f"OK. LEVERAGE={LEVERAGE}")
    except Exception:
        await update.message.reply_text("Формат: /set_lev 3")

async def cmd_set_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SL_PCT, TP_PCT
    try:
        sl = float(context.args[0])
        tp = float(context.args[1])
        if sl <= 0 or tp <= 0:
            raise ValueError
        SL_PCT, TP_PCT = sl, tp
        await update.message.reply_text(f"OK. SL={SL_PCT:.2f}%  TP={TP_PCT:.2f}%")
    except Exception:
        await update.message.reply_text("Формат: /set_risk 3 5")

# ---------- Автоскан/торгівля ----------
async def try_open_trades_from_signals():
    if not TRADE_ENABLED:
        return
    await refresh_open_trades()
    if len(open_trades) >= OPEN_LIMIT:
        return

    data = await get_tickers_linear()
    rows = data.get("result", {}).get("list", []) or []
    syms = pick_strong_symbols(rows, limit=OPEN_LIMIT*2)

    busy = set(t["symbol"] for t in open_trades)
    candidates = [s for s in syms if s not in busy]
    to_open = max(0, OPEN_LIMIT - len(open_trades))

    for s in candidates[:to_open]:
        try:
            deal = await place_market_with_sl_tp(s, "Buy", SIZE_USDT, SL_PCT, TP_PCT)
            open_trades.append({"symbol": s, "side": "Buy", "qty": float(deal["qty"]), "entryPrice": deal["price"], "orderId": deal["orderId"]})
            text = f"✅ Відкрито {s} • qty {deal['qty']} • @ {deal['price']:.6f}"
            if ADMIN_ID:
                try: await app.bot.send_message(ADMIN_ID, text)
                except: pass
            L.info(text)
        except Exception as e:
            err = f"❌ Не зміг відкрити {s}: {e}"
            L.error(err)
            if ADMIN_ID:
                try: await app.bot.send_message(ADMIN_ID, err)
                except: pass

async def heartbeat(_: ContextTypes.DEFAULT_TYPE):
    if ADMIN_ID:
        try: await app.bot.send_message(ADMIN_ID, "💗 heartbeat")
        except: pass

async def auto_scan_tick():
    try: await refresh_open_trades()
    except Exception as e: L.warning("refresh_open_trades: %s", e)
    try: await try_open_trades_from_signals()
    except Exception as e: L.error("auto trade step error: %s", e)

# ---------- Auto on/off ----------
async def cmd_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_scan_job
    try:
        minutes = int(context.args[0]) if context.args else DEFAULT_SCAN_MIN
        minutes = max(1, minutes)
    except Exception:
        minutes = DEFAULT_SCAN_MIN

    if auto_scan_job:
        try: scheduler.remove_job(auto_scan_job.id)
        except Exception: pass
        auto_scan_job = None

    auto_scan_job = scheduler.add_job(auto_scan_tick, "interval", minutes=minutes, next_run_time=None)
    await update.message.reply_text(f"✓ Автоскан увімкнено: кожні {minutes} хв.")

async def cmd_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_scan_job
    if auto_scan_job:
        try: scheduler.remove_job(auto_scan_job.id)
        except Exception: pass
        auto_scan_job = None
    await update.message.reply_text("✕ Автоскан вимкнено.")

# ---------- Main ----------
async def main():
    global app, scheduler, auto_scan_job
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .rate_limiter(AIORateLimiter())
        .build()
    )

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("signals", cmd_signals))
    app.add_handler(CommandHandler("trade_on",  cmd_trade_on))
    app.add_handler(CommandHandler("trade_off", cmd_trade_off))
    app.add_handler(CommandHandler("set_size", cmd_set_size))
    app.add_handler(CommandHandler("set_lev",  cmd_set_lev))
    app.add_handler(CommandHandler("set_risk", cmd_set_risk))
    app.add_handler(CommandHandler("auto_on",  cmd_auto_on))
    app.add_handler(CommandHandler("auto_off", cmd_auto_off))

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.start()
    scheduler.add_job(lambda: asyncio.create_task(heartbeat(None)), "interval", minutes=60)

    L.info("Starting bot…")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
