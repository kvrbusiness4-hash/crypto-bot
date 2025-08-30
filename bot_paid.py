# bot_paid.py
# -*- coding: utf-8 -*-

import os
import hmac
import json
import time
import math
import hashlib
import logging
import asyncio
from typing import Any, Dict, Optional, List

import requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    AIORateLimiter,
)

# ─────────── Логи ───────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
L = logging.getLogger("bot")

# ─────────── ENV / Конфіг ───────────
BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_ID    = int(os.getenv("ADMIN_ID", "0") or "0")

# Bybit (реал): https://api.bybit.com  |  (за потреби testnet: https://api-testnet.bybit.com)
BYBIT_BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").rstrip("/")

# Проксі для requests (http/https/socks5)
BYBIT_PROXY = os.getenv("BYBIT_PROXY", "").strip()

BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY", "").strip()
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "").strip()

# Торгові налаштування (дефолти — як у тебе)
DEFAULT_SCAN_MIN = int(os.getenv("DEFAULT_SCAN_MIN", os.getenv("HEARTBEAT_MIN", "15")))
SIZE_USDT = float(os.getenv("SIZE_USDT", "5"))
LEVERAGE  = int(os.getenv("LEVERAGE",  "3"))
SL_PCT    = float(os.getenv("SL_PCT",   "3"))   # стоп у %
TP_PCT    = float(os.getenv("TP_PCT",   "5"))   # тейк у %
TRADE_ENABLED = os.getenv("TRADE_ENABLED", "ON").upper() == "ON"

# Ліміт одночасних угод
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "2"))

UTC_FMT = "%Y-%m-%d %H:%M:%SZ"

# ─────────── Глобальні ───────────
app: Optional["Application"] = None
scheduler: Optional[AsyncIOScheduler] = None
auto_scan_job = None  # APScheduler job
OPEN_TRADES_CACHE: Dict[str, Dict[str, Any]] = {}  # symbol -> {'side','qty','ts'}

# ─────────── Утіліти ───────────
def utc_now_str() -> str:
    import datetime as dt
    return dt.datetime.utcnow().strftime(UTC_FMT)

def _requests_proxies() -> Optional[Dict[str, str]]:
    if not BYBIT_PROXY:
        return None
    return {"http": BYBIT_PROXY, "https": BYBIT_PROXY}

def _ts_ms() -> str:
    return str(int(time.time() * 1000))

def _bybit_sign(payload: str) -> str:
    return hmac.new(
        BYBIT_API_SECRET.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

def _bybit_headers(payload: str) -> Dict[str, str]:
    """v5 headers: timestamp + api_key + recv_window + body/query  → HMAC"""
    return {
        "X-BAPI-SIGN": _bybit_sign(payload),
        "X-BAPI-API-KEY": BYBIT_API_KEY,
        "X-BAPI-TIMESTAMP": payload.split("|", 1)[0],   # перша частина — timestamp
        "X-BAPI-RECV-WINDOW": "5000",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def _payload(ts: str, body_or_query: str) -> str:
    # Формула v5: pre_sign = timestamp + api_key + recv_window + (query_string|json_body)
    return f"{ts}{BYBIT_API_KEY}5000{body_or_query}"

async def api_get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Асинхронний wrapper над requests.get з проксі та перевіркою JSON."""
    url = f"{BYBIT_BASE_URL}{path}"
    proxies = _requests_proxies()
    headers = {"Accept": "application/json"}

    def _do() -> Dict[str, Any]:
        r = requests.get(url, params=params, headers=headers, timeout=20, proxies=proxies)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "application/json" not in ct.lower():
            raise RuntimeError(f"Bybit non-JSON (possible block): {r.text[:200]}")
        return r.json()

    return await asyncio.to_thread(_do)

async def api_get_json_auth(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """GET з підписом (v5)"""
    url = f"{BYBIT_BASE_URL}{path}"
    proxies = _requests_proxies()
    # query_string: сортування не обов'язкове, але бажано стабільне
    items = sorted((k, str(v)) for k, v in (params or {}).items())
    query = "&".join(f"{k}={v}" for k, v in items)
    ts = _ts_ms()
    pre = _payload(ts, query)
    headers = _bybit_headers(pre)

    def _do() -> Dict[str, Any]:
        r = requests.get(url, params=params, headers=headers, timeout=20, proxies=proxies)
        r.raise_for_status()
        return r.json()

    return await asyncio.to_thread(_do)

async def api_post_json_auth(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """POST з підписом (v5)"""
    url = f"{BYBIT_BASE_URL}{path}"
    proxies = _requests_proxies()
    body_json = json.dumps(body, separators=(",", ":"))
    ts = _ts_ms()
    pre = _payload(ts, body_json)
    headers = _bybit_headers(pre)

    def _do() -> Dict[str, Any]:
        r = requests.post(url, data=body_json, headers=headers, timeout=20, proxies=proxies)
        r.raise_for_status()
        return r.json()

    return await asyncio.to_thread(_do)

# ─────────── Сигнали (простий відбір top30) ───────────
async def fetch_tickers_linear() -> List[Dict[str, Any]]:
    data = await api_get_json("/v5/market/tickers", {"category": "linear"})
    return (data.get("result") or {}).get("list") or []

def pick_top30_strong(tickers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Вибір «сильних»: беремо top30 за 24h turnover, і дивимось 24h % (price24hPcnt).
    Якщо pcnt >= +2% → LONG, <= -2% → SHORT. (Просто і стабільно без історії.)
    """
    rows = []
    for t in tickers:
        try:
            sym = t["symbol"]              # напр. BTCUSDT
            last = float(t["lastPrice"])
            pcnt = float(t.get("price24hPcnt") or 0) * 100.0
            turn = float(t.get("turnover24h") or 0)  # USD
            rows.append((turn, sym, last, pcnt))
        except Exception:
            continue
    rows.sort(key=lambda x: x[0], reverse=True)
    top = rows[:30]

    out = []
    for _, sym, last, pcnt in top:
        if pcnt >= 2.0:
            side = "Buy"
        elif pcnt <= -2.0:
            side = "Sell"
        else:
            continue
        out.append({"symbol": sym, "last": last, "pcnt": pcnt, "side": side})
    # повертаємо максимум 2 кращих за абсолютним % рухом
    out.sort(key=lambda z: abs(z["pcnt"]), reverse=True)
    return out[:2]

def price_sl_tp(side: str, px: float, sl_pct: float, tp_pct: float) -> (float, float):
    if side == "Buy":
        sl = px * (1 - sl_pct/100.0)
        tp = px * (1 + tp_pct/100.0)
    else:
        sl = px * (1 + sl_pct/100.0)
        tp = px * (1 - tp_pct/100.0)
    # округлимо до 6 знаків — універсально
    return round(sl, 6), round(tp, 6)

# ─────────── Облік відкритих угод (ліміт = 2) ───────────
def can_open_new_trades() -> bool:
    return len(OPEN_TRADES_CACHE) < MAX_OPEN_TRADES

def remember_open_trade(symbol: str, side: str, qty: float):
    OPEN_TRADES_CACHE[symbol] = {"side": side, "qty": qty, "ts": int(time.time())}

async def bybit_fetch_open_positions() -> List[Dict[str, Any]]:
    """Повертає відкриті позиції (linear). Якщо ключі не задані — порожньо."""
    if not (BYBIT_API_KEY and BYBIT_API_SECRET):
        return []
    resp = await api_get_json_auth("/v5/position/list", {"category": "linear"})
    rows = (resp.get("result") or {}).get("list") or []
    out = []
    for it in rows:
        try:
            sym = it["symbol"]
            side = it["side"]    # Buy/Sell
            size = float(it.get("size") or 0)
            if size > 0:
                out.append({"symbol": sym, "side": side, "qty": size})
        except Exception:
            pass
    return out

async def sync_open_trades_cache():
    """Синхронізуємо кеш з біржею (щоб бачити закриті позиції)."""
    live = await bybit_fetch_open_positions()
    new_cache = {}
    for r in live:
        sym = r["symbol"]
        new_cache[sym] = {
            "side": r["side"],
            "qty":  r["qty"],
            "ts":   OPEN_TRADES_CACHE.get(sym, {}).get("ts", int(time.time())),
        }
    OPEN_TRADES_CACHE.clear()
    OPEN_TRADES_CACHE.update(new_cache)

async def ensure_leverage(symbol: str, leverage: int) -> None:
    """Ставимо плече для символу (не фатально, якщо помилка)."""
    if not (BYBIT_API_KEY and BYBIT_API_SECRET):
        return
    body = {"category": "linear", "symbol": symbol, "buyLeverage": str(leverage), "sellLeverage": str(leverage)}
    try:
        await api_post_json_auth("/v5/position/set-leverage", body)
    except Exception as e:
        L.warning("set-leverage failed for %s: %s", symbol, e)

async def bybit_place_market_order(symbol: str, side: str, qty: float, tp: float, sl: float) -> (bool, str):
    """
    Створює MARKET-ордер з TP/SL (tpSlMode=Full).
    qty — у контратах/к-сті (для USDT perpetual це «qty» у валюті контракту).
    """
    if not (BYBIT_API_KEY and BYBIT_API_SECRET):
        return False, "BYBIT_API_KEY/SECRET not set"

    body = {
        "category": "linear",
        "symbol": symbol,
        "side": side,                     # Buy / Sell
        "orderType": "Market",
        "qty": str(qty),
        "timeInForce": "IOC",
        "tpSlMode": "Full",
        "takeProfit": str(tp),
        "stopLoss": str(sl),
        # Можна ще додати reduceOnly=False, але за замовч. False
    }
    try:
        # гарантуємо плече (не блокує угоду, лише намагаємось один раз)
        await ensure_leverage(symbol, LEVERAGE)

        resp = await api_post_json_auth("/v5/order/create", body)
        if str(resp.get("retCode")) == "0":
            return True, "OK"
        return False, f"retCode={resp.get('retCode')} {resp.get('retMsg')}"
    except Exception as e:
        return False, str(e)

def qty_from_usdt(symbol_price: float, size_usdt: float, lev: int) -> float:
    """
    Для USDT-перп: приблизно qty = (size_usdt * lev) / price.
    Округлимо до 3 знаків — універсально (для більшості інструментів підійде).
    """
    raw = (size_usdt * lev) / max(1e-9, symbol_price)
    # Зазвичай мін. крок 0.001 (залежить від символу). Робимо безпечне округлення:
    q = math.floor(raw * 1000) / 1000.0
    return max(q, 0.001)

async def trade_loop_pick_best():
    """
    1) Тягнемо тікери, беремо top30 найліквідніших
    2) Обираємо до 2 найсильніших (за 24h %)
    3) Якщо є слоти — відкриваємо ринки з SL/TP
    """
    await sync_open_trades_cache()
    slots = MAX_OPEN_TRADES - len(OPEN_TRADES_CACHE)
    if slots <= 0:
        L.info("No slots: open=%d/%d", len(OPEN_TRADES_CACHE), MAX_OPEN_TRADES)
        return

    tickers = await fetch_tickers_linear()
    cands = pick_top30_strong(tickers)  # до 2

    opened = 0
    for c in cands:
        if opened >= slots:
            break
        sym  = c["symbol"]
        last = c["last"]
        side = c["side"]

        if sym in OPEN_TRADES_CACHE:
            continue  # вже відкрита — не дублюємо

        sl, tp = price_sl_tp(side, last, SL_PCT, TP_PCT)
        qty = qty_from_usdt(last, SIZE_USDT, LEVERAGE)
        if qty <= 0:
            continue

        ok, msg = await bybit_place_market_order(sym, side, qty, tp, sl)
        if ok:
            remember_open_trade(sym, side, qty)
            opened += 1
            L.info("Opened %s %s qty=%s SL=%s TP=%s", sym, side, qty, sl, tp)
        else:
            L.warning("Open fail %s %s: %s", sym, side, msg)

# ─────────── Команди ───────────
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

def _interval_min_from_job(job) -> int:
    # захищено дістає інтервал з APScheduler job
    try:
        if not job:
            return int(DEFAULT_SCAN_MIN)
        trig = getattr(job, "trigger", None)
        if trig is not None:
            iv = getattr(trig, "interval", None)
            if iv is not None:
                ts = getattr(iv, "total_seconds", None)
                if callable(ts):
                    sec = int(ts())
                else:
                    sec = int(iv)
                return max(1, sec // 60) if sec >= 60 else max(1, sec)
    except Exception:
        pass
    return int(DEFAULT_SCAN_MIN)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    interval_min = _interval_min_from_job(auto_scan_job)
    proxy_state = "використовується" if BYBIT_PROXY else "не використовується"
    open_list = ", ".join([f"{s}:{d['side']}" for s, d in OPEN_TRADES_CACHE.items()]) or "—"

    text = (
        f"Статус автоскану: {'ON' if auto_scan_job else 'OFF'} · кожні {interval_min} хв.\n"
        f"SL={SL_PCT:.2f}% · TP={TP_PCT:.2f}%\n"
        f"TRADE_ENABLED={'ON' if TRADE_ENABLED else 'OFF'} · SIZE={SIZE_USDT:.2f} USDT · LEV={LEVERAGE}\n"
        f"Фільтр: TOP30 · Проксі: {proxy_state}\n"
        f"Відкриті угоди ({len(OPEN_TRADES_CACHE)}/{MAX_OPEN_TRADES}): {open_list}\n"
        f"UTC: {utc_now_str()}"
    )
    await update.message.reply_text(text)

async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔎 Сканую ринок…")
    try:
        tickers = await fetch_tickers_linear()
        cands = pick_top30_strong(tickers)
        if not cands:
            await msg.edit_text(f"ℹ️ Сильних сигналів зараз немає · UTC {utc_now_str()}")
            return
        lines = []
        for c in cands:
            sl, tp = price_sl_tp(c["side"], c["last"], SL_PCT, TP_PCT)
            lines.append(
                f"• {c['symbol']} {c['side']} @ {c['last']} · 24h {c['pcnt']:+.2f}% · SL {sl} · TP {tp}"
            )
        await msg.edit_text("📈 Сильні сигнали:\n" + "\n".join(lines))
    except Exception as e:
        await msg.edit_text(f"✖️ Помилка сканера: {e}")

async def cmd_trade_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TRADE_ENABLED
    TRADE_ENABLED = True
    await update.message.reply_text("Автотрейд: УВІМКНЕНО ✓")

async def cmd_trade_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TRADE_ENABLED
    TRADE_ENABLED = False
    await update.message.reply_text("Автотрейд: ВИМКНЕНО ⛔")

async def cmd_set_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SIZE_USDT
    try:
        v = float(context.args[0])
        if v <= 0: raise ValueError
        SIZE_USDT = v
        await update.message.reply_text(f"OK. SIZE_USDT={SIZE_USDT:.2f}")
    except Exception:
        await update.message.reply_text("Формат: /set_size 5")

async def cmd_set_lev(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LEVERAGE
    try:
        v = int(context.args[0])
        if v < 1: raise ValueError
        LEVERAGE = v
        await update.message.reply_text(f"OK. LEVERAGE={LEVERAGE}")
    except Exception:
        await update.message.reply_text("Формат: /set_lev 3")

async def cmd_set_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SL_PCT, TP_PCT
    try:
        sl = float(context.args[0]); tp = float(context.args[1])
        if sl <= 0 or tp <= 0: raise ValueError
        SL_PCT, TP_PCT = sl, tp
        await update.message.reply_text(f"OK. SL={SL_PCT:.2f}%  TP={TP_PCT:.2f}%")
    except Exception:
        await update.message.reply_text("Формат: /set_risk 3 5")

# ─────────── Автоскан / Heartbeat ───────────
async def heartbeat(_):
    if ADMIN_ID:
        try:
            await app.bot.send_message(ADMIN_ID, "✅ heartbeat: я працюю")
        except Exception:
            pass

async def auto_scan_tick():
    try:
        # 1) Якщо автотрейд увімкнено — запускаємо трейд-цикл з лімітом 2
        if TRADE_ENABLED:
            await trade_loop_pick_best()
        else:
            # Якщо ні — хоча б прогріваємо ринок, щоб бачити лог
            tickers = await fetch_tickers_linear()
            L.info("Auto-scan (no-trade) OK: %s tickers", len(tickers))
    except Exception as e:
        L.error("Auto-scan error: %s", e)

async def cmd_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_scan_job
    try:
        minutes = int(context.args[0]) if context.args else DEFAULT_SCAN_MIN
        minutes = max(1, minutes)
    except Exception:
        minutes = DEFAULT_SCAN_MIN

    if auto_scan_job:
        try:
            scheduler.remove_job(auto_scan_job.id)
        except Exception:
            pass
        auto_scan_job = None

    auto_scan_job = scheduler.add_job(auto_scan_tick, "interval", minutes=minutes, next_run_time=None)
    await update.message.reply_text(f"✓ Автоскан увімкнено: кожні {minutes} хв.")

async def cmd_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_scan_job
    if auto_scan_job:
        try:
            scheduler.remove_job(auto_scan_job.id)
        except Exception:
            pass
        auto_scan_job = None
    await update.message.reply_text("⛔ Автоскан вимкнено.")

# ─────────── Main ───────────
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

    # Команди
    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("signals",  cmd_signals))
    app.add_handler(CommandHandler("trade_on", cmd_trade_on))
    app.add_handler(CommandHandler("trade_off",cmd_trade_off))
    app.add_handler(CommandHandler("set_size", cmd_set_size))
    app.add_handler(CommandHandler("set_lev",  cmd_set_lev))
    app.add_handler(CommandHandler("set_risk", cmd_set_risk))
    app.add_handler(CommandHandler("auto_on",  cmd_auto_on))
    app.add_handler(CommandHandler("auto_off", cmd_auto_off))

    # Планувальник
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.start()

    # Heartbeat адміну раз на годину
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
