# -*- coding: utf-8 -*-
"""
bot_alpaca.py — мінімальна версія з:
- Telegram-бот (python-telegram-bot v20+)
- інтеграція Alpaca (Data API + Trading API)
- відкриття ордерів (ринкових) з TP/SL у STATE
- моніторинг TP/SL у фоні + повідомлення OPEN/CLOSE
- статус-команди і демо-сканер

ENV:
  TELEGRAM_BOT_TOKEN
  ALPACA_API_KEY
  ALPACA_API_SECRET
  ALPACA_BASE_URL         (наприклад: https://paper-api.alpaca.markets)
  ALPACA_DATA_URL         (наприклад: https://data-api.alpaca.markets)
  MODE                    (scalp/aggressive/safe) — опційно
  SCAN_INTERVAL_SEC       (опційно, за замовчуванням 300)
  TP_PCT                  (0.008 => 0.8%) – дефолт для демонстрації
  SL_PCT                  (0.015 => 1.5%) – дефолт для демонстрації
"""
import os
import asyncio
import math
import json
from typing import Dict, Any, Optional, Tuple

import aiohttp
from aiohttp import ClientSession, ClientTimeout

from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes
)

# ================== ENV ==================
TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_TOKEN") or "").strip()

ALPACA_API_KEY   = (os.getenv("ALPACA_API_KEY") or "").strip()
ALPACA_API_SECRET= (os.getenv("ALPACA_API_SECRET") or "").strip()

ALPACA_BASE_URL  = (os.getenv("ALPACA_BASE_URL")  or "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_URL  = (os.getenv("ALPACA_DATA_URL")  or "https://data-api.alpaca.markets").rstrip("/")

SCAN_INTERVAL_SEC= int(os.getenv("SCAN_INTERVAL_SEC") or 300)

TP_PCT_DEFAULT   = float(os.getenv("TP_PCT") or 0.008)   # +0.8%
SL_PCT_DEFAULT   = float(os.getenv("SL_PCT") or 0.015)   # -1.5%

# ================== GLOBALS ==================
TIMEOUT = ClientTimeout(total=30)
HTTP: Optional[ClientSession] = None

STATE: Dict[int, Dict[str, Any]] = {}  # STATE[chat_id] = {"positions": {symbol: {...}}, "mode": "scalp", ...}

# демо-список крипти (символи Alpaca)
CRYPTO_LIST = ["AAVE/USD", "AVAX/USD", "BAT/USD"]

# ================== HELPERS ==================
def ensure_chat_state(chat_id: int) -> Dict[str, Any]:
    st = STATE.setdefault(chat_id, {})
    st.setdefault("positions", {})
    st.setdefault("mode", "scalp")
    return st

def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def fmt_price(p: float) -> str:
    if p >= 100:
        return f"{p:.4f}"
    elif p >= 1:
        return f"{p:.5f}"
    else:
        return f"{p:.8f}"

def symbol_is_crypto(sym: str) -> bool:
    return "/" in sym

def alpaca_auth_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }

async def alpaca_get(url: str, params: Optional[dict] = None) -> Any:
    async with HTTP.get(url, headers=alpaca_auth_headers(), params=params) as r:
        if r.status >= 400:
            raise RuntimeError(f"Alpaca GET {url} {r.status}: {await r.text()}")
        return await r.json()

async def alpaca_post(url: str, payload: dict) -> Any:
    async with HTTP.post(url, headers=alpaca_auth_headers(), data=json.dumps(payload)) as r:
        txt = await r.text()
        if r.status >= 400:
            raise RuntimeError(f"Alpaca POST {url} {r.status}: {txt}")
        try:
            return json.loads(txt)
        except Exception:
            return txt

# ---- quotes/last trade ----
async def get_last_price(symbol: str) -> float:
    """Повертає останню ціну (close) через Alpaca Data API (stocks або crypto)."""
    if symbol_is_crypto(symbol):
        # data v1beta3 crypto/us/{symbol}/quotes/latest
        # Але на paper часто зручніше брати last trade:
        # /v1beta3/crypto/us/trades/latest?symbols=AAVE/USD
        url = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/trades/latest"
        data = await alpaca_get(url, params={"symbols": symbol})
        # {'trades': {'AAVE/USD': {'p': 310.1, ...}}}
        price = float(data["trades"][symbol]["p"])
        return price
    else:
        # stocks: /v2/stocks/{symbol}/trades/latest
        url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest"
        data = await alpaca_get(url)
        price = float(data["trade"]["p"])
        return price

# ================== TELEGRAM NOTIFY ==================
async def notify_open(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, symbol: str, side: str,
                      entry_price: float, tp_price: float, sl_price: float,
                      tp_pct: float, sl_pct: float, qty: float):
    msg = (f"🟢 ORDER OK: {symbol} {side.upper()} ${qty:.2f}\n"
           f"Вхід @ {fmt_price(entry_price)} · "
           f"TP {fmt_price(tp_price)} (+{fmt_pct(tp_pct)}) · "
           f"SL {fmt_price(sl_price)} (-{fmt_pct(sl_pct)})")
    await ctx.bot.send_message(chat_id, msg)

async def notify_close(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, symbol: str, side: str,
                       entry_price: float, exit_price: float, reason: str):
    if side.lower() == "long":
        pnl_pct = (exit_price / entry_price - 1.0) * 100.0
    else:
        pnl_pct = (entry_price / exit_price - 1.0) * 100.0
    msg = (f"🔴 CLOSE ORDER: {symbol} {side.upper()}\n"
           f"Вихід @ {fmt_price(exit_price)} · Причина: {reason}\n"
           f"PnL: {pnl_pct:.2f}% (вхід {fmt_price(entry_price)})")
    await ctx.bot.send_message(chat_id, msg)

# ================== TRADING ==================
async def place_market_order(symbol: str, notional_usd: float, side: str) -> Tuple[str, float, float]:
    """
    Ринковий ордер у доларовому нотіоналі.
    Вертає: (order_id, filled_qty, avg_fill_price)
    На paper ринок виконується миттєво як правило.
    """
    url = f"{ALPACA_BASE_URL}/v2/orders"
    payload = {
        "symbol": symbol.replace("/", ""),  # для crypto у trading api — без слеша? (у папері часто із слешем недозволено)
        "notional": round(notional_usd, 2),
        "side": "buy" if side.lower() in ("buy", "long") else "sell",
        "type": "market",
        "time_in_force": "gtc",
        "asset_class": "crypto" if symbol_is_crypto(symbol) else "us_equity",
    }
    # Для crypto paper trading на Alpaca зазвичай symbol як "AAVEUSD" (без "/")
    if symbol_is_crypto(symbol):
        payload["symbol"] = symbol.replace("/", "")

    data = await alpaca_post(url, payload)
    oid = data.get("id", "")
    filled_qty = float(data.get("filled_qty") or 0.0) if data.get("filled_qty") else 0.0
    avg_price = float(data.get("filled_avg_price") or 0.0) if data.get("filled_avg_price") else 0.0

    # Якщо ще не filled, пробуємо підтягнути ордер
    if avg_price <= 0.0:
        # почекаємо коротко й запросимо
        await asyncio.sleep(0.7)
        url_o = f"{ALPACA_BASE_URL}/v2/orders/{oid}"
        od = await alpaca_get(url_o)
        filled_qty = float(od.get("filled_qty") or 0.0) if od.get("filled_qty") else 0.0
        avg_price = float(od.get("filled_avg_price") or 0.0) if od.get("filled_avg_price") else 0.0

    # Фолбек на last price
    if avg_price <= 0.0:
        avg_price = await get_last_price(symbol)

    return oid, filled_qty, avg_price

async def close_position_market(symbol: str, qty: float, side: str) -> Tuple[str, float]:
    """
    Закриття: якщо відкривали long (buy), то тут sell qty; якщо short — buy qty.
    Повертає (order_id, exit_price_approx).
    """
    exit_side = "sell" if side.lower() == "long" else "buy"
    notional = 0.0  # можна пустити по qty; для crypto — краще qty
    url = f"{ALPACA_BASE_URL}/v2/orders"
    payload = {
        "symbol": symbol.replace("/", ""),
        "qty": str(qty),
        "side": exit_side,
        "type": "market",
        "time_in_force": "gtc",
        "asset_class": "crypto" if symbol_is_crypto(symbol) else "us_equity",
    }
    if symbol_is_crypto(symbol):
        payload["symbol"] = symbol.replace("/", "")

    data = await alpaca_post(url, payload)
    oid = data.get("id", "")
    avg_exit = float(data.get("filled_avg_price") or 0.0) if data.get("filled_avg_price") else 0.0
    if avg_exit <= 0.0:
        await asyncio.sleep(0.7)
        od = await alpaca_get(f"{ALPACA_BASE_URL}/v2/orders/{oid}")
        avg_exit = float(od.get("filled_avg_price") or 0.0) if od.get("filled_avg_price") else 0.0
    if avg_exit <= 0.0:
        avg_exit = await get_last_price(symbol)
    return oid, avg_exit

# ================== POSITION STATE ==================
async def open_position(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, symbol: str,
                        side: str, notional_usd: float,
                        tp_pct: float = TP_PCT_DEFAULT,
                        sl_pct: float = SL_PCT_DEFAULT):
    """
    Відкрити позицію: MARKET notional, зберегти entry/TP/SL у STATE і повідомити.
    """
    try:
        oid, filled_qty, entry_price = await place_market_order(symbol, notional_usd, side)
    except Exception as e:
        await ctx.bot.send_message(chat_id, f"🔴 ORDER FAIL {symbol}: {e}")
        return

    is_long = side.lower() in ("buy", "long")
    if is_long:
        tp_price = entry_price * (1.0 + tp_pct)
        sl_price = entry_price * (1.0 - sl_pct)
    else:
        tp_price = entry_price * (1.0 - tp_pct)
        sl_price = entry_price * (1.0 + sl_pct)

    st = ensure_chat_state(chat_id)
    st["positions"][symbol] = {
        "side": "long" if is_long else "short",
        "entry": float(entry_price),
        "tp": float(tp_price),
        "sl": float(sl_price),
        "tp_pct": float(tp_pct),
        "sl_pct": float(sl_pct),
        "qty": float(filled_qty) if filled_qty > 0 else round(notional_usd / max(entry_price, 1e-9), 6),
        "order_id": oid,
    }

    await notify_open(ctx, chat_id, symbol, "buy" if is_long else "sell",
                      entry_price, tp_price, sl_price, tp_pct, sl_pct,
                      st["positions"][symbol]["qty"])

async def maybe_close_on_target(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, symbol: str):
    """
    Якщо ціна дійшла до TP/SL — закриваємо і шлемо повідомлення.
    """
    st = ensure_chat_state(chat_id)
    pos = st["positions"].get(symbol)
    if not pos:
        return
    side   = pos["side"]
    entry  = pos["entry"]
    tp     = pos["tp"]
    sl     = pos["sl"]
    qty    = pos["qty"]

    try:
        price = await get_last_price(symbol)
    except Exception:
        return

    reason = None
    if side == "long":
        if price >= tp:
            reason = "TP"
        elif price <= sl:
            reason = "SL"
    else:
        if price <= tp:
            reason = "TP"
        elif price >= sl:
            reason = "SL"

    if reason:
        try:
            _, exit_price = await close_position_market(symbol, qty, side)
        except Exception:
            # fallback — беремо спотову
            exit_price = price

        await notify_close(ctx, chat_id, symbol, side, entry, exit_price, reason)
        # чистимо
        st["positions"].pop(symbol, None)

# ================== BACKGROUND SCANNER ==================
async def scanner_loop(app: Application):
    """
    Примітивний фоновий цикл:
      - пробігає по чатах, по відкритих позиціях — перевіряє TP/SL
      - для демо: за командою /signals_crypto можна викликати open_position
    """
    await asyncio.sleep(3)
    while True:
        try:
            for chat_id, st in list(STATE.items()):
                # Перевірка TP/SL для відкритих позицій
                syms = list(st.get("positions", {}).keys())
                for s in syms:
                    try:
                        await maybe_close_on_target(app.bot, chat_id, s)
                    except Exception:
                        pass
        except Exception:
            pass
        await asyncio.sleep(5)  # частий монітор

# ================== TELEGRAM COMMANDS ==================
MAIN_KB = ReplyKeyboardMarkup(
    [
        ["/signals_crypto", "/alp_status"],
        ["/auto_on", "/auto_off"],
    ],
    resize_keyboard=True
)

async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ensure_chat_state(chat_id)
    await update.message.reply_text(
        "Привіт! Я готовий 🚀\n"
        "• /signals_crypto — демо-сканер (відкриє 3 позиції по $200)\n"
        "• /alp_status — статус і відкриті позиції\n"
        "• /auto_on, /auto_off — фоновий монітор TP/SL\n",
        reply_markup=MAIN_KB
    )

async def alp_status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = ensure_chat_state(chat_id)
    lines = [f"Mode={st.get('mode')}"]
    pos = st["positions"]
    if pos:
        lines.append("📌 Відкриті позиції:")
        for sym, p in pos.items():
            lines.append(f"• {sym} {p['side'].upper()}: "
                         f"entry {fmt_price(p['entry'])} · "
                         f"TP {fmt_price(p['tp'])} (+{fmt_pct(p['tp_pct'])}) · "
                         f"SL {fmt_price(p['sl'])} (-{fmt_pct(p['sl_pct'])}) · "
                         f"qty {p['qty']}")
    else:
        lines.append("Відкритих позицій немає.")
    await update.message.reply_text("\n".join(lines))

async def signals_crypto_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Демо: відкриває 3 позиції по $200 за списком CRYPTO_LIST.
    """
    chat_id = update.effective_chat.id
    st = ensure_chat_state(chat_id)
    await update.message.reply_text(
        "🛰️ Сканер (крипта): відкриємо по $200 для перших 3 символів",
    )
    # не створювати дубль, якщо вже відкрито
    limit = 3
    opened = 0
    for sym in CRYPTO_LIST:
        if opened >= limit:
            break
        if sym in st["positions"]:
            await update.message.reply_text(f"◻️ SKIP (позиція вже відкрита): {sym}")
            continue
        try:
            await open_position(ctx, chat_id, sym, "long", 200.0, TP_PCT_DEFAULT, SL_PCT_DEFAULT)
            opened += 1
        except Exception as e:
            await update.message.reply_text(f"🔴 ORDER FAIL {sym}: {e}")

async def auto_on_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    st = ensure_chat_state(update.effective_chat.id)
    st["auto"] = True
    await update.message.reply_text("✅ AUTO-SCAN: ON (монітор TP/SL кожні ~5с)")

async def auto_off_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    st = ensure_chat_state(update.effective_chat.id)
    st["auto"] = False
    await update.message.reply_text("⛔ AUTO-SCAN: OFF")

# ================== MAIN ==================
async def main():
    global HTTP
    HTTP = aiohttp.ClientSession(timeout=TIMEOUT)

    app = Application.builder().token(TG_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("alp_status", alp_status_cmd))
    app.add_handler(CommandHandler("signals_crypto", signals_crypto_cmd))
    app.add_handler(CommandHandler("auto_on", auto_on_cmd))
    app.add_handler(CommandHandler("auto_off", auto_off_cmd))

    # запуск фонового монітору
    app.job_queue.run_repeating(lambda *_: None, interval=3600, first=0)  # «якор» для JobQueue
    asyncio.create_task(scanner_loop(app))

    print("Bot started.")
    try:
        await app.run_polling(close_loop=False)
    finally:
        await HTTP.close()

if __name__ == "__main__":
    asyncio.run(main())
