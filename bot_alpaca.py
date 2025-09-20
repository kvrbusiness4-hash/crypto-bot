# -*- coding: utf-8 -*-
"""
bot_alpaca.py
- Telegram-бот (python-telegram-bot v20+)
- Alpaca Trading + Data
- Динамічні TP/SL для SCALP (5m/15m свічки)
- Нотифікації OPEN/CLOSE з цінами та %PnL
- Фоновий монітор TP/SL

ENV:
  TELEGRAM_BOT_TOKEN / TG_TOKEN
  ALPACA_API_KEY
  ALPACA_API_SECRET
  ALPACA_BASE_URL      (default: https://paper-api.alpaca.markets)
  ALPACA_DATA_URL      (default: https://data.alpaca.markets)
  SCAN_INTERVAL_SEC    (optional, default 300)
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, Tuple, List

import aiohttp
from aiohttp import ClientSession, ClientTimeout

from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes

# ============== ENV ==============
TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_TOKEN") or "").strip()

ALPACA_API_KEY = (os.getenv("ALPACA_API_KEY") or "").strip()
ALPACA_API_SECRET = (os.getenv("ALPACA_API_SECRET") or "").strip()

ALPACA_BASE_URL = (os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_URL = (os.getenv("ALPACA_DATA_URL") or "https://data.alpaca.markets").rstrip("/")

SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC") or 300)

TIMEOUT = ClientTimeout(total=30)
HTTP: Optional[ClientSession] = None

# ============== STATE ==============
# STATE[chat_id] = {
#   "mode": "scalp"|"aggressive"|"safe",
#   "positions": {
#       "AAVE/USD": {"side":"long","entry":..,"tp":..,"sl":..,"tp_pct":..,"sl_pct":..,"qty":..}
#   }
# }
STATE: Dict[int, Dict[str, Any]] = {}

# Демо-список крипти для /signals_crypto
CRYPTO_LIST = ["AAVE/USD", "AVAX/USD", "BAT/USD"]

# Базові пресети для режимів (ліше для не-скальпу TP/SL або як база)
MODE_PARAMS = {
    "aggressive": {"tp_pct": 0.015, "sl_pct": 0.008},
    "scalp":      {"tp_pct": 0.010, "sl_pct": 0.006},  # використовується як "база" для динаміки
    "safe":       {"tp_pct": 0.009, "sl_pct": 0.006},
}

# ============== HELPERS ==============
def ensure_chat_state(chat_id: int) -> Dict[str, Any]:
    st = STATE.setdefault(chat_id, {})
    st.setdefault("positions", {})
    st.setdefault("mode", "scalp")
    return st

def symbol_is_crypto(sym: str) -> bool:
    return "/" in sym

def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def fmt_price(p: float) -> str:
    if p >= 100:
        return f"{p:.4f}"
    elif p >= 1:
        return f"{p:.5f}"
    else:
        return f"{p:.8f}"

def alpaca_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }

async def alpaca_get(url: str, params: Optional[dict] = None) -> Any:
    async with HTTP.get(url, headers=alpaca_headers(), params=params) as r:
        txt = await r.text()
        if r.status >= 400:
            raise RuntimeError(f"GET {url} {r.status}: {txt}")
        try:
            return json.loads(txt) if txt else {}
        except Exception:
            return txt

async def alpaca_post(url: str, payload: dict) -> Any:
    async with HTTP.post(url, headers=alpaca_headers(), data=json.dumps(payload)) as r:
        txt = await r.text()
        if r.status >= 400:
            raise RuntimeError(f"POST {url} {r.status}: {txt}")
        try:
            return json.loads(txt) if txt else {}
        except Exception:
            return txt

# ============== DATA HELPERS ==============
async def get_last_price(symbol: str) -> float:
    if symbol_is_crypto(symbol):
        url = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/trades/latest"
        data = await alpaca_get(url, params={"symbols": symbol})
        return float((data.get("trades", {}).get(symbol) or {}).get("p", 0.0))
    else:
        url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest"
        data = await alpaca_get(url)
        return float((data.get("trade") or {}).get("p", 0.0))

async def get_recent_bars(symbol: str, timeframe: str = "5Min", limit: int = 50) -> List[dict]:
    """Повертає список барів (ASC) з ключами o,h,l,c."""
    if symbol_is_crypto(symbol):
        url = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/bars"
        data = await alpaca_get(url, params={
            "symbols": symbol,
            "timeframe": timeframe,
            "limit": str(limit),
            "sort": "asc",
        })
        return (data.get("bars", {}).get(symbol) or [])
    else:
        url = f"{ALPACA_DATA_URL}/v2/stocks/bars"
        data = await alpaca_get(url, params={
            "symbols": symbol,
            "timeframe": timeframe,
            "limit": str(limit),
            "sort": "asc",
        })
        return (data.get("bars", {}).get(symbol) or [])

def _safe_f(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return d

def candle_body_pct(bar: dict) -> float:
    """(close-open)/open"""
    o = _safe_f(bar.get("o"))
    c = _safe_f(bar.get("c"))
    if o <= 0: return 0.0
    return (c - o) / o

def candle_range_pct(bar: dict) -> float:
    """(high-low)/open"""
    o = _safe_f(bar.get("o"))
    h = _safe_f(bar.get("h"))
    l = _safe_f(bar.get("l"))
    if o <= 0: return 0.0
    return (h - l) / o

def mini_atr_pct(bars: List[dict], n: int = 7) -> float:
    """Проста оцінка ATR%: середня (high-low)/open за останні n барів."""
    if not bars: return 0.0
    arr = bars[-n:]
    vals = [max(0.0, candle_range_pct(b)) for b in arr if b]
    if not vals: return 0.0
    return sum(vals) / len(vals)

# ============== DYNAMIC TP/SL FOR SCALP ==============
async def dynamic_tp_sl_scalp(symbol: str, base_tp: float, base_sl: float) -> Tuple[float, float]:
    """
    Повертає (tp_pct, sl_pct) для SCALP залежно від сили імпульсу на 5m і підтвердження на 15m.
    Логіка:
      - body% останньої 5m свічки → імпульс
      - mini-ATR%(5m,7) → волатильність
      - 15m остання свічка того ж напрямку → +підсилення
    Мапінг у межі: TP ∈ [0.003, 0.02], SL ≈ TP*0.6..0.9
    """
    try:
        bars5 = await get_recent_bars(symbol, "5Min", 30)
        bars15 = await get_recent_bars(symbol, "15Min", 30)
    except Exception:
        # фолбек: база
        return base_tp, base_sl

    if not bars5:
        return base_tp, base_sl

    last5 = bars5[-1]
    body5 = abs(candle_body_pct(last5))          # сила свічки
    atr5  = mini_atr_pct(bars5, n=7)             # міні-ATR
    rng5  = candle_range_pct(last5)

    # напрямок свічки 5m
    dir5_up = (_safe_f(last5.get("c")) >= _safe_f(last5.get("o")))

    confirm15 = 0.0
    if bars15:
        last15 = bars15[-1]
        dir15_up = (_safe_f(last15.get("c")) >= _safe_f(last15.get("o")))
        confirm15 = 0.2 if dir15_up == dir5_up else -0.1  # невелике підсилення/штраф

    # сирий скор
    score = (
        body5 * 3.0 +       # тіло — найважливіше
        atr5 * 1.5 +        # волатильність
        rng5 * 0.5 +        # діапазон свічки
        confirm15           # підтвердження на 15m
    )
    # типові масштаби body5 ~ 0.001..0.01, atr5 ~ 0.003..0.02

    # Мапінг score -> TP%
    # мінімум/максимум
    tp_min, tp_max = 0.003, 0.020
    # нормалізуємо score (жорстке обрізання, щоб не вилітало)
    # приблизно: score 0.002 -> низький, 0.015 -> сильний
    s = max(0.0, min(score, 0.02)) / 0.02  # 0..1
    tp_pct = tp_min + s * (tp_max - tp_min)

    # SL як частка TP: у скальпі SL ближчий
    # якщо імпульс сильний (s → 1), даємо SL трошки ширше (0.85), інакше 0.6
    w = 0.6 + 0.25 * s
    sl_pct = tp_pct * w

    # Запобіжники: якщо база більша/менша — можемо злегка підтягнути до бази
    # (щоб не було занадто дрібних або занадто великих при мертвому ринку)
    tp_pct = max(min(tp_pct, max(0.8*base_tp, tp_max)), min(0.6*base_tp, tp_min))
    sl_pct = max(sl_pct, 0.5 * tp_pct)

    return tp_pct, sl_pct

# ============== NOTIFICATIONS ==============
async def notify_open(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, symbol: str, side: str,
                      entry_price: float, tp_price: float, sl_price: float,
                      tp_pct: float, sl_pct: float, qty: float):
    msg = (f"🟢 ORDER OK: {symbol} {side.upper()} qty≈{qty}\n"
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

# ============== TRADING ==============
async def place_market_order(symbol: str, notional_usd: float, side: str) -> Tuple[str, float, float]:
    """
    MARKET notional; повертає (order_id, filled_qty, filled_avg_price)
    """
    url = f"{ALPACA_BASE_URL}/v2/orders"
    payload = {
        "symbol": symbol.replace("/", ""),
        "notional": round(notional_usd, 2),
        "side": "buy" if side.lower() in ("buy", "long") else "sell",
        "type": "market",
        "time_in_force": "gtc",
        "asset_class": "crypto" if symbol_is_crypto(symbol) else "us_equity",
    }
    if symbol_is_crypto(symbol):
        payload["symbol"] = symbol.replace("/", "")

    data = await alpaca_post(url, payload)
    oid = data.get("id", "")
    qty = float(data.get("filled_qty") or 0.0) if data.get("filled_qty") else 0.0
    avg = float(data.get("filled_avg_price") or 0.0) if data.get("filled_avg_price") else 0.0

    if avg <= 0.0:
        await asyncio.sleep(0.7)
        od = await alpaca_get(f"{ALPACA_BASE_URL}/v2/orders/{oid}")
        qty = float(od.get("filled_qty") or 0.0) if od.get("filled_qty") else qty
        avg = float(od.get("filled_avg_price") or 0.0) if od.get("filled_avg_price") else avg
    if avg <= 0.0:
        avg = await get_last_price(symbol)
    if qty <= 0.0 and avg > 0:
        qty = round(notional_usd / avg, 6)
    return oid, qty, avg

async def close_position_market(symbol: str, qty: float, side: str) -> Tuple[str, float]:
    """
    Закриття маркетом. Повертає (order_id, exit_price).
    """
    exit_side = "sell" if side.lower() == "long" else "buy"
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
    avg = float(data.get("filled_avg_price") or 0.0) if data.get("filled_avg_price") else 0.0
    if avg <= 0.0:
        await asyncio.sleep(0.7)
        od = await alpaca_get(f"{ALPACA_BASE_URL}/v2/orders/{oid}")
        avg = float(od.get("filled_avg_price") or 0.0) if od.get("filled_avg_price") else avg
    if avg <= 0.0:
        avg = await get_last_price(symbol)
    return oid, avg

# ============== POSITION MGMT ==============
async def open_position(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, symbol: str,
                        side: str, notional_usd: float):
    """
    Відкрити позицію MARKET і визначити TP/SL:
      - якщо режим SCALP → динамічні % від 5m/15m
      - інакше → з MODE_PARAMS для поточного режиму
    """
    st = ensure_chat_state(chat_id)
    mode = st.get("mode", "scalp")
    base = MODE_PARAMS.get(mode, MODE_PARAMS["scalp"])

    # 1) Вхід
    try:
        _, qty, entry_price = await place_market_order(symbol, notional_usd, side)
    except Exception as e:
        await ctx.bot.send_message(chat_id, f"🔴 ORDER FAIL {symbol}: {e}")
        return

    # 2) Обчислення TP/SL %
    if mode == "scalp":
        tp_pct, sl_pct = await dynamic_tp_sl_scalp(symbol, base["tp_pct"], base["sl_pct"])
    else:
        tp_pct, sl_pct = base["tp_pct"], base["sl_pct"]

    # 3) Ціни TP/SL
    is_long = side.lower() in ("buy", "long")
    if is_long:
        tp_price = entry_price * (1.0 + tp_pct)
        sl_price = entry_price * (1.0 - sl_pct)
    else:
        tp_price = entry_price * (1.0 - tp_pct)
        sl_price = entry_price * (1.0 + sl_pct)

    # 4) Зберегти в STATE
    st["positions"][symbol] = {
        "side": "long" if is_long else "short",
        "entry": float(entry_price),
        "tp": float(tp_price),
        "sl": float(sl_price),
        "tp_pct": float(tp_pct),
        "sl_pct": float(sl_pct),
        "qty": float(qty),
    }

    # 5) Нотифікація OPEN
    await notify_open(ctx, chat_id, symbol, "buy" if is_long else "sell",
                      entry_price, tp_price, sl_price, tp_pct, sl_pct, qty)

async def maybe_close_on_target(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, symbol: str):
    """Закриває позицію при досягненні TP/SL і надсилає нотифікацію."""
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
        if price >= tp: reason = "TP"
        elif price <= sl: reason = "SL"
    else:
        if price <= tp: reason = "TP"
        elif price >= sl: reason = "SL"

    if reason:
        try:
            _, exit_price = await close_position_market(symbol, qty, side)
        except Exception:
            exit_price = price
        await notify_close(ctx, chat_id, symbol, side, entry, exit_price, reason)
        st["positions"].pop(symbol, None)

# ============== BACKGROUND MONITOR ==============
async def scanner_loop(app: Application):
    await asyncio.sleep(3)
    while True:
        try:
            for chat_id, st in list(STATE.items()):
                # якщо немає позицій — пропускаємо
                syms = list(st.get("positions", {}).keys())
                for s in syms:
                    try:
                        await maybe_close_on_target(app.bot, chat_id, s)
                    except Exception:
                        pass
        except Exception:
            pass
        await asyncio.sleep(5)

# ============== TELEGRAM COMMANDS ==============
MAIN_KB = ReplyKeyboardMarkup(
    [
        ["/signals_crypto", "/alp_status"],
        ["/scalp", "/aggressive", "/safe"],
    ],
    resize_keyboard=True
)

async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ensure_chat_state(chat_id)
    await update.message.reply_text(
        "Готово! 🚀\n"
        "• /signals_crypto — демо (відкриє до 3 позицій по $200)\n"
        "• /alp_status — статус/позиції\n"
        "• /scalp /aggressive /safe — перемкнути режим (TP/SL беруться з режиму; для SCALP — динамічно)\n",
        reply_markup=MAIN_KB
    )

async def alp_status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = ensure_chat_state(chat_id)
    lines = [f"Mode={st.get('mode', 'scalp')}"]
    pos = st["positions"]
    if pos:
        lines.append("📌 Відкриті позиції:")
        for sym, p in pos.items():
            lines.append(
                f"• {sym} {p['side'].upper()}: "
                f"entry {fmt_price(p['entry'])} · "
                f"TP {fmt_price(p['tp'])} (+{fmt_pct(p['tp_pct'])}) · "
                f"SL {fmt_price(p['sl'])} (-{fmt_pct(p['sl_pct'])}) · "
                f"qty {p['qty']}"
            )
    else:
        lines.append("Відкритих позицій немає.")
    await update.message.reply_text("\n".join(lines))

async def set_mode_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE, mode: str):
    chat_id = update.effective_chat.id
    st = ensure_chat_state(chat_id)
    st["mode"] = mode
    await update.message.reply_text(f"Режим встановлено: {mode.upper()}")

async def scalp_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await set_mode_cmd(update, ctx, "scalp")

async def aggressive_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await set_mode_cmd(update, ctx, "aggressive")

async def safe_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await set_mode_cmd(update, ctx, "safe")

async def signals_crypto_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = ensure_chat_state(chat_id)
    await update.message.reply_text("🛰️ Сканер (крипта): відкриємо до 3 позицій по $200")
    opened = 0
    for sym in CRYPTO_LIST:
        if opened >= 3:
            break
        if sym in st["positions"]:
            await update.message.reply_text(f"◻️ SKIP (позиція вже відкрита): {sym}")
            continue
        try:
            await open_position(ctx, chat_id, sym, "long", 200.0)
            opened += 1
        except Exception as e:
            await update.message.reply_text(f"🔴 ORDER FAIL {sym}: {e}")

# ============== MAIN ==============
async def main():
    global HTTP
    HTTP = aiohttp.ClientSession(timeout=TIMEOUT)

    app = Application.builder().token(TG_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("alp_status", alp_status_cmd))
    app.add_handler(CommandHandler("signals_crypto", signals_crypto_cmd))

    app.add_handler(CommandHandler("scalp", scalp_cmd))
    app.add_handler(CommandHandler("aggressive", aggressive_cmd))
    app.add_handler(CommandHandler("safe", safe_cmd))

    # «якор» для JobQueue (не використовуємо тут, але хай ініціалізується)
    app.job_queue.run_repeating(lambda *_: None, interval=3600, first=0)

    # фон. монітор TP/SL
    asyncio.create_task(scanner_loop(app))

    print("Bot started.")
    try:
        await app.run_polling(close_loop=False)
    finally:
        await HTTP.close()

if __name__ == "__main__":
    asyncio.run(main())
