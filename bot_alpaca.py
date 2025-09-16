# bot_alpaca.py
# -*- coding: utf-8 -*-

import os
import math
import time
import uuid
import asyncio
from typing import Any, Dict, List, Tuple

from aiohttp import ClientSession, ClientTimeout

from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes,
)

# ========= ENV =========
TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()

ALPACA_API_KEY = (os.getenv("ALPACA_API_KEY") or "").strip()
ALPACA_API_SECRET = (os.getenv("ALPACA_API_SECRET") or "").strip()
ALPACA_BASE_URL = (os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").strip()
ALPACA_DATA_URL = (os.getenv("ALPACA_DATA_URL") or "https://data.alpaca.markets").strip()

ALPACA_NOTIONAL = float(os.getenv("ALPACA_NOTIONAL") or 25)
ALPACA_TOP_N = int(os.getenv("ALPACA_TOP_N") or 3)

# ========= GLOBAL STATE =========
STATE: Dict[int, Dict[str, Any]] = {}  # per-chat

def stedef(chat_id: int) -> Dict[str, Any]:
    if chat_id not in STATE:
        STATE[chat_id] = {
            "autotrade": False,
            "mode": "aggressive",
            "top_n": ALPACA_TOP_N,
        }
    return STATE[chat_id]

# ========= MODES (таймфрейми, фільтри, ризик) =========
MODE_PARAMS = {
    "aggressive": {
        "bars": ("15Min", "30Min", "1Hour"),
        "rsi_buy": 55.0,
        "ema_fast": 15,
        "ema_slow": 30,
        "top_n": 3,
        "tp_pct": 0.015,   # 1.5%
        "sl_pct": 0.008,   # 0.8%
    },
    "scalp": {
        "bars": ("5Min", "15Min", "1Hour"),
        "rsi_buy": 58.0,
        "ema_fast": 12,
        "ema_slow": 26,
        "top_n": 3,
        "tp_pct": 0.010,
        "sl_pct": 0.006,
    },
    "default": {
        "bars": ("15Min", "30Min", "1Hour"),
        "rsi_buy": 57.0,
        "ema_fast": 20,
        "ema_slow": 50,
        "top_n": 3,
        "tp_pct": 0.012,
        "sl_pct": 0.007,
    },
    "swing": {
        "bars": ("30Min", "1Hour", "1Hour"),
        "rsi_buy": 60.0,
        "ema_fast": 20,
        "ema_slow": 60,
        "top_n": 2,
        "tp_pct": 0.020,
        "sl_pct": 0.010,
    },
    "safe": {
        "bars": ("30Min", "1Hour", "1Hour"),
        "rsi_buy": 62.0,
        "ema_fast": 20,
        "ema_slow": 60,
        "top_n": 2,
        "tp_pct": 0.015,
        "sl_pct": 0.008,
    },
}

# ========= SYMBOLS =========
CRYPTO_USD_PAIRS = [
    "BTC/USD","ETH/USD","SOL/USD","LTC/USD","DOGE/USD","AVAX/USD","AAVE/USD",
    "MKR/USD","DOT/USD","LINK/USD","UNI/USD","PEPE/USD","XRP/USD","TRUMP/USD",
    "CRV/USD","BCH/USD","BAT/USD","GRT/USD","XTZ/USD","USDC/USD","USDT/USD",
    "USDG/USD","YFI/USD","LDO/USD",
]

# ========= UTILS =========
def alp_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def to_alpaca_symbol(sym_slash: str) -> str:
    return sym_slash.replace("/", "")

# ----- RSI / EMA -----
def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1:
        return series or []
    k = 2 / (period + 1.0)
    out = [series[0]]
    for x in series[1:]:
        out.append(x * k + out[-1] * (1.0 - k))
    return out

def rsi(series: List[float], period: int = 14) -> List[float]:
    if len(series) <= period:
        return [50.0] * len(series)
    gains, losses = [], []
    for i in range(1, period+1):
        ch = series[i] - series[i-1]
        gains.append(max(0, ch))
        losses.append(max(0, -ch))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    rs = avg_gain / (avg_loss if avg_loss != 0 else 1e-9)
    out = [0.0]*(period) + [100 - 100/(1+rs)]

    for i in range(period+1, len(series)):
        ch = series[i] - series[i-1]
        gain = max(0, ch)
        loss = max(0, -ch)
        avg_gain = (avg_gain*(period-1) + gain) / period
        avg_loss = (avg_loss*(period-1) + loss) / period
        rs = avg_gain / (avg_loss if avg_loss != 0 else 1e-9)
        out.append(100 - 100/(1+rs))
    return out

# ========= DATA =========
async def get_bars_crypto(pairs: List[str], timeframe: str, limit: int) -> Dict[str, Any]:
    """Alpaca DATA v1beta3"""
    # ліміт для 60Min — 59 (API обмеження)
    if timeframe == "60Min" or timeframe == "1Hour":
        timeframe = "60Min"
        limit = min(limit, 59)

    syms = ",".join([to_alpaca_symbol(x) for x in pairs])
    url = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/bars?symbols={syms}&timeframe={timeframe}&limit={limit}&sort=asc"
    async with ClientSession(timeout=ClientTimeout(total=12)) as s:
        async with s.get(url, headers=alp_headers()) as r:
            txt = await r.text()
            if r.status != 200:
                raise RuntimeError(f"GET {url} {r.status}: {txt}")
            return await r.json()

def extract_bars(raw: Dict[str, Any], sym: str) -> List[Dict[str, Any]]:
    alp = to_alpaca_symbol(sym)
    d = (raw.get("bars") or {}).get(alp) or []
    return d

# ========= RANK & SCAN =========
def rank_score(c15: List[float], c30: List[float], c60: List[float], conf: Dict[str, Any]) -> float:
    if not c15 or not c30 or not c60:
        return -1e9

    r1 = rsi(c15, 14)[-1]
    r2 = rsi(c30, 14)[-1]
    r3 = rsi(c60, 14)[-1]

    ef = ema(c60, conf["ema_fast"])[-1]
    es = ema(c60, conf["ema_slow"])[-1]
    trend = 0.0
    if es != 0:
        trend = max(-1.0, min(1.0, (ef - es) / abs(es)))

    bias_long = 0
    bias_long += 1 if r1 >= conf["rsi_buy"] else 0
    bias_long += 1 if r2 >= conf["rsi_buy"] else 0
    bias_long += 1 if r3 >= conf["rsi_buy"] else 0

    # базовий скор
    return bias_long*100 + trend*50 - abs(50.0 - r1)

async def scan_rank_crypto(st: Dict[str, Any]) -> Tuple[str, List[Tuple[float, str, List[Dict[str, Any]]]]]:
    conf = MODE_PARAMS[st.get("mode", "aggressive")]
    tf15, tf30, tf60 = conf["bars"]

    pairs = CRYPTO_USD_PAIRS[:]  # фіксований whitelist

    bars15 = await get_bars_crypto(pairs, tf15, 120)
    bars30 = await get_bars_crypto(pairs, tf30, 120)
    bars60 = await get_bars_crypto(pairs, tf60, 59)

    ranked: List[Tuple[float, str, List[Dict[str, Any]]]] = []
    for sym in pairs:
        raw15 = extract_bars(bars15, sym)
        raw30 = extract_bars(bars30, sym)
        raw60 = extract_bars(bars60, sym)
        if not raw15 or not raw30 or not raw60:
            continue

        c15 = [float(x["c"]) for x in raw15]
        c30 = [float(x["c"]) for x in raw30]
        c60 = [float(x["c"]) for x in raw60]

        s = rank_score(c15, c30, c60, conf)
        ranked.append((s, sym, raw15))  # raw15 для SL/TP

    ranked.sort(reverse=True)
    report = (
        "🛰️ Сканер (крипта):\n"
        f"• Активних USD-пар: {len(pairs)}\n"
        f"• Використаємо для торгівлі (лімітом): {min(conf['top_n'], len(ranked))}\n"
        f"• Перші 25: " + ", ".join([s for _, s, _ in ranked[:25]]) if ranked else "Немає сигналів"
    )
    return report, ranked

# ========= SL/TP =========
def calc_sl_tp(side: str, px: float, highs: List[float], lows: List[float], closes: List[float], st: Dict[str, Any]) -> Tuple[float, float]:
    conf = MODE_PARAMS[st.get("mode", "aggressive")]
    tp_pct = conf["tp_pct"]
    sl_pct = conf["sl_pct"]
    if side == "buy":
        tp = px * (1 + tp_pct)
        sl = px * (1 - sl_pct)
    else:
        tp = px
        sl = px
    return sl, tp

# ========= ANTI-DUPLICATE / POSITIONS =========
RECENT_TRADES: Dict[str, float] = {}
TRADE_COOLDOWN_SEC = 120
TRADE_LOCK = asyncio.Lock()

def anti_duplicate(symbol_slash: str) -> bool:
    now = time.time()
    last = RECENT_TRADES.get(symbol_slash, 0)
    if now - last < TRADE_COOLDOWN_SEC:
        return False
    RECENT_TRADES[symbol_slash] = now
    return True

async def get_positions_map() -> Dict[str, float]:
    url = f"{ALPACA_BASE_URL}/v2/positions"
    async with ClientSession(timeout=ClientTimeout(total=10)) as s:
        async with s.get(url, headers=alp_headers()) as r:
            if r.status != 200:
                return {}
            data = await r.json()
    out = {}
    for p in data:
        sym = p.get("symbol", "")
        qty = float(p.get("qty", "0") or 0)
        out[sym] = qty
    return out

# ========= ORDERS =========
async def safe_place_bracket(symbol_slash: str, notional: float, tp: float, sl: float) -> Tuple[bool, str]:
    if not anti_duplicate(symbol_slash):
        return False, f"skip duplicate for {symbol_slash}"

    positions = await get_positions_map()
    alp_sym = to_alpaca_symbol(symbol_slash)
    if positions.get(alp_sym, 0.0) > 0:
        return False, f"skip: already long {symbol_slash}"

    payload = {
        "symbol": alp_sym,
        "side": "buy",
        "type": "market",
        "time_in_force": "gtc",
        "notional": f"{notional:.2f}",
        "client_order_id": f"psb-{alp_sym}-{uuid.uuid4().hex[:10]}",
        "take_profit": {"limit_price": f"{tp:.6f}"},
        "stop_loss":  {"stop_price":  f"{sl:.6f}"},
    }

    url = f"{ALPACA_BASE_URL}/v2/orders"
    async with ClientSession(timeout=ClientTimeout(total=12)) as s:
        async with s.post(url, json=payload, headers=alp_headers()) as r:
            txt = await r.text()
            if r.status in (200, 201):
                return True, f"ORDER OK: {symbol_slash} BUY ${notional:.2f}"
            return False, f"ORDER FAIL {symbol_slash}: {r.status}: {txt}"

# ========= TELEGRAM CMDS =========
KB = ReplyKeyboardMarkup(
    [
        ["/aggressive", "/scalp", "/default"],
        ["/swing", "/safe", "/help"],
        ["/signals_crypto", "/trade_crypto"],
        ["/alp_on", "/alp_status", "/alp_off"],
    ],
    resize_keyboard=True
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(update.effective_chat.id)
    await update.message.reply_text(
        "Привіт! Це ProfitSignalsBot (Alpaca, crypto). Лише LONG.\n"
        "• /signals_crypto — скан без торгівлі\n"
        "• /trade_crypto — миттєвий трейд ТОП-N з TP/SL\n"
        "• /alp_on /alp_off /alp_status — автотрейд\n"
        "• режими: /aggressive /scalp /default /swing /safe",
        reply_markup=KB
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start(update, context)

async def mode_set(update: Update, context: ContextTypes.DEFAULT_TYPE, mode: str) -> None:
    st = stedef(update.effective_chat.id)
    st["mode"] = mode
    await update.message.reply_text(f"Режим: {mode.upper()}")

async def aggressive(update, ctx): await mode_set(update, ctx, "aggressive")
async def scalp(update, ctx):      await mode_set(update, ctx, "scalp")
async def default(update, ctx):    await mode_set(update, ctx, "default")
async def swing(update, ctx):      await mode_set(update, ctx, "swing")
async def safe(update, ctx):       await mode_set(update, ctx, "safe")

async def alp_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(update.effective_chat.id)
    st["autotrade"] = True
    await update.message.reply_text("✅ Alpaca AUTOTRADE: ON")

async def alp_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(update.effective_chat.id)
    st["autotrade"] = False
    await update.message.reply_text("⛔ Alpaca AUTOTRADE: OFF")

async def alp_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(update.effective_chat.id)
    # простий ping балансу (не обов’язково)
    await update.message.reply_text(
        f"Alpaca:\n"
        f"• Mode={st.get('mode')} · Autotrade={'ON' if st.get('autotrade') else 'OFF'}\n"
        f"• Side=LONG"
    )

async def signals_crypto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(update.effective_chat.id)
    try:
        rep, ranked = await scan_rank_crypto(st)
        await update.message.reply_text(rep)
    except Exception as e:
        await update.message.reply_text(f"🔴 signals error: {e}")

async def trade_crypto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(update.effective_chat.id)
    try:
        async with TRADE_LOCK:
            rep, ranked = await scan_rank_crypto(st)
            await update.message.reply_text(rep)
            if not ranked:
                return
            picks = ranked[:MODE_PARAMS[st["mode"]]["top_n"]]
            for _, sym, arr in picks:
                h = [float(x["h"]) for x in arr]
                l = [float(x["l"]) for x in arr]
                c = [float(x["c"]) for x in arr]
                px = c[-1]
                sl, tp = calc_sl_tp("buy", px, h, l, c, st)
                ok, msg = await safe_place_bracket(sym, ALPACA_NOTIONAL, tp, sl)
                await update.message.reply_text(("🟢 " if ok else "🔴 ") + msg)
    except Exception as e:
        await update.message.reply_text(f"🔴 trade_crypto error: {e}")

# ========= PERIODIC JOB (автотрейд) =========
async def periodic_scan_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        async with TRADE_LOCK:
            for chat_id, st in list(STATE.items()):
                if not st.get("autotrade"):
                    continue
                try:
                    rep, ranked = await scan_rank_crypto(st)
                    await ctx.bot.send_message(chat_id=chat_id, text=rep)
                    if not ranked:
                        continue
                    picks = ranked[:MODE_PARAMS[st["mode"]]["top_n"]]
                    for _, sym, arr in picks:
                        h = [float(x["h"]) for x in arr]
                        l = [float(x["l"]) for x in arr]
                        c = [float(x["c"]) for x in arr]
                        px = c[-1]
                        sl, tp = calc_sl_tp("buy", px, h, l, c, st)
                        ok, msg = await safe_place_bracket(sym, ALPACA_NOTIONAL, tp, sl)
                        await ctx.bot.send_message(chat_id=chat_id, text=("🟢 " if ok else "🔴 ")+msg)
                except Exception as e:
                    await ctx.bot.send_message(chat_id=chat_id, text=f"🔴 periodic error: {e}")
    except Exception:
        pass

# ========= MAIN =========
def main() -> None:
    app = Application.builder().token(TG_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))

    app.add_handler(CommandHandler("aggressive", aggressive))
    app.add_handler(CommandHandler("scalp", scalp))
    app.add_handler(CommandHandler("default", default))
    app.add_handler(CommandHandler("swing", swing))
    app.add_handler(CommandHandler("safe", safe))

    app.add_handler(CommandHandler("alp_on", alp_on))
    app.add_handler(CommandHandler("alp_off", alp_off))
    app.add_handler(CommandHandler("alp_status", alp_status))

    app.add_handler(CommandHandler("signals_crypto", signals_crypto))
    app.add_handler(CommandHandler("trade_crypto", trade_crypto))

    # періодичний джоб кожні 5 хв
    app.job_queue.run_repeating(periodic_scan_job, interval=300, first=30)

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
