# bot_alpaca.py
# -*- coding: utf-8 -*-

import os
import json
import math
import asyncio
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, timedelta, timezone

from aiohttp import ClientSession, ClientTimeout

from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes, CallbackContext
)

# ========= ENV =========
TG_TOKEN         = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_TOKEN") or "").strip()

ALPACA_API_KEY   = (os.getenv("ALPACA_API_KEY") or "").strip()
ALPACA_API_SECRET= (os.getenv("ALPACA_API_SECRET") or "").strip()
ALPACA_BASE_URL  = (os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").strip()
ALPACA_DATA_URL  = (os.getenv("ALPACA_DATA_URL") or "https://data.alpaca.markets").strip()

ALPACA_NOTIONAL  = float(os.getenv("ALPACA_NOTIONAL") or 25)
ALPACA_TOP_N     = int(os.getenv("ALPACA_TOP_N") or 4)

SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC") or 300)   # 5 хв для автотрейду
DUP_COOLDOWN_MIN  = int(os.getenv("DUP_COOLDOWN_MIN") or 15)     # блок дубляжу ордерів

# ===== Timeframe normalization (Alpaca accepts 1Min,5Min,15Min,30Min,1Hour,1Day) =====
ALPACA_TF_CANON = {
    "1MIN":"1Min","1min":"1Min","1Min":"1Min",
    "5MIN":"5Min","5min":"5Min","5Min":"5Min",
    "15MIN":"15Min","15min":"15Min","15Min":"15Min",
    "30MIN":"30Min","30min":"30Min","30Min":"30Min",
    "60MIN":"1Hour","60min":"1Hour","60Min":"1Hour","1H":"1Hour","1h":"1Hour","1HOUR":"1Hour","1Hour":"1Hour",
    "1DAY":"1Day","1day":"1Day","1D":"1Day","1d":"1Day","1Day":"1Day",
}
def normalize_tf(tf: str) -> str:
    key = tf.strip()
    return ALPACA_TF_CANON.get(key, ALPACA_TF_CANON.get(key.upper(), key))

def normalize_bars_tuple(bars_tuple: Tuple[str, str, str]) -> Tuple[str, str, str]:
    return tuple(normalize_tf(x) for x in bars_tuple)  # type: ignore


# ========= MODE PROFILES (таймфрейми, фільтри, ризик) =========
MODE_PARAMS: Dict[str, Dict[str, Any]] = {
    "aggressive": {
        "bars": ("15Min", "30Min", "1Hour"),
        "rsi_buy": 55.0,   # LONG: RSI вище
        "rsi_sell": 45.0,  # SHORT: (не застосовується для кріпти)
        "ema_fast": 15, "ema_slow": 30,
        "top_n": 4,
        "tp_pct": 0.015,   # 1.5% TP
        "sl_pct": 0.008,   # 0.8% SL
    },
    "scalp": {
        "bars": ("5Min", "15Min", "1Hour"),
        "rsi_buy": 58.0, "rsi_sell": 42.0,
        "ema_fast": 12, "ema_slow": 26,
        "top_n": 3,
        "tp_pct": 0.010,
        "sl_pct": 0.006,
    },
    "default": {
        "bars": ("15Min", "30Min", "1Hour"),
        "rsi_buy": 56.0, "rsi_sell": 44.0,
        "ema_fast": 14, "ema_slow": 28,
        "top_n": 3,
        "tp_pct": 0.012,
        "sl_pct": 0.007,
    },
    "swing": {
        "bars": ("30Min", "1Hour", "1Day"),
        "rsi_buy": 55.0, "rsi_sell": 45.0,
        "ema_fast": 20, "ema_slow": 50,
        "top_n": 3,
        "tp_pct": 0.020,
        "sl_pct": 0.010,
    },
    "safe": {
        "bars": ("30Min", "1Hour", "1Day"),
        "rsi_buy": 60.0, "rsi_sell": 40.0,
        "ema_fast": 20, "ema_slow": 50,
        "top_n": 2,
        "tp_pct": 0.015,
        "sl_pct": 0.008,
    },
}

# ======= CRYPTO USD pairs (whitelist) =======
CRYPTO_USD_PAIRS: List[str] = [
    # Постав тільки ті, що доступні у твоєму регіоні/акаунті Alpaca
    "BTC/USD","ETH/USD","SOL/USD","LTC/USD","DOGE/USD","AVAX/USD","AAVE/USD","MKR/USD",
    "DOT/USD","LINK/USD","UNI/USD","PEPE/USD","XRP/USD","TRUMP/USD","CRV/USD","BCH/USD",
    "BAT/USD","GRT/USD","XTZ/USD","USDC/USD","USDT/USD","USDG/USD","YFI/USD","LDO/USD",
]

# ========= STATE (профілі користувачів і антидубль) =========
STATE: Dict[int, Dict[str, Any]] = {}
GLOBAL_RECENT_ORDERS: Dict[str, datetime] = {}  # key = f"{symbol}|{side}"

DEFAULT_MODE = "aggressive"
DEFAULT_SIDE_MODE = "long"   # тільки LONG для кріпти

# ========= Utils =========
def stedef(chat_id: int) -> Dict[str, Any]:
    st = STATE.setdefault(chat_id, {})
    st.setdefault("mode", DEFAULT_MODE)
    st.setdefault("autotrade", True)
    st.setdefault("side_mode", DEFAULT_SIDE_MODE)  # long|short|both (short буде ігноруватись)
    return st

def mode_conf(st: Dict[str, Any]) -> Dict[str, Any]:
    prof = st.get("mode", DEFAULT_MODE)
    conf = dict(MODE_PARAMS.get(prof, MODE_PARAMS[DEFAULT_MODE]))
    # гарантуємо нормалізацію ТФ
    conf["bars"] = normalize_bars_tuple(conf["bars"])
    return conf

def is_crypto_pair(sym: str) -> bool:
    return "/" in sym and sym.endswith("/USD")

def rsi(values: List[float], period: int) -> float:
    if len(values) < period + 1:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(-period, 0):
        change = values[i] - values[i-1]
        if change >= 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period if losses > 0 else 1e-9
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def ema(series: List[float], period: int) -> List[float]:
    if not series:
        return []
    k = 2.0 / (period + 1.0)
    out = [series[0]]
    for x in series[1:]:
        out.append(out[-1] + k * (x - out[-1]))
    return out

async def fetch_json(session: ClientSession, url: str, headers: Dict[str, str]) -> Any:
    async with session.get(url, headers=headers, timeout=ClientTimeout(total=30)) as r:
        if r.status >= 400:
            txt = await r.text()
            raise RuntimeError(f"GET {url} {r.status}: {txt}")
        return await r.json()

async def post_json(session: ClientSession, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Any:
    async with session.post(url, headers=headers, json=payload, timeout=ClientTimeout(total=30)) as r:
        txt = await r.text()
        if r.status >= 400:
            raise RuntimeError(f"POST {url} {r.status}: {txt}")
        try:
            return json.loads(txt)
        except Exception:
            return {"raw": txt}

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def dup_key(symbol: str, side: str) -> str:
    return f"{symbol}|{side}"

def is_recently_traded(symbol: str, side: str) -> bool:
    key = dup_key(symbol, side)
    ts = GLOBAL_RECENT_ORDERS.get(key)
    if not ts:
        return False
    return (now_utc() - ts) < timedelta(minutes=DUP_COOLDOWN_MIN)

def mark_traded(symbol: str, side: str) -> None:
    GLOBAL_RECENT_ORDERS[dup_key(symbol, side)] = now_utc()

# ========= Market data =========
def bars_url(symbols: List[str], timeframe: str, limit: int = 120) -> str:
    syms = ",".join(symbols)
    return (
        f"{ALPACA_DATA_URL}/v1beta3/crypto/us/bars?"
        f"symbols={syms}&timeframe={timeframe}&limit={limit}&sort=asc"
    )

async def get_bars_crypto(symbols: List[str], timeframe: str, limit: int = 120) -> Dict[str, Any]:
    headers = {"accept": "application/json"}
    url = bars_url(symbols, timeframe, limit)
    async with ClientSession() as s:
        return await fetch_json(s, url, headers)

# ========= Ranking (RSI + EMA trend) =========
def rank_score(c15: List[float], c30: List[float], c60: List[float],
               rsi_buy: float, ema_fast_p: int, ema_slow_p: int) -> float:
    # RSI 
    r1 = rsi(c15, 14)
    r2 = rsi(c30, 14)
    r3 = rsi(c60, 14)

    # EMA trend (старший)
    e_fast = ema(c60, ema_fast_p)
    e_slow = ema(c60, ema_slow_p)
    trend = 0.0
    if e_fast and e_slow:
        trend = (e_fast[-1] - e_slow[-1]) / max(1e-9, abs(e_slow[-1]))

    # скільки ТФ підтверджують LONG
    bias_long = (1 if r1 >= rsi_buy else 0) + (1 if r2 >= rsi_buy else 0) + (1 if r3 >= rsi_buy else 0)
    # базовий скор
    return bias_long*100 + trend*50 - abs(50.0 - r1)

async def scan_rank_crypto(st: Dict[str, Any]) -> Tuple[str, List[Tuple[float, str, List[Dict[str, Any]]]]]:
    conf = mode_conf(st)
    tf15, tf30, tf60 = conf["bars"]
    pairs = CRYPTO_USD_PAIRS[:]  # whitelist

    # берём бари для трьох ТФ
    b15 = await get_bars_crypto(pairs, tf15, limit=120)
    b30 = await get_bars_crypto(pairs, tf30, limit=120)
    b60 = await get_bars_crypto(pairs, tf60, limit=120)

    ranked: List[Tuple[float, str, List[Dict[str, Any]]]] = []
    for sym in pairs:
        raw15 = (b15.get("bars") or {}).get(sym, [])
        raw30 = (b30.get("bars") or {}).get(sym, [])
        raw60 = (b60.get("bars") or {}).get(sym, [])
        if not raw15 or not raw30 or not raw60:
            continue

        c15 = [float(x["c"]) for x in raw15]
        c30 = [float(x["c"]) for x in raw30]
        c60 = [float(x["c"]) for x in raw60]

        score = rank_score(
            c15, c30, c60,
            conf["rsi_buy"], conf["ema_fast"], conf["ema_slow"]
        )
        ranked.append((score, sym, raw15))

    ranked.sort(reverse=True)
    rep = (
        "🛰️ Сканер (крипта):\n"
        f"• Активних USD-пар: {len(pairs)}\n"
        f"• Використаємо для торгівлі (лімітом): {min(conf['top_n'], len(ranked))}\n"
        f"• Перші 25: " + ", ".join([s for _, s, _ in ranked[:25]]) if ranked else "Немає сигналів"
    )
    return rep, ranked

# ========= Orders =========
def order_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "accept": "application/json",
        "content-type": "application/json",
    }

async def place_bracket_notional_order(symbol: str, side: str, notional: float,
                                       tp_price: float, sl_price: float) -> Any:
    """
    Для кріпти на Alpaca (spot) робимо ринковий ордер + TP/SL (bracket).
    side: buy (short не підтримується у spot).
    """
    if side != "buy":
        raise RuntimeError("Short для крипти (spot) не підтримується Alpaca.")

    payload = {
        "symbol": symbol,            # формат "AVAX/USD"
        "side": side,                # buy
        "type": "market",
        "time_in_force": "gtc",
        "notional": round(float(notional), 2),
        "take_profit": {"limit_price": round(float(tp_price), 6)},
        "stop_loss":  {"stop_price":  round(float(sl_price), 6)},
    }
    url = f"{ALPACA_BASE_URL}/v2/orders"
    async with ClientSession() as s:
        return await post_json(s, url, order_headers(), payload)

def calc_sl_tp(side: str, px: float, h: List[float], l: List[float], c: List[float],
               tp_pct: float, sl_pct: float) -> Tuple[float, float]:
    """
    Простий варіант SL/TP: від ціни входу відсотком.
    Для кріпти (LONG) — TP вище ціни, SL нижче.
    """
    if side == "buy":
        tp = px * (1.0 + tp_pct)
        sl = px * (1.0 - sl_pct)
    else:
        # на всяк випадок, але ми не викликаємо sell для кріпти
        tp = px * (1.0 - tp_pct)
        sl = px * (1.0 + sl_pct)
    return tp, sl

# ========= Telegram UI =========
def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            ["/aggressive", "/scalp", "/default"],
            ["/swing", "/safe", "/help"],
            ["/signals_crypto", "/trade_crypto"],
            ["/long_mode", "/short_mode", "/both_mode"],
            ["/alp_on", "/alp_status", "/alp_off"],
        ],
        resize_keyboard=True
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(update.effective_chat.id)
    await update.message.reply_text(
        "Привіт! Бот готовий. Вибирай режим і команди.\n"
        "Short для крипти вимкнено (Alpaca spot не підтримує).\n",
        reply_markup=main_keyboard()
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Команди:\n"
        "/signals_crypto — сканер + (якщо ввімкнено) автотрейд топ-N\n"
        "/trade_crypto — миттєвий трейд топ-N\n"
        "/alp_on /alp_off /alp_status — автотрейд\n"
        "/long_mode /short_mode /both_mode — напрям (short буде проігноровано)\n"
        "/aggressive /scalp /default /swing /safe — профілі ризику\n"
    )

async def set_mode(update: Update, context: ContextTypes.DEFAULT_TYPE, name: str) -> None:
    st = stedef(update.effective_chat.id)
    st["mode"] = name
    await update.message.reply_text(f"Режим: {name.upper()}")

async def aggressive(update: Update, context: ContextTypes.DEFAULT_TYPE): await set_mode(update, context, "aggressive")
async def scalp(update: Update, context: ContextTypes.DEFAULT_TYPE):      await set_mode(update, context, "scalp")
async def default(update: Update, context: ContextTypes.DEFAULT_TYPE):    await set_mode(update, context, "default")
async def swing(update: Update, context: ContextTypes.DEFAULT_TYPE):      await set_mode(update, context, "swing")
async def safe(update: Update, context: ContextTypes.DEFAULT_TYPE):       await set_mode(update, context, "safe")

async def long_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    st = stedef(update.effective_chat.id)
    st["side_mode"] = "long"
    await update.message.reply_text("Режим входів: LONG")

async def short_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    st = stedef(update.effective_chat.id)
    st["side_mode"] = "short"
    await update.message.reply_text("Режим входів: SHORT (⚠️ для крипти буде проігноровано)")

async def both_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    st = stedef(update.effective_chat.id)
    st["side_mode"] = "both"
    await update.message.reply_text("Режим входів: BOTH (⚠️ sell для крипти буде проігноровано)")

async def alp_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    st = stedef(update.effective_chat.id)
    st["autotrade"] = True
    await update.message.reply_text("✅ Alpaca AUTOTRADE: ON")

async def alp_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    st = stedef(update.effective_chat.id)
    st["autotrade"] = False
    await update.message.reply_text("⛔ Alpaca AUTOTRADE: OFF")

async def alp_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    st = stedef(update.effective_chat.id)
    mode = st.get("mode")
    side = st.get("side_mode")
    await update.message.reply_text(
        f"Alpaca:\n• Mode={mode} · Autotrade={'ON' if st.get('autotrade') else 'OFF'} · Side={side}",
        reply_markup=main_keyboard()
    )

# ========= Signals / Trade =========
async def signals_crypto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        st = stedef(update.effective_chat.id)
        report, ranked = await scan_rank_crypto(st)
        await update.message.reply_text(report)

        if not ranked:
            return

        if st.get("autotrade"):
            conf = mode_conf(st)
            picks = ranked[:min(conf["top_n"], len(ranked))]
            for _, sym, arr in picks:
                if not is_crypto_pair(sym):
                    continue
                # only LONG for crypto
                side = "buy"
                # дубль-фільтр
                if is_recently_traded(sym, side):
                    continue

                h = [float(x["h"]) for x in arr]
                l = [float(x["l"]) for x in arr]
                c = [float(x["c"]) for x in arr]
                px = c[-1]
                tp, sl = calc_sl_tp(side, px, h, l, c, conf["tp_pct"], conf["sl_pct"])
                try:
                    r = await place_bracket_notional_order(sym, side, ALPACA_NOTIONAL, tp, sl)
                    mark_traded(sym, side)
                    await update.message.reply_text(
                        f"🟢 ORDER OK: {sym} BUY ${ALPACA_NOTIONAL:.2f}\n"
                        f"TP:{tp:.6f} · SL:{sl:.6f}"
                    )
                except Exception as e:
                    await update.message.reply_text(f"🔴 ORDER FAIL {sym}: {e}")
    except Exception as e:
        await update.message.reply_text(f"🔴 signals_crypto error: {e}")

async def trade_crypto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Миттєвий трейд топ-N без окремого звіту (але з показом пари)."""
    try:
        st = stedef(update.effective_chat.id)
        _, ranked = await scan_rank_crypto(st)
        if not ranked:
            await update.message.reply_text("⚠️ Сигналів недостатньо")
            return

        conf = mode_conf(st)
        picks = ranked[:min(conf["top_n"], len(ranked))]
        for _, sym, arr in picks:
            if not is_crypto_pair(sym):
                continue
            side = "buy"
            if is_recently_traded(sym, side):
                continue

            h = [float(x["h"]) for x in arr]
            l = [float(x["l"]) for x in arr]
            c = [float(x["c"]) for x in arr]
            px = c[-1]
            tp, sl = calc_sl_tp(side, px, h, l, c, conf["tp_pct"], conf["sl_pct"])
            try:
                r = await place_bracket_notional_order(sym, side, ALPACA_NOTIONAL, tp, sl)
                mark_traded(sym, side)
                await update.message.reply_text(
                    f"🟢 ORDER OK: {sym} BUY ${ALPACA_NOTIONAL:.2f}\n"
                    f"TP:{tp:.6f} · SL:{sl:.6f}"
                )
            except Exception as e:
                await update.message.reply_text(f"🔴 ORDER FAIL {sym}: {e}")
    except Exception as e:
        await update.message.reply_text(f"🔴 trade_crypto error: {e}")

# ========= Periodic job (autotrade) =========
async def periodic_scan_job(ctx: CallbackContext) -> None:
    for chat_id, st in list(STATE.items()):
        try:
            if not st.get("autotrade"):
                continue
            # той же пайплайн що й у /signals_crypto (без зайвого спаму)
            _, ranked = await scan_rank_crypto(st)
            if not ranked:
                continue
            conf = mode_conf(st)
            picks = ranked[:min(conf["top_n"], len(ranked))]
            for _, sym, arr in picks:
                if not is_crypto_pair(sym):
                    continue
                side = "buy"
                if is_recently_traded(sym, side):
                    continue
                h = [float(x["h"]) for x in arr]
                l = [float(x["l"]) for x in arr]
                c = [float(x["c"]) for x in arr]
                px = c[-1]
                tp, sl = calc_sl_tp(side, px, h, l, c, conf["tp_pct"], conf["sl_pct"])
                try:
                    await place_bracket_notional_order(sym, side, ALPACA_NOTIONAL, tp, sl)
                    mark_traded(sym, side)
                    await ctx.bot.send_message(chat_id, f"🟢 ORDER OK: {sym} BUY ${ALPACA_NOTIONAL:.2f} (auto)")
                except Exception as e:
                    await ctx.bot.send_message(chat_id, f"🔴 ORDER FAIL {sym} (auto): {e}")
        except Exception as e:
            await ctx.bot.send_message(chat_id, f"🔴 periodic_scan error: {e}")

# ========= App =========
def build_app() -> Application:
    app = Application.builder().token(TG_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))

    app.add_handler(CommandHandler("aggressive", aggressive))
    app.add_handler(CommandHandler("scalp", scalp))
    app.add_handler(CommandHandler("default", default))
    app.add_handler(CommandHandler("swing", swing))
    app.add_handler(CommandHandler("safe", safe))

    app.add_handler(CommandHandler("long_mode", long_mode))
    app.add_handler(CommandHandler("short_mode", short_mode))
    app.add_handler(CommandHandler("both_mode", both_mode))

    app.add_handler(CommandHandler("alp_on", alp_on))
    app.add_handler(CommandHandler("alp_off", alp_off))
    app.add_handler(CommandHandler("alp_status", alp_status))

    app.add_handler(CommandHandler("signals_crypto", signals_crypto))
    app.add_handler(CommandHandler("trade_crypto", trade_crypto))

    app.job_queue.run_repeating(periodic_scan_job, interval=SCAN_INTERVAL_SEC, first=10)
    return app

if __name__ == "__main__":
    if not TG_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is missing")
    if not (ALPACA_API_KEY and ALPACA_API_SECRET):
        raise SystemExit("ALPACA_API_KEY/ALPACA_API_SECRET are missing")
    app = build_app()
    app.run_polling(close_loop=False)
