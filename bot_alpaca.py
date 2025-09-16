# bot_alpaca.py
# -*- coding: utf-8 -*-

import os
import json
import math
import asyncio
import time
from typing import Dict, Any, Tuple, List

from aiohttp import ClientSession, ClientTimeout

from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# ========= ENV =========
TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_TOKEN") or "").strip()

ALPACA_API_KEY = (os.getenv("ALPACA_API_KEY") or "").strip()
ALPACA_API_SECRET = (os.getenv("ALPACA_API_SECRET") or "").strip()

ALPACA_BASE_URL = (os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_URL = (os.getenv("ALPACA_DATA_URL") or "https://data.alpaca.markets").rstrip("/")

ALPACA_NOTIONAL = float(os.getenv("ALPACA_NOTIONAL") or 25)
ALPACA_TOP_N = int(os.getenv("ALPACA_TOP_N") or 3)
ALPACA_MAX_CRYPTO = int(os.getenv("ALPACA_MAX_CRYPTO") or 25)

# ====== GLOBAL STATE (per chat) ======
STATE: Dict[int, Dict[str, Any]] = {}

# ====== MODE PROFILES (таймфрейми, фільтри, ризики) ======
MODE_PARAMS = {
    "aggressive": {
        "bars": ("15Min", "30Min", "1Hour"),
        "rsi_buy": 55.0,   # long: понад
        "rsi_sell": 45.0,  # short: нижче (не використовується для крипти)
        "ema_fast": 15, "ema_slow": 30,
        "top_n": ALPACA_TOP_N,
        "tp_pct": 0.015,   # 1.5%
        "sl_pct": 0.008,   # 0.8%
    },
    "scalp": {
        "bars": ("5Min", "15Min", "1Hour"),
        "rsi_buy": 58.0, "rsi_sell": 42.0,
        "ema_fast": 9, "ema_slow": 21,
        "top_n": ALPACA_TOP_N,
        "tp_pct": 0.010,
        "sl_pct": 0.006,
    },
    "default": {
        "bars": ("15Min", "30Min", "1Hour"),
        "rsi_buy": 56.0, "rsi_sell": 44.0,
        "ema_fast": 12, "ema_slow": 26,
        "top_n": ALPACA_TOP_N,
        "tp_pct": 0.012,
        "sl_pct": 0.008,
    },
    "swing": {
        "bars": ("30Min", "1Hour", "1Day"),
        "rsi_buy": 55.0, "rsi_sell": 45.0,
        "ema_fast": 20, "ema_slow": 40,
        "top_n": ALPACA_TOP_N,
        "tp_pct": 0.020,
        "sl_pct": 0.010,
    },
    "safe": {
        "bars": ("15Min", "30Min", "1Hour"),
        "rsi_buy": 60.0, "rsi_sell": 40.0,
        "ema_fast": 15, "ema_slow": 35,
        "top_n": max(1, ALPACA_TOP_N - 1),
        "tp_pct": 0.009,
        "sl_pct": 0.006,
    },
}

# ====== CRYPTO WHITELIST (USD пари) ======
CRYPTO_USD_PAIRS = [
    "BTC/USD","ETH/USD","SOL/USD","LTC/USD","DOGE/USD","AVAX/USD","AAVE/USD","MKR/USD","DOT/USD",
    "LINK/USD","UNI/USD","PEPE/USD","XRP/USD","TRUMP/USD","CRV/USD","BCH/USD","BAT/USD","GRT/USD",
    "XTZ/USD","USDC/USD","USDT/USD","USDG/USD","YFI/USD","LDO/USD"
][:ALPACA_MAX_CRYPTO]

# ============ HELPERS (timeframe, symbols, dedup, http) ============
def map_tf(tf: str) -> str:
    """Alpaca data API не приймає 60Min — треба 1Hour."""
    t = (tf or "").strip()
    return "1Hour" if t.lower() in ("60min", "60", "1h", "60мин", "60мін") else t

def to_order_sym(sym: str) -> str:
    return sym.replace("/", "").upper()

def to_data_sym(sym: str) -> str:
    s = sym.replace(" ", "").upper()
    if "/" in s:
        return s
    if s.endswith("USD"):
        return s[:-3] + "/USD"
    return s

def now_s() -> float:
    return time.time()

DEDUP_COOLDOWN_MIN = 60
RECENT_TRADES: Dict[str, float] = {}  # "AAVEUSD|buy" -> timestamp
def skip_as_duplicate(sym: str, side: str) -> bool:
    key = f"{to_order_sym(sym)}|{side.lower()}"
    last = RECENT_TRADES.get(key, 0)
    if now_s() - last < DEDUP_COOLDOWN_MIN * 60:
        return True
    RECENT_TRADES[key] = now_s()
    return False

def _mode_conf(st: Dict[str, Any]) -> Dict[str, Any]:
    mode = st.get("mode") or "default"
    return MODE_PARAMS.get(mode, MODE_PARAMS["default"])

def stdef(chat_id: int) -> Dict[str, Any]:
    st = STATE.setdefault(chat_id, {})
    st.setdefault("mode", "aggressive")
    st.setdefault("autotrade", False)
    st.setdefault("side_mode", "long")
    return st

def kb() -> ReplyKeyboardMarkup:
    rows = [
        ["/aggressive", "/scalp", "/default"],
        ["/swing", "/safe", "/help"],
        ["/signals_crypto", "/trade_crypto"],
        ["/long_mode", "/short_mode", "/both_mode"],
        ["/alp_on", "/alp_status", "/alp_off"],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# -------- HTTP ----------
def _alp_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }

async def alp_get_json(path: str, params: Dict[str, Any] | None = None) -> Any:
    url = f"{ALPACA_BASE_URL}{path}" if path.startswith("/v") else f"{ALPACA_DATA_URL}{path}"
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        async with s.get(url, headers=_alp_headers(), params=params) as r:
            r.raise_for_status()
            return await r.json()

async def alp_post_json(path: str, payload: Dict[str, Any]) -> Any:
    url = f"{ALPACA_BASE_URL}{path}"
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        async with s.post(url, headers=_alp_headers(), data=json.dumps(payload)) as r:
            txt = await r.text()
            if r.status >= 400:
                raise RuntimeError(f"POST {path} {r.status}: {txt}")
            return json.loads(txt) if txt else {}

# -------- DATA: crypto bars --------
async def get_bars_crypto(pairs: List[str], timeframe: str, limit: int = 120) -> Dict[str, Any]:
    tf = map_tf(timeframe)
    syms = ",".join([to_data_sym(p) for p in pairs])
    path = f"/v1beta3/crypto/us/bars"
    params = {"symbols": syms, "timeframe": tf, "limit": str(limit), "sort": "asc"}
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        url = f"{ALPACA_DATA_URL}{path}"
        async with s.get(url, headers=_alp_headers(), params=params) as r:
            txt = await r.text()
            if r.status >= 400:
                raise RuntimeError(f"GET {url} {r.status}: {txt}")
            return json.loads(txt) if txt else {}

# -------- INDICATORS --------
def ema(values: List[float], period: int) -> List[float]:
    if not values or period <= 0:
        return []
    k = 2.0 / (period + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out

def rsi(values: List[float], period: int) -> float:
    if len(values) < period + 1:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(-period, 0):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 70.0
    rs = gains / losses
    return 100.0 - (100.0 / (1 + rs))

def rank_score(c15: List[float], c30: List[float], c60: List[float],
               rsi_buy: float, rsi_sell: float,
               ema_fast_p: int, ema_slow_p: int) -> float:
    r1 = rsi(c15, 14)
    r2 = rsi(c30, 14)
    r3 = rsi(c60, 14)
    e_fast = ema(c60, ema_fast_p)
    e_slow = ema(c60, ema_slow_p)
    trend = 0.0
    if e_fast and e_slow:
        trend = (e_fast[-1] - e_slow[-1]) / max(1e-9, abs(e_slow[-1]))
    bias_long = (1 if r1 >= rsi_buy else 0) + (1 if r2 >= rsi_buy else 0) + (1 if r3 >= rsi_buy else 0)
    bias_short = (1 if r1 <= rsi_sell else 0) + (1 if r2 <= rsi_sell else 0) + (1 if r3 <= rsi_sell else 0)
    bias = max(bias_long, bias_short)
    return bias*100 + trend*50 - abs(50.0 - r1)

# -------- SCAN (CRYPTO) --------
async def scan_rank_crypto(st: Dict[str, Any]) -> Tuple[str, List[Tuple[float, str, List[Dict[str, Any]]]]]:
    conf = _mode_conf(st)
    tf15, tf30, tf60 = conf["bars"]
    tf15, tf30, tf60 = map_tf(tf15), map_tf(tf30), map_tf(tf60)

    pairs = CRYPTO_USD_PAIRS[:]
    data_pairs = [to_data_sym(p) for p in pairs]

    bars15 = await get_bars_crypto(data_pairs, tf15, limit=120)
    bars30 = await get_bars_crypto(data_pairs, tf30, limit=120)
    bars60 = await get_bars_crypto(data_pairs, tf60, limit=120)

    ranked: List[Tuple[float, str, List[Dict[str, Any]]]] = []
    for sym in data_pairs:
        raw15 = (bars15.get("bars") or {}).get(sym, [])
        raw30 = (bars30.get("bars") or {}).get(sym, [])
        raw60 = (bars60.get("bars") or {}).get(sym, [])
        if not raw15 or not raw30 or not raw60:
            continue
        c15 = [float(x["c"]) for x in raw15]
        c30 = [float(x["c"]) for x in raw30]
        c60 = [float(x["c"]) for x in raw60]
        score = rank_score(c15, c30, c60, conf["rsi_buy"], conf["rsi_sell"], conf["ema_fast"], conf["ema_slow"])
        ranked.append((score, sym, raw15))

    ranked.sort(reverse=True)
    rep = (
        "🛰️ Сканер (крипта):\n"
        f"• Активних USD-пар: {len(data_pairs)}\n"
        f"• Використаємо для торгівлі (лімітом): {min(conf['top_n'], len(ranked))}\n"
        + (f"• Перші 25: " + ", ".join([s for _, s, _ in ranked[:25]])
           if ranked else "• Немає сигналів")
    )
    return rep, ranked

# -------- ORDERS --------
async def place_bracket_notional_order(sym: str, side: str, notional: float, tp: float | None, sl: float | None) -> Any:
    payload = {
        "symbol": to_order_sym(sym),
        "side": side,
        "type": "market",
        "time_in_force": "gtc",
        "notional": str(notional),
    }
    if tp:
        payload["take_profit"] = {"limit_price": f"{tp:.6f}"}
    if sl:
        payload["stop_loss"] = {"stop_price": f"{sl:.6f}"}
    return await alp_post_json("/v2/orders", payload)

def calc_sl_tp(side: str, price: float, conf: Dict[str, Any]) -> Tuple[float | None, float | None]:
    tp_pct = float(conf.get("tp_pct", 0.012))
    sl_pct = float(conf.get("sl_pct", 0.008))
    if side == "buy":
        tp = price * (1 + tp_pct)
        sl = price * (1 - sl_pct)
    else:
        # short для крипти не використовуємо
        tp = sl = None
    return sl, tp

# -------- COMMANDS --------
async def start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    stdef(u.effective_chat.id)
    await u.message.reply_text("Привіт! Це бот Alpaca. Команди на клавіатурі нижче.", reply_markup=kb())

async def help_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "/signals_crypto — показати топ-N і (якщо Autotrade=ON) поставити ордери\n"
        "/trade_crypto — миттєво торгувати топ-N без додаткового звіту\n"
        "/alp_on /alp_off /alp_status — автотрейд\n"
        "/long_mode /short_mode /both_mode — напрям (short ігнорується для крипти)\n"
        "/aggressive /scalp /default /swing /safe — профілі ризику",
        reply_markup=kb()
    )

async def set_mode(u: Update, c: ContextTypes.DEFAULT_TYPE, mode: str):
    st = stdef(u.effective_chat.id)
    st["mode"] = mode
    await u.message.reply_text(f"Режим встановлено: {mode.upper()}")

async def long_mode(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    st["side_mode"] = "long"
    await u.message.reply_text("Режим входів: LONG")

async def short_mode(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    st["side_mode"] = "short"
    await u.message.reply_text("Режим входів: SHORT (для крипти буде проігноровано)")

async def both_mode(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    st["side_mode"] = "both"
    await u.message.reply_text("Режим входів: BOTH (для крипти застосуємо лише LONG)")

async def alp_on(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    st["autotrade"] = True
    await u.message.reply_text("✅ Alpaca AUTOTRADE: ON")

async def alp_off(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    st["autotrade"] = False
    await u.message.reply_text("⛔ Alpaca AUTOTRADE: OFF")

async def alp_status(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        acc = await alp_get_json("/v2/account")
        st = stdef(u.effective_chat.id)
        txt = (
            "📦 Alpaca:\n"
            f"• status={acc.get('status','UNKNOWN')}\n"
            f"• cash=${float(acc.get('cash',0)):.2f}\n"
            f"• buying_power=${float(acc.get('buying_power',0)):.2f}\n"
            f"• equity=${float(acc.get('equity',0)):.2f}\n"
            f"Mode={st.get('mode','default')} · Autotrade={'ON' if st.get('autotrade') else 'OFF'} · "
            f"Side={st.get('side_mode','long')}"
        )
        await u.message.reply_text(txt)
    except Exception as e:
        await u.message.reply_text(f"🔴 alp_status error: {e}")

async def signals_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        report, ranked = await scan_rank_crypto(st)
        await u.message.reply_text(report)

        if not st.get("autotrade") or not ranked:
            return

        picks = ranked[: _mode_conf(st)["top_n"]]
        for _, sym, arr in picks:
            side = "buy"  # short ігноруємо
            px = float(arr[-1]["c"])
            conf = _mode_conf(st)
            sl, tp = calc_sl_tp(side, px, conf)

            # антидубль
            if skip_as_duplicate(sym, side):
                await u.message.reply_text(f"⚪ SKIP (дубль): {sym} {side.upper()}")
                continue

            try:
                await place_bracket_notional_order(sym, side, ALPACA_NOTIONAL, tp, sl)
                await u.message.reply_text(
                    f"🟢 ORDER OK: {sym} BUY ${ALPACA_NOTIONAL:.2f}\n"
                    f"(auto) TP:{tp:.6f} SL:{sl:.6f}"
                )
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")

    except Exception as e:
        await u.message.reply_text(f"🔴 signals_crypto error: {e}")

async def trade_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE):
    """Миттєвий трейд без довгого репорту — топ-N."""
    st = stdef(u.effective_chat.id)
    try:
        _, ranked = await scan_rank_crypto(st)
        if not ranked:
            await u.message.reply_text("⚠️ Немає сигналів")
            return
        picks = ranked[: _mode_conf(st)["top_n"]]
        for _, sym, arr in picks:
            side = "buy"
            px = float(arr[-1]["c"])
            conf = _mode_conf(st)
            sl, tp = calc_sl_tp(side, px, conf)

            if skip_as_duplicate(sym, side):
                await u.message.reply_text(f"⚪ SKIP (дубль): {sym} {side.upper()}")
                continue

            try:
                await place_bracket_notional_order(sym, side, ALPACA_NOTIONAL, tp, sl)
                await u.message.reply_text(
                    f"🟢 ORDER OK: {sym} BUY ${ALPACA_NOTIONAL:.2f}\n"
                    f"(auto) TP:{tp:.6f} SL:{sl:.6f}"
                )
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")
    except Exception as e:
        await u.message.reply_text(f"🔴 trade_crypto error: {e}")

# ======= MODE SHORTCUTS =======
async def aggressive(u, c): await set_mode(u, c, "aggressive")
async def scalp(u, c): await set_mode(u, c, "scalp")
async def default(u, c): await set_mode(u, c, "default")
async def swing(u, c): await set_mode(u, c, "swing")
async def safe(u, c): await set_mode(u, c, "safe")

# ========= MAIN =========
def main() -> None:
    if not TG_TOKEN:
        raise SystemExit("No TELEGRAM_BOT_TOKEN provided")

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

    print("Bot started.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
