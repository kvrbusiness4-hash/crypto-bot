# bot_alpaca.py
# -*- coding: utf-8 -*-

import os
import json
import aiohttp
from typing import Dict, Any, Tuple, List

from aiohttp import ClientSession, ClientTimeout

from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes
)
# ---- MODE PROFILES (таймфрейми, фільтри, ризик) ----
MODE_PARAMS = {
    "aggressive": {   # багато сигналів, більше ризику
        "bars": ("5Min", "15Min", "1Hour"),
        "rsi_buy": 55.0,      # long: понад
        "rsi_sell": 45.0,     # short: нижче
        "ema_fast": 15, "ema_slow": 30,
        "top_n": 10,          # скільки інструментів взяти
        "tp_pct": 0.015,      # 1.5%
        "sl_pct": 0.008,      # 0.8%
    },
    "scalp": {        # короткі рухи, вузькі SL/TP
        "bars": ("5Min", "15Min", "1Hour"),
        "rsi_buy": 58.0,
        "rsi_sell": 42.0,
        "ema_fast": 15, "ema_slow": 30,
        "top_n": 6,
        "tp_pct": 0.012,
        "sl_pct": 0.007,
    },
    "default": {      # баланс
        "bars": ("15Min", "30Min", "1Hour"),
        "rsi_buy": 60.0,
        "rsi_sell": 40.0,
        "ema_fast": 30, "ema_slow": 60,
        "top_n": 5,
        "tp_pct": 0.02,
        "sl_pct": 0.01,
    },
    "swing": {        # менше угод, довші рухи
        "bars": ("30Min", "1Hour", "1Day"),
        "rsi_buy": 62.0,
        "rsi_sell": 38.0,
        "ema_fast": 30, "ema_slow": 60,
        "top_n": 3,
        "tp_pct": 0.035,
        "sl_pct": 0.015,
    },
    "safe": {         # лише найсильніші
        "bars": ("15Min", "1Hour", "1Day"),
        "rsi_buy": 65.0,
        "rsi_sell": 35.0,
        "ema_fast": 30, "ema_slow": 60,
        "top_n": 3,
        "tp_pct": 0.03,
        "sl_pct": 0.012,
    },
}
DEFAULT_MODE = "default"

def _mode_conf(st: dict) -> dict:
    return MODE_PARAMS.get(st.get("mode", DEFAULT_MODE), MODE_PARAMS[DEFAULT_MODE])
# =========================
# ENV
# =========================
TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
            or os.getenv("TELEGRAM_TOKEN", "").strip())

ALPACA_API_KEY   = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_API_SECRET= os.getenv("ALPACA_API_SECRET", "").strip()
ALPACA_BASE_URL  = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_URL  = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")

ALPACA_NOTIONAL  = float(os.getenv("ALPACA_NOTIONAL", "25") or 25)
ALPACA_TOP_N     = int(os.getenv("ALPACA_TOP_N", "5") or 5)

# SL/TP і режим входів
ALP_SL_K         = float(os.getenv("ALP_SL_K", "1.3") or 1.3)   # множник ATR для SL
ALP_RR_K         = float(os.getenv("ALP_RR_K", "2.2") or 2.2)   # співвідношення TP/ризик
DEFAULT_SIDE_MODE= os.getenv("ALP_SIDE_MODE", "both").lower()   # long|short|both

# =========================
# СТАН
# =========================
def default_state() -> Dict[str, Any]:
    return {
        "autotrade": False,
        "mode": "default",
        "last_scan_txt": "",
        "side_mode": DEFAULT_SIDE_MODE,
    }

STATE: Dict[int, Dict[str, Any]] = {}
def stedef(chat_id: int) -> Dict[str, Any]:
    return STATE.setdefault(chat_id, default_state())

# =========================
# КЛАВІАТУРА
# =========================
def main_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        ["/aggressive", "/scalp", "/default"],
        ["/swing", "/safe", "/help"],
        ["/signals_crypto", "/trade_crypto"],
        ["/long_mode", "/short_mode", "/both_mode"],
        ["/alp_on", "/alp_status", "/alp_off"],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# =========================
# HTTP (Alpaca)
# =========================
def alp_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }

def t_url(p: str) -> str:
    return f"{ALPACA_BASE_URL}/v2/{p.lstrip('/')}"

async def alp_get(path: str) -> Any:
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        async with s.get(t_url(path), headers=alp_headers()) as r:
            if r.status >= 400:
                raise RuntimeError(f"GET {r.url} {r.status}: {await r.text()}")
            return await r.json()

async def alp_post(path: str, payload: Dict[str, Any]) -> Any:
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        async with s.post(t_url(path), headers=alp_headers(), data=json.dumps(payload)) as r:
            if r.status >= 400:
                raise RuntimeError(f"POST {r.url} {r.status}: {await r.text()}")
            return await r.json()

async def alp_account() -> Dict[str, Any]:
    return await alp_get("account")

# ----- Market Data (crypto bars 15/30/60) -----
async def md_get(url: str, params: Dict[str, str]) -> Any:
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        async with s.get(url, headers=alp_headers(), params=params) as r:
            if r.status >= 400:
                raise RuntimeError(f"GET {r.url} {r.status}: {await r.text()}")
            return await r.json()

async def get_bars_crypto(symbols: List[str], timeframe: str, limit: int = 120) -> Dict[str, Any]:
    url = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/bars"
    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "limit": str(limit),
        "sort": "asc",
    }
    return await md_get(url, params)

async def get_active_crypto_usd_pairs() -> List[str]:
    # беремо всі активні крипто-асети -> залишаємо лише /USD
    url = f"{ALPACA_BASE_URL}/v2/assets"
    params = {"asset_class": "crypto", "status": "active"}
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        async with s.get(url, headers=alp_headers(), params=params) as r:
            if r.status >= 400:
                raise RuntimeError(f"GET {r.url} {r.status}: {await r.text()}")
            items = await r.json()
            return [x["symbol"] for x in items if str(x.get("symbol","")).endswith("/USD")]

# =========================
# TA helpers
# =========================
def ema(vals: List[float], n: int):
    if not vals or len(vals) < n: return None
    k = 2/(n+1); e = vals[0]
    for v in vals[1:]: e = v*k + e*(1-k)
    return e

def rsi(vals: List[float], n: int = 14):
    if len(vals) < n+1: return None
    gains, losses = [], []
    for i in range(1, len(vals)):
        d = vals[i] - vals[i-1]
        gains.append(max(d, 0.0)); losses.append(max(-d, 0.0))
    ag = sum(gains[-n:])/n; al = sum(losses[-n:])/n
    if al == 0: return 100.0
    rs = ag/al
    return 100 - (100/(1+rs))

def atr_from_ohlc(h: List[float], l: List[float], c: List[float], n: int = 14):
    if len(c) < n+1: return None
    trs = []
    for i in range(1, len(c)):
        trs.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    return sum(trs[-n:])/n

def side_by_trend(close_list: List[float]) -> str:
    if len(close_list) < 60: return "buy"
    e15, e30, e60 = ema(close_list, 15), ema(close_list, 30), ema(close_list, 60)
    px = close_list[-1]
    if px < min(e15, e30, e60): return "sell"
    if px > max(e15, e30, e60): return "buy"
    return "buy"

def calc_sl_tp(side: str, px: float, h: List[float], l: List[float], c: List[float]) -> Tuple[float,float]:
    atr = atr_from_ohlc(h, l, c, 14) or (px*0.01)
    if side == "buy":
        sl = px - ALP_SL_K*atr
        tp = px + ALP_RR_K*(px - sl)
    else:
        sl = px + ALP_SL_K*atr
        tp = px - ALP_RR_K*(sl - px)
    return sl, tp

async def place_bracket_notional_order(
    symbol: str,
    side: str,                   # "buy" або "sell"
    notional: float,
    take_profit: float,          # ціна TP
    stop_loss: float             # ціна SL (stop)
) -> dict:
    """
    Виставляє market bracket-order за сумою (notional) з TP/SL.
    Для crypto 'sell' як відкриття шорту – не підтримується Alpaca.
    """
    # Захист від шорту крипти (Alpaca spot не дозволяє short crypto)
    if "/" in symbol and side.lower() == "sell":
        raise RuntimeError("Short для крипти не підтримується Alpaca (spot).")

    order = {
        "symbol": symbol,
        "side": side.lower(),            # "buy" | "sell"
        "type": "market",
        "time_in_force": "gtc",
        "notional": str(float(notional)),
        "order_class": "bracket",
        "take_profit": {"limit_price": str(float(take_profit))},
        "stop_loss":   {"stop_price":  str(float(stop_loss))},
    }

    async with aiohttp.ClientSession(timeout=ClientTimeout(total=30)) as s:
        return await alp_post(s, "orders", order)
# =========================
# БАЗОВІ КОМАНДИ
# =========================
async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "👋 Готово. Команди:\n"
        "• /signals_crypto — звіт і (за бажанням) автотрейд\n"
        "• /trade_crypto — миттєвий трейд за топ-N\n"
        "• /alp_on /alp_off /alp_status\n"
        "• /long_mode /short_mode /both_mode — режим входів\n"
        "Крипта 24/7.",
        reply_markup=main_keyboard(),
        parse_mode=ParseMode.MARKDOWN,
    )

async def aggressive_cmd(u, c): stedef(u.effective_chat.id).update(mode="aggressive"); await u.message.reply_text("✅ Mode: AGGRESSIVE", reply_markup=main_keyboard())
async def scalp_cmd(u, c):      stedef(u.effective_chat.id).update(mode="scalp");      await u.message.reply_text("✅ Mode: SCALP", reply_markup=main_keyboard())
async def default_cmd(u, c):    stedef(u.effective_chat.id).update(mode="default");    await u.message.reply_text("✅ Mode: DEFAULT", reply_markup=main_keyboard())
async def swing_cmd(u, c):      stedef(u.effective_chat.id).update(mode="swing");      await u.message.reply_text("✅ Mode: SWING", reply_markup=main_keyboard())
async def safe_cmd(u, c):       stedef(u.effective_chat.id).update(mode="safe");       await u.message.reply_text("✅ Mode: SAFE", reply_markup=main_keyboard())

async def long_mode_cmd(u, c):  stedef(u.effective_chat.id)["side_mode"]="long";  await u.message.reply_text("📈 Режим входів: LONG",  reply_markup=main_keyboard())
async def short_mode_cmd(u, c): stedef(u.effective_chat.id)["side_mode"]="short"; await u.message.reply_text("📉 Режим входів: SHORT", reply_markup=main_keyboard())
async def both_mode_cmd(u, c):  stedef(u.effective_chat.id)["side_mode"]="both";  await u.message.reply_text("🔁 Режим входів: BOTH",  reply_markup=main_keyboard())

async def alp_on_cmd(u, c):  stedef(u.effective_chat.id)["autotrade"]=True;  await u.message.reply_text("✅ Alpaca AUTOTRADE: ON",  reply_markup=main_keyboard())
async def alp_off_cmd(u, c): stedef(u.effective_chat.id)["autotrade"]=False; await u.message.reply_text("⏹ Alpaca AUTOTRADE: OFF", reply_markup=main_keyboard())

async def alp_status_cmd(u, c):
    try:
        acc = await alp_account()
        txt = (
            "💼 Alpaca:\n"
            f"• status={acc.get('status','?')}\n"
            f"• cash=${float(acc.get('cash',0)):,.2f}\n"
            f"• buying_power=${float(acc.get('buying_power',0)):,.2f}\n"
            f"• equity=${float(acc.get('equity',0)):,.2f}\n"
            f"Mode={stedef(u.effective_chat.id).get('mode')} · "
            f"Autotrade={'ON' if stedef(u.effective_chat.id).get('autotrade') else 'OFF'} · "
            f"Side={stedef(u.effective_chat.id).get('side_mode')}"
        )
    except Exception as e:
        txt = f"❌ Alpaca error: {e}"
    await u.message.reply_text(txt, reply_markup=main_keyboard())

# =========================
# СКАН/ТРЕЙД КРИПТИ (15/30/60)
# =========================
def _rank_by_rsi_ema(
    c15: List[float], c30: List[float], c60: List[float],
    rsi_buy: float, rsi_sell: float, ema_fast: int, ema_slow: int
) -> float:
    def rsi(arr, n=14):
        import math
        if len(arr) < n+1: return 50.0
        gains = [max(0, arr[i]-arr[i-1]) for i in range(1, len(arr))]
        losses = [max(0, arr[i-1]-arr[i]) for i in range(1, len(arr))]
        avg_gain = sum(gains[-n:]) / n
        avg_loss = sum(losses[-n:]) / n
        if avg_loss == 0: return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def ema(arr, n):
        if len(arr) < n: return arr[-1]
        k = 2/(n+1)
        e = arr[0]
        for x in arr[1:]:
            e = x*k + e*(1-k)
        return e

    r = [rsi(c15,14), rsi(c30,14), rsi(c60,14)]
    e_fast = [ema(c15, ema_fast), ema(c30, ema_fast), ema(c60, ema_fast)]
    e_slow = [ema(c15, ema_slow), ema(c30, ema_slow), ema(c60, ema_slow)]
    e_spread = abs(e_fast[0]-e_slow[0]) / max(1e-9, e_slow[0])

    # “сильніше”, якщо RSI підтверджує тренд на кількох ТФ
    bias_long = sum(1 for x in r if x >= rsi_buy)
    bias_short = sum(1 for x in r if x <= rsi_sell)
    bias = max(bias_long, bias_short)

    # базовий скор
    return bias*100 + e_spread*50 - abs(50.0 - r[0])  # легкий пріоритет на 1-му ТФ

async def _scan_rank_crypto(st: dict) -> Tuple[str, List[Tuple[float, str, List[dict]]]]:
    """
    Повертає:
      report: текст короткого звіту
      ranked: список кортежів (score, symbol, bars_15m)
    """
    conf  = _mode_conf(st)  # бере параметри з MODE_PARAMS згідно режиму ризику
    pairs = await get_active_crypto_usd_pairs()
    if not pairs:
        return "Немає активних USD-пар", []

    tf15, tf30, tf60 = conf["bars"]          # напр., ("5Min","15Min","1Hour")
    bars15 = await get_bars_crypto(pairs, tf15, limit=120)
    bars30 = await get_bars_crypto(pairs, tf30, limit=120)
    bars60 = await get_bars_crypto(pairs, tf60, limit=120)

    ranked: List[Tuple[float, str, List[dict]]] = []

    for sym in pairs:
        raw15 = (bars15.get("bars") or {}).get(sym, [])
        raw30 = (bars30.get("bars") or {}).get(sym, [])
        raw60 = (bars60.get("bars") or {}).get(sym, [])
        if not raw15 or not raw30 or not raw60:
            continue

        c15 = [float(x["c"]) for x in raw15]
        c30 = [float(x["c"]) for x in raw30]
        c60 = [float(x["c"]) for x in raw60]

        score = _rank_by_rsi_ema(
            c15, c30, c60,
            rsi_buy = conf["rsi_buy"],
            rsi_sell= conf["rsi_sell"],
            ema_fast= conf["ema_fast"],
            ema_slow= conf["ema_slow"],
        )
        ranked.append((score, sym, raw15))

    ranked.sort(reverse=True)

    report = (
        f"🛰 Сканер (крипта):\n"
        f"• Активних USD-пар: {len(pairs)}\n"
        f"• Використаємо для торгівлі (лімітом): {min(conf['top_n'], len(ranked))}\n"
        f"• Перші 25: " + ", ".join([s for _, s, _ in ranked[:25]]) if ranked else "Немає сигналів"
    )
    return report, ranked

# --- /signals_crypto ---
async def signals_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Сканер крипти + (за потреби) автотрейд ТОП-N з TP/SL.
    Працює з режимами входів: long / short / both.
    """
    st = stedef(u.effective_chat.id)
    try:
        report, ranked = await _scan_rank_crypto(st)
        await u.message.reply_text(report)
    except Exception as e:
        await u.message.reply_text(f"🔴 crypto scan error: {e}")
        return

    if not st.get("autotrade") or not ranked:
        return

    # торгуємо топ-N
    picks = ranked[:ALPACA_TOP_N]
    mode  = st.get("side_mode", DEFAULT_SIDE_MODE)              # "long" | "short" | "both"
    sides_template = ["buy"] if mode == "long" else ["sell"] if mode == "short" else ["buy", "sell"]

    for _, sym, arr in picks:
        h  = [float(x["h"]) for x in arr]
        l  = [float(x["l"]) for x in arr]
        cc = [float(x["c"]) for x in arr]
        px = cc[-1]

        for side in sides_template:
            # Short у спот-крипті Alpaca не підтримується — пропускаємо
            if is_crypto_pair(sym) and side == "sell":
                await u.message.reply_text(f"🔴 ORDER SKIP {sym} SELL: short для крипти (spot) недоступний в Alpaca.")
                continue

            sl, tp = calc_sl_tp(side, px, h, l, cc)
            try:
                await place_bracket_notional_order(
                    sym, side, ALPACA_NOTIONAL,
                    take_profit=tp, stop_loss=sl
                )
                await u.message.reply_text(
                    f"🟢 ORDER OK: {sym} {'LONG' if side=='buy' else 'SHORT'} "
                    f"@~{px:.6f}\nTP:{tp:.6f} · SL:{sl:.6f} · ${ALPACA_NOTIONAL:.2f}"
                )
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} {side.upper()}: {e}")

# --- /trade_crypto (миттєва торгівля без звіту) ---
async def trade_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Миттєва торгівля ТОП-N без окремого звіту (корисно, коли вже знаємо, що є сигнали).
    Використовує ті самі правила TP/SL і режими входів (long / short / both).
    """
    st = stedef(u.effective_chat.id)
    try:
        _, ranked = await _scan_rank_crypto(st)
        if not ranked:
            await u.message.reply_text("⚠️ Сигналів недостатньо")
            return
    except Exception as e:
        await u.message.reply_text(f"🔴 trade_crypto error: {e}")
        return

    picks = ranked[:ALPACA_TOP_N]
    mode  = st.get("side_mode", DEFAULT_SIDE_MODE)
    sides_template = ["buy"] if mode == "long" else ["sell"] if mode == "short" else ["buy", "sell"]

    for _, sym, arr in picks:
        h  = [float(x["h"]) for x in arr]
        l  = [float(x["l"]) for x in arr]
        cc = [float(x["c"]) for x in arr]
        px = cc[-1]

        for side in sides_template:
            if is_crypto_pair(sym) and side == "sell":
                await u.message.reply_text(f"🔴 ORDER SKIP {sym} SELL: short для крипти (spot) недоступний в Alpaca.")
                continue

            sl, tp = calc_sl_tp(side, px, h, l, cc)
            try:
                await place_bracket_notional_order(
                    sym, side, ALPACA_NOTIONAL,
                    take_profit=tp, stop_loss=sl
                )
                await u.message.reply_text(
                    f"🟢 ORDER OK: {sym} {'LONG' if side=='buy' else 'SHORT'} "
                    f"@~{px:.6f}\nTP:{tp:.6f} · SL:{sl:.6f} · ${ALPACA_NOTIONAL:.2f}"
                )
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} {side.upper()}: {e}")

# =========================
# ФОНОВИЙ JOB (автотрейд)
# =========================
# --- фоновий джоб ---
async def periodic_scan_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    for chat_id, st in list(STATE.items()):
        try:
            report, ranked = await _scan_rank_crypto(st)  # <— ПЕРЕДАЄМО st
            await ctx.bot.send_message(chat_id, report)

            if st.get("autotrade") and ranked:
                conf = _mode_conf(st)
                picks = ranked[:conf["top_n"]]
                side_mode = st.get("side_mode", DEFAULT_SIDE_MODE)

                for _, sym, arr in picks:
                    h  = [float(x["h"]) for x in arr]
                    l  = [float(x["l"]) for x in arr]
                    cc = [float(x["c"]) for x in arr]
                    px = cc[-1]

                    sides = ["buy"] if side_mode=="long" else ["sell"] if side_mode=="short" else ["buy","sell"]
                    for side in sides:
                        sl, tp = calc_sl_tp(side, px, h, l, cc)
                        try:
                            await place_bracket_notional_order(sym, side, ALPACA_NOTIONAL, tp, sl)
                            await ctx.bot.send_message(
                                chat_id,
                                f"🟢 ORDER OK: {sym} {('LONG' if side=='buy' else 'SHORT')} "
                                f"TP:{tp:.6f} · SL:{sl:.6f} · ${ALPACA_NOTIONAL:.2f}"
                            )
                        except Exception as e:
                            await ctx.bot.send_message(chat_id, f"🔴 ORDER FAIL {sym} {side.upper()}: {e}")
        except Exception as e:
            try:
                await ctx.bot.send_message(chat_id, f"🔴 periodic_scan error: {e}")
            except Exception:
                pass

# =========================
# HELP
# =========================
async def help_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "Команди:\n"
        "• /signals_crypto — звіт (і автотрейд, якщо ввімкнено)\n"
        "• /trade_crypto — миттєвий трейд топ-N\n"
        "• /alp_on /alp_off /alp_status\n"
        "• /long_mode /short_mode /both_mode\n"
        "• /aggressive /scalp /default /swing /safe",
        reply_markup=main_keyboard()
    )

# =========================
# MAIN
# =========================
def main():
    if not TG_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задано")

    app = Application.builder().token(TG_TOKEN).build()

    # handlers
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))

    app.add_handler(CommandHandler("aggressive", aggressive_cmd))
    app.add_handler(CommandHandler("scalp", scalp_cmd))
    app.add_handler(CommandHandler("default", default_cmd))
    app.add_handler(CommandHandler("swing", swing_cmd))
    app.add_handler(CommandHandler("safe", safe_cmd))

    app.add_handler(CommandHandler("long_mode", long_mode_cmd))
    app.add_handler(CommandHandler("short_mode", short_mode_cmd))
    app.add_handler(CommandHandler("both_mode", both_mode_cmd))

    app.add_handler(CommandHandler("alp_on", alp_on_cmd))
    app.add_handler(CommandHandler("alp_off", alp_off_cmd))
    app.add_handler(CommandHandler("alp_status", alp_status_cmd))

    app.add_handler(CommandHandler("signals_crypto", signals_crypto))
    app.add_handler(CommandHandler("trade_crypto", trade_crypto))

    # фоновий сканер
    app.job_queue.run_repeating(periodic_scan_job, interval=120, first=10)

    app.run_polling()

if __name__ == "__main__":
    main()
