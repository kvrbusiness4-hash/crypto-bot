# bot_alpaca.py
# -*- coding: utf-8 -*-

import os
import json
import math
from typing import Dict, Any, Tuple, List

from aiohttp import ClientSession, ClientTimeout

from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes
# ==== GLOBAL STATE (встав під імпортами) ====
OPEN_TRADES: Dict[str, Dict[str, Any]] = {}  # активні угоди по символу


# ==== SL/TP розрахунок (встав поруч з іншими хелперами) ====
def calc_sl_tp(px: float, hi: float, lo: float, conf: Dict[str, Any]) -> Tuple[float, float]:
    """
    Розрахунок TP/SL на основі режиму:
      - tp_pct / sl_pct з MODE_PARAMS
      - додатково врахуємо локальний діапазон (hi/lo) як підстраховку
    Повертає: (tp_price, sl_price)
    """
    tp_pct = float(conf.get("tp_pct", 0.015))  # 1.5% за замовчуванням
    sl_pct = float(conf.get("sl_pct", 0.008))  # 0.8% за замовчуванням

    tp_price = round(px * (1.0 + tp_pct), 6)
    sl_price = round(px * (1.0 - sl_pct), 6)

    # підстрахуємося від «занадто близько»: SL нижче локального low щонайменше на тик
    if lo is not None:
        sl_price = min(sl_price, round(lo * 0.999, 6))
    # TP вище локального high трохи
    if hi is not None:
        tp_price = max(tp_price, round(hi * 1.001, 6))

    return tp_price, sl_price


# ==== HTTP-створення BRACKET-ордера (встав поруч з іншими Alpaca-хелперами) ====
async def place_bracket_notional_order(sym: str, notional: float, tp: float, sl: float) -> Dict[str, Any]:
    """
    LONG bracket order для crypto на Alpaca (paper):
      POST /v2/orders з order_class="bracket", notional, take_profit, stop_loss.
    """
    url = f"{ALPACA_BASE_URL}/v2/orders"
    payload = {
        "symbol": sym.replace("/", ""),  # "AAVEUSD"
        "side": "buy",
        "type": "market",
        "time_in_force": "gtc",
        "notional": f"{notional:.2f}",
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{tp:.6f}"},
        "stop_loss": {"stop_price": f"{sl:.6f}"}
    }
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        async with s.post(url, headers=headers, json=payload) as r:
            data = await r.json()
            if r.status >= 300:
                raise RuntimeError(f"POST {url} {r.status}: {data}")
            return data


# ==== СКАНЕР/РАНКЕР (залиш свій, якщо вже працює) ====


# ==== /signals_crypto — СКАН + АВТОТРЕЙД TOP-N з TP/SL (повністю заміни) ====
async def signals_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedet(u.effective_chat.id)
    try:
        # отримуємо ранжований список (твоя функція ранжування)
        report, ranked = await scan_rank_crypto(st)
        await u.message.reply_text(report)

        # якщо автотрейд увімкнено — торгуємо TOP-N (LONG only)
        if st.get("autotrade") and ranked:
            conf = mode_conf(st)
            top_n = min(int(conf.get("top_n", 3)), len(ranked))
            picks = ranked[:top_n]

            for _, sym, raw in picks:
                # антидубль: не відкриваємо ще раз ту ж саму пару
                if sym in OPEN_TRADES:
                    await u.message.reply_text(f"⏩ Пропуск {sym}: вже відкрита позиція")
                    continue

                h = [float(x["h"]) for x in raw]
                l = [float(x["l"]) for x in raw]
                ccls = [float(x["c"]) for x in raw]
                px = ccls[-1]

                tp, sl = calc_sl_tp(px, max(h), min(l), conf)

                try:
                    resp = await place_bracket_notional_order(sym, ALPACA_NOTIONAL, tp, sl)
                    OPEN_TRADES[sym] = {"entry": px, "tp": tp, "sl": sl, "order_id": resp.get("id")}
                    await u.message.reply_text(
                        f"🟢 ORDER OK: {sym} BUY ${ALPACA_NOTIONAL:.2f}\nTP: {tp:.6f}  ·  SL: {sl:.6f}"
                    )
                except Exception as e:
                    await u.message.reply_text(f"🔴 ORDER FAIL {sym}: {e}")
    except Exception as e:
        await u.message.reply_text(f"🔴 signals_crypto error: {e}")


# ==== /trade_crypto — миттєва торгівля TOP-N з TP/SL (повністю заміни) ====
async def trade_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedet(u.effective_chat.id)
    try:
        # беремо TOP-N з уже відсортованого списку (без повторного звіту)
        _, ranked = await scan_rank_crypto(st)
        if not ranked:
            await u.message.reply_text("⚠️ Сигналів недостатньо"); return

        conf = mode_conf(st)
        top_n = min(int(conf.get("top_n", 3)), len(ranked))
        picks = ranked[:top_n]

        for _, sym, raw in picks:
            if sym in OPEN_TRADES:
                await u.message.reply_text(f"⏩ Пропуск {sym}: вже відкрита позиція")
                continue

            h = [float(x["h"]) for x in raw]
            l = [float(x["l"]) for x in raw]
            ccls = [float(x["c"]) for x in raw]
            px = ccls[-1]

            tp, sl = calc_sl_tp(px, max(h), min(l), conf)

            try:
                resp = await place_bracket_notional_order(sym, ALPACA_NOTIONAL, tp, sl)
                OPEN_TRADES[sym] = {"entry": px, "tp": tp, "sl": sl, "order_id": resp.get("id")}
                await u.message.reply_text(
                    f"🟢 ORDER OK: {sym} BUY ${ALPACA_NOTIONAL:.2f}\nTP: {tp:.6f}  ·  SL: {sl:.6f}"
                )
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym}: {e}")

    except Exception as e:
        await u.message.reply_text(f"🔴 trade_crypto error: {e}")
# ========= ENV =========
TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or "").strip()

ALPACA_API_KEY = (os.getenv("ALPACA_API_KEY") or "").strip()
ALPACA_API_SECRET = (os.getenv("ALPACA_API_SECRET") or "").strip()

ALPACA_BASE_URL = (os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_URL = (os.getenv("ALPACA_DATA_URL") or "https://data.alpaca.markets").rstrip("/")

ALPACA_NOTIONAL = float(os.getenv("ALPACA_NOTIONAL") or 25)          # $ на ордер
ALPACA_TOP_N   = int(os.getenv("ALPACA_TOP_N") or 2)                  # скільки інструментів торгувати
SCAN_EVERY_SEC = int(os.getenv("SCAN_EVERY_SEC") or 120)

# ========= STATE per chat =========
def default_state() -> Dict[str, Any]:
    return {
        "mode": "aggressive",      # профіль: aggressive/scalp/default/swing/safe
        "autotrade": False,
        "side_mode": "long",       # long | short | both (для крипти short все одно блокуємо)
    }

STATE: Dict[int, Dict[str, Any]] = {}

def stedef(chat_id: int) -> Dict[str, Any]:
    return STATE.setdefault(chat_id, default_state())

# ========= UI =========
def main_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        ["/aggressive", "/scalp", "/default"],
        ["/swing", "/safe", "/help"],
        ["/signals_crypto", "/trade_crypto"],
        ["/long_mode", "/short_mode", "/both_mode"],
        ["/alp_on", "/alp_status", "/alp_off"],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# ========= Helpers: HTTP to Alpaca =========
def alp_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }

async def fetch_json_full(url: str, params: Dict[str, Any] | None = None) -> Any:
    timeout = ClientTimeout(total=30)
    async with ClientSession(timeout=timeout) as s:
        async with s.get(url, headers=alp_headers(), params=params) as r:
            if r.status >= 400:
                raise RuntimeError(f"GET {r.url} {r.status}: {await r.text()}")
            return await r.json()

async def alp_post(path: str, payload: Dict[str, Any]) -> Any:
    url = f"{ALPACA_BASE_URL}/v2/{path.lstrip('/')}"
    timeout = ClientTimeout(total=30)
    async with ClientSession(timeout=timeout) as s:
        async with s.post(url, headers=alp_headers(), data=json.dumps(payload)) as r:
            if r.status >= 400:
                raise RuntimeError(f"POST {r.url} {r.status}: {await r.text()}")
            return await r.json()

async def alp_account() -> Dict[str, Any]:
    url = f"{ALPACA_BASE_URL}/v2/account"
    return await fetch_json_full(url)

# ========= Symbols helpers =========
CRYPTO_QUOTES = {"USD", "USDT", "USDC", "USDG"}

def is_crypto_pair(sym: str) -> bool:
    parts = sym.split("/")
    return len(parts) == 2 and parts[1].upper() in CRYPTO_QUOTES

# Невеликий whitelist найліквідніших USD-пар (паперова торгівля підтримує їх)
CRYPTO_USD_PAIRS: List[str] = [
    "BTC/USD","ETH/USD","SOL/USD","LTC/USD","DOGE/USD","AVAX/USD","AAVE/USD","MKR/USD",
    "DOT/USD","LINK/USD","SHIB/USD","UNI/USD","PEPE/USD","XRP/USD","TRUMP/USD","CRV/USD",
    "BCH/USD","BAT/USD","GRT/USD","XTZ/USD","USDC/USD","USDT/USD","USDG/USD","YFI/USD",
    "LDO/USD"
]

# ========= Data API (bars v1beta3) =========
async def get_bars_crypto(pairs: List[str], timeframe: str, limit: int) -> Dict[str, Any]:
    """
    timeframe: '5Min' | '15Min' | '30Min' | '60Min' | '1Hour'
    """
    url = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/bars"
    sym_csv = ",".join(pairs)
    params = {
        "symbols": sym_csv,
        "timeframe": timeframe,
        "limit": str(int(limit)),
        "sort": "asc",
    }
    return await fetch_json_full(url, params)

# ========= Indicators =========
def ema(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2.0 / (period + 1.0)
    out: List[float] = []
    ema_prev = values[0]
    for v in values:
        ema_prev = v * k + ema_prev * (1 - k)
        out.append(ema_prev)
    return out

def rsi(values: List[float], period: int = 14) -> float:
    if len(values) < period + 1:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        diff = values[i] - values[i-1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 70.0
    rs = gains / losses
    return 100.0 - 100.0 / (1.0 + rs)

# ========= Mode profiles =========
MODE_PARAMS: Dict[str, Dict[str, Any]] = {
    # багато сигналів, більше ризику
    "aggressive": {
        "bars": ("15Min", "30Min", "60Min"),
        "rsi_buy": 55.0,     # LONG якщо RSI >=
        "rsi_sell": 45.0,    # SHORT якщо RSI <= (для крипти short блокується)
        "ema_fast": 15,
        "ema_slow": 30,
        "top_n": ALPACA_TOP_N,
    },
    # короткі рухи, вужчі SL/TP — у цій версії SL/TP лише розраховуються і показуються
    "scalp": {
        "bars": ("5Min", "15Min", "1Hour"),
        "rsi_buy": 58.0,
        "rsi_sell": 42.0,
        "ema_fast": 10,
        "ema_slow": 25,
        "top_n": max(1, ALPACA_TOP_N),
    },
    "default": {
        "bars": ("15Min", "30Min", "60Min"),
        "rsi_buy": 56.0,
        "rsi_sell": 44.0,
        "ema_fast": 15,
        "ema_slow": 30,
        "top_n": ALPACA_TOP_N,
    },
    "swing": {
        "bars": ("30Min", "60Min", "1Hour"),
        "rsi_buy": 57.0,
        "rsi_sell": 43.0,
        "ema_fast": 20,
        "ema_slow": 40,
        "top_n": max(1, ALPACA_TOP_N - 1),
    },
    "safe": {
        "bars": ("30Min", "60Min", "1Hour"),
        "rsi_buy": 60.0,
        "rsi_sell": 40.0,
        "ema_fast": 20,
        "ema_slow": 50,
        "top_n": 1,
    },
}

def mode_conf(st: Dict[str, Any]) -> Dict[str, Any]:
    return MODE_PARAMS.get(st.get("mode", "default"), MODE_PARAMS["default"])

# ========= Ranking =========
def rank_score(c15: List[float], c30: List[float], c60: List[float],
               rsi_buy: float, rsi_sell: float, ema_fast_p: int, ema_slow_p: int) -> float:
    # RSI на 3 ТФ
    r1 = rsi(c15, 14)
    r2 = rsi(c30, 14)
    r3 = rsi(c60, 14)
    # EMA тренд (на старшому)
    e_fast = ema(c60, ema_fast_p)
    e_slow = ema(c60, ema_slow_p)
    trend = 0.0
    if e_fast and e_slow:
        trend = (e_fast[-1] - e_slow[-1]) / max(1e-9, abs(e_slow[-1]))
    # скільки ТФ за LONG / SHORT
    bias_long = (1 if r1 >= rsi_buy else 0) + (1 if r2 >= rsi_buy else 0) + (1 if r3 >= rsi_buy else 0)
    bias_short = (1 if r1 <= rsi_sell else 0) + (1 if r2 <= rsi_sell else 0) + (1 if r3 <= rsi_sell else 0)
    bias = max(bias_long, bias_short)
    return bias*100 + trend*50 - abs(50.0 - r1)

async def scan_rank_crypto(st: Dict[str, Any]) -> Tuple[str, List[Tuple[float, str, List[Dict[str, Any]]]]]:
    conf = mode_conf(st)
    tf15, tf30, tf60 = conf["bars"]

    # замість 60Min → 1Hour
    bars15 = await get_bars_crypto(CRYPTO_USD_PAIRS, "15Min", limit=120)
    bars30 = await get_bars_crypto(CRYPTO_USD_PAIRS, "30Min", limit=120)
    bars60 = await get_bars_crypto(CRYPTO_USD_PAIRS, "1Hour", limit=120)

    ranked: List[Tuple[float, str, List[Dict[str, Any]]]] = []
    for sym in CRYPTO_USD_PAIRS:
        raw15 = (bars15.get("bars") or {}).get(sym, [])
        raw30 = (bars30.get("bars") or {}).get(sym, [])
        raw60 = (bars60.get("bars") or {}).get(sym, [])
        if not raw15 or not raw30 or not raw60:
            continue

        c15 = [float(x["c"]) for x in raw15]
        c30 = [float(x["c"]) for x in raw30]
        c60 = [float(x["c"]) for x in raw60]

        score = rank_score(
            c15, c30, c60,
            conf["rsi_buy"], conf["rsi_sell"],
            conf["ema_fast"], conf["ema_slow"]
        )
        ranked.append((score, sym, raw15))

    ranked.sort(reverse=True)
    rep = (
        f"📡 Сканер (крипта):\n"
        f"• Активних USD-пар: {len(CRYPTO_USD_PAIRS)}\n"
        f"• Використаємо для торгівлі (лімітом): {min(conf['top_n'], len(ranked))}\n"
        f"• Перші 25: " + ", ".join([s for _, s, _ in ranked[:25]]) if ranked else "Немає сигналів"
    )
    return rep, ranked
# ========= Orders =========
async def place_notional_order(sym: str, side: str, notional: float) -> Any:
    payload = {
        "symbol": sym.replace("/", ""),   # для crypto допускається формат без слеша
        "side": side,                     # buy / sell
        "type": "market",
        "time_in_force": "gtc",
        "notional": str(float(notional)),
    }
    return await alp_post("orders", payload)

# ========= Commands =========
async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    stedef(u.effective_chat.id)
    await u.message.reply_text(
        "👋 Бот готовий.\n"
        "• /signals_crypto — скан і, якщо увімкнено, автотрейд\n"
        "• /trade_crypto — миттєвий трейд топ-N\n"
        "• /alp_on /alp_off /alp_status\n"
        "• /long_mode /short_mode /both_mode\n"
        "• Режими ризику: /aggressive /scalp /default /swing /safe\n\n"
        "Short для крипти (spot) **не підтримується** — бот блокує такі заявки.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_keyboard()
    )

async def help_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    await start_cmd(u, c)

async def set_mode(u: Update, mode: str) -> None:
    st = stedef(u.effective_chat.id)
    st["mode"] = mode
    await u.message.reply_text(f"✅ Mode: {mode.upper()}", reply_markup=main_keyboard())

async def aggressive_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE): await set_mode(u, "aggressive")
async def scalp_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):      await set_mode(u, "scalp")
async def default_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):    await set_mode(u, "default")
async def swing_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):      await set_mode(u, "swing")
async def safe_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):       await set_mode(u, "safe")

async def long_mode_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    stedef(u.effective_chat.id)["side_mode"] = "long"
    await u.message.reply_text("🔁 Режим входів: LONG", reply_markup=main_keyboard())

async def short_mode_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    stedef(u.effective_chat.id)["side_mode"] = "short"
    await u.message.reply_text("🔁 Режим входів: SHORT (для крипти буде заблоковано)", reply_markup=main_keyboard())

async def both_mode_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    stedef(u.effective_chat.id)["side_mode"] = "both"
    await u.message.reply_text("🔁 Режим входів: BOTH (крипта short — заблокуємо)", reply_markup=main_keyboard())

async def alp_on_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    stedef(u.effective_chat.id)["autotrade"] = True
    await u.message.reply_text("✅ Alpaca AUTOTRADE: ON", reply_markup=main_keyboard())

async def alp_off_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    stedef(u.effective_chat.id)["autotrade"] = False
    await u.message.reply_text("⏹ Alpaca AUTOTRADE: OFF", reply_markup=main_keyboard())

async def alp_status_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        acc = await alp_account()
        await u.message.reply_text(
            "💼 Alpaca:\n"
            f"• status={acc.get('status','?')}\n"
            f"• cash=${float(acc.get('cash',0)):,.2f}\n"
            f"• buying_power=${float(acc.get('buying_power',0)):,.2f}\n"
            f"• equity=${float(acc.get('equity',0)):,.2f}\n"
            f"Mode={stedef(u.effective_chat.id)['mode']} · "
            f"Autotrade={'ON' if stedef(u.effective_chat.id)['autotrade'] else 'OFF'} · "
            f"Side={stedef(u.effective_chat.id)['side_mode']}",
            reply_markup=main_keyboard()
        )
    except Exception as e:
        await u.message.reply_text(f"❌ Alpaca error: {e}")

# ---- core scanning ----
async def signals_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(u.effective_chat.id)
    try:
        report, ranked = await scan_rank_crypto(st)
        await u.message.reply_text(report)
        if st.get("autotrade") and ranked:
            picks = ranked[: mode_conf(st)["top_n"]]
            for _, sym, arr in picks:
                # напрямок згідно side_mode
                sides = ["buy"] if st.get("side_mode","long") == "long" \
                    else (["sell"] if st.get("side_mode") == "short" else ["buy","sell"])

                for side in sides:
                    if is_crypto_pair(sym) and side == "sell":
                        await u.message.reply_text("Short для крипти не підтримується Alpaca (spot).")
                        continue
                    try:
                        await place_notional_order(sym, side, ALPACA_NOTIONAL)
                        await u.message.reply_text(f"🟢 ORDER OK: {sym} {side.upper()} ${ALPACA_NOTIONAL:.2f}")
                    except Exception as e:
                        await u.message.reply_text(f"🔴 ORDER FAIL {sym} {side.upper()}: {e}")
    except Exception as e:
        await u.message.reply_text(f"🔴 crypto scan error: {e}")

# ---- one-tap trading without extra report ----
async def trade_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(u.effective_chat.id)
    try:
        _, ranked = await scan_rank_crypto(st)
        if not ranked:
            await u.message.reply_text("⚠️ Сигналів недостатньо")
            return
        picks = ranked[: mode_conf(st)["top_n"]]
        for _, sym, _ in picks:
            sides = ["buy"] if st.get("side_mode","long") == "long" \
                else (["sell"] if st.get("side_mode") == "short" else ["buy","sell"])
            for side in sides:
                if is_crypto_pair(sym) and side == "sell":
                    await u.message.reply_text("Short для крипти не підтримується Alpaca (spot).")
                    continue
                try:
                    await place_notional_order(sym, side, ALPACA_NOTIONAL)
                    await u.message.reply_text(f"🟢 ORDER OK: {sym} {side.upper()} ${ALPACA_NOTIONAL:.2f}")
                except Exception as e:
                    await u.message.reply_text(f"🔴 ORDER FAIL {sym} {side.upper()}: {e}")
    except Exception as e:
        await u.message.reply_text(f"🔴 trade_crypto error: {e}")

# ========= Background job =========
async def periodic_scan_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    for chat_id, st in list(STATE.items()):
        try:
            report, ranked = await scan_rank_crypto(st)
            await ctx.bot.send_message(chat_id, report)
            if st.get("autotrade") and ranked:
                picks = ranked[: mode_conf(st)["top_n"]]
                for _, sym, _ in picks:
                    sides = ["buy"] if st.get("side_mode","long") == "long" \
                        else (["sell"] if st.get("side_mode") == "short" else ["buy","sell"])
                    for side in sides:
                        if is_crypto_pair(sym) and side == "sell":
                            await ctx.bot.send_message(chat_id, "Short для крипти не підтримується Alpaca (spot).")
                            continue
                        try:
                            await place_notional_order(sym, side, ALPACA_NOTIONAL)
                            await ctx.bot.send_message(chat_id, f"🟢 ORDER OK: {sym} {side.upper()} ${ALPACA_NOTIONAL:.2f}")
                        except Exception as e:
                            await ctx.bot.send_message(chat_id, f"🔴 ORDER FAIL {sym} {side.upper()}: {e}")
        except Exception as e:
            try:
                await ctx.bot.send_message(chat_id, f"🔴 periodic_scan error: {e}")
            except Exception:
                pass

# ========= APP =========
def main():
    if not TG_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задано")

    app = Application.builder().token(TG_TOKEN).build()

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

    app.job_queue.run_repeating(periodic_scan_job, interval=SCAN_EVERY_SEC, first=5)

    app.run_polling()

if __name__ == "__main__":
    main()
