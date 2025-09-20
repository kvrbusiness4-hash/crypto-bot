# -*- coding: utf-8 -*-

import os
import json
import math
import asyncio
import time
from typing import Dict, Any, Tuple, List, Optional

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
ALPACA_MAX_STOCKS = int(os.getenv("ALPACA_MAX_STOCKS") or 50)

SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC") or 300)
DEDUP_COOLDOWN_MIN = int(os.getenv("DEDUP_COOLDOWN_MIN") or 240)

# ====== GLOBAL STATE (per chat) ======
STATE: Dict[int, Dict[str, Any]] = {}

# ====== MODE PROFILES ======
MODE_PARAMS = {
    "aggressive": {
        "bars": ("15Min", "30Min", "1Hour"),
        "rsi_buy": 55.0, "rsi_sell": 45.0,
        "ema_fast": 15, "ema_slow": 30,
        "top_n": ALPACA_TOP_N,
        "tp_pct": 0.015,
        "sl_pct": 0.008,
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

# ====== CRYPTO WHITELIST ======
CRYPTO_USD_PAIRS = [
    "BTC/USD","ETH/USD","SOL/USD","LTC/USD","DOGE/USD","AVAX/USD","AAVE/USD","MKR/USD","DOT/USD",
    "LINK/USD","UNI/USD","PEPE/USD","XRP/USD","TRUMP/USD","CRV/USD","BCH/USD","BAT/USD","GRT/USD",
    "XTZ/USD","USDC/USD","USDT/USD","USDG/USD","YFI/USD","LDO/USD"
][:ALPACA_MAX_CRYPTO]

# ====== STOCKS UNIVERSE ======
STOCKS_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","ADBE","CRM","ORCL","AMD","AMAT","INTC","CSCO","QCOM",
    "BAC","JPM","GS","BRK.B","V","MA","KO","PEP","MCD","NKE",
    "SPY","QQQ","IWM","DIA","XLF","XLK","XLV","XLE","XLY","XLP",
][:ALPACA_MAX_STOCKS]

# ============ HELPERS ============
def map_tf(tf: str) -> str:
    t = (tf or "").strip()
    return "1Hour" if t.lower() in ("60min", "60", "1h", "60мин", "60мін") else t

def to_order_sym(sym: str) -> str:
    return sym.replace("/", "").upper()

def to_data_sym(sym: str) -> str:
    s = (sym or "").replace(" ", "").upper()
    if "/" in s:
        return s
    if s.endswith("USD"):
        return s[:-3] + "/USD"
    return s

def is_crypto_sym(sym: str) -> bool:
    return "/" in (sym or "")

def now_s() -> float:
    return time.time()

RECENT_TRADES: Dict[str, float] = {}
def skip_as_duplicate(market: str, sym: str, side: str) -> bool:
    key = f"{market}|{to_order_sym(sym)}|{side.lower()}"
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
    st.setdefault("auto_scan", False)
    st.setdefault("side_mode", "long")
    return st

def kb() -> ReplyKeyboardMarkup:
    rows = [
        ["/aggressive", "/scalp", "/default"],
        ["/swing", "/safe", "/help"],
        ["/signals_crypto", "/trade_crypto"],
        ["/signals_stocks", "/trade_stocks"],
        ["/alp_on", "/alp_status", "/alp_off"],
        ["/auto_on", "/auto_status", "/auto_off"],
        ["/long_mode", "/short_mode", "/both_mode"],
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
            txt = await r.text()
            if r.status >= 400:
                raise RuntimeError(f"GET {path} {r.status}: {txt}")
            return json.loads(txt) if txt else {}

async def alp_post_json(path: str, payload: Dict[str, Any]) -> Any:
    url = f"{ALPACA_BASE_URL}{path}"
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        async with s.post(url, headers=_alp_headers(), data=json.dumps(payload)) as r:
            txt = await r.text()
            if r.status >= 400:
                raise RuntimeError(f"POST {path} {r.status}: {txt}")
            return json.loads(txt) if txt else {}

# ===== helper: clock & positions =====
async def alp_clock() -> Dict[str, Any]:
    return await alp_get_json("/v2/clock")

async def alp_positions() -> List[Dict[str, Any]]:
    return await alp_get_json("/v2/positions")

async def has_open_long(sym: str) -> bool:
    try:
        pos = await alp_get_json(f"/v2/positions/{to_order_sym(sym)}")
        qty = float(pos.get("qty", 0) or 0)
        return qty > 0
    except Exception:
        return False

# -------- DATA: /bars ----------
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

async def get_bars_stocks(symbols: List[str], timeframe: str, limit: int = 120) -> Dict[str, Any]:
    tf = map_tf(timeframe)
    syms = ",".join([s.upper() for s in symbols])
    path = f"/v2/stocks/bars"
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
    return bias * 100.0 + trend * 50.0 - abs(50.0 - r1)

def calc_sl_tp(side: str, price: float, conf: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    tp_pct = float(conf.get("tp_pct", 0.01))
    sl_pct = float(conf.get("sl_pct", 0.008))
    if side.lower() == "buy":
        tp = price * (1.0 + tp_pct)
        sl = price * (1.0 - sl_pct)
        return (tp, sl)
    else:
        tp = price * (1.0 - tp_pct)
        sl = price * (1.0 + sl_pct)
        return (tp, sl)

# ======== ORDERS (оновлено) ========

def _round_crypto_qty(qty: float) -> float:
    return float(f"{qty:.6f}")

def _round_stock_qty(qty: float) -> float:
    return round(qty, 3)

# ---- STOCK buy by NOTIONAL (як було) ----
async def place_market_buy_notional(sym: str, notional: float) -> dict:
    payload = {
        "symbol": to_order_sym(sym),
        "side": "buy",
        "type": "market",
        "time_in_force": "gtc",
        "notional": f"{notional}",
    }
    return await alp_post_json("/v2/orders", payload)

# ---- CRYPTO buy by QTY (НОВЕ) ----
async def place_market_buy_crypto_qty(sym: str, qty: float) -> dict:
    qty = _round_crypto_qty(qty)
    payload = {
        "symbol": to_order_sym(sym),
        "side": "buy",
        "type": "market",
        "time_in_force": "gtc",
        "qty": f"{qty}",
    }
    return await alp_post_json("/v2/orders", payload)

async def get_order(order_id: str) -> dict:
    return await alp_get_json(f"/v2/orders/{order_id}")

async def place_tp_sl_as_simple_sells(sym: str, filled_qty: float, tp: float | None, sl: float | None, is_crypto: bool):
    if filled_qty <= 0:
        return
    qty = _round_crypto_qty(filled_qty) if is_crypto else _round_stock_qty(filled_qty)

    if tp is not None:
        payload_tp = {
            "symbol": to_order_sym(sym),
            "side": "sell",
            "type": "limit",
            "time_in_force": "gtc",
            "limit_price": f"{tp:.6f}",
            "qty": f"{qty}",
        }
        await alp_post_json("/v2/orders", payload_tp)

    if sl is not None:
        payload_sl = {
            "symbol": to_order_sym(sym),
            "side": "sell",
            "type": "stop",
            "time_in_force": "gtc",
            "stop_price": f"{sl:.6f}",
            "qty": f"{qty}",
        }
        await alp_post_json("/v2/orders", payload_sl)

# ---- CRYPTO bracket via qty ----
async def place_bracket_notional_order_crypto(sym: str, side: str, notional: float,
                                              tp: float | None, sl: float | None, last_price: float) -> Any:
    if side.lower() != "buy":
        raise RuntimeError("crypto: лише long buy підтримано")
    # конвертація notional -> qty
    qty = max(0.000001, notional / max(1e-12, last_price))
    buy = await place_market_buy_crypto_qty(sym, qty)

    order_id = buy.get("id", "")
    filled_qty = 0.0
    for _ in range(12):
        od = await get_order(order_id)
        status = od.get("status")
        if status in ("filled", "partially_filled"):
            filled_qty = float(od.get("filled_qty") or 0)
            if status == "filled":
                break
        await asyncio.sleep(0.7)

    await place_tp_sl_as_simple_sells(sym, filled_qty, tp, sl, is_crypto=True)
    return buy

# ---- STOCK bracket via notional ----
async def place_bracket_notional_order_stock(sym: str, side: str, notional: float,
                                             tp: float | None, sl: float | None) -> Any:
    if side.lower() != "buy":
        raise RuntimeError("stocks: лише long buy підтримано")
    buy = await place_market_buy_notional(sym, notional)

    order_id = buy.get("id", "")
    filled_qty = 0.0
    for _ in range(12):
        od = await get_order(order_id)
        status = od.get("status")
        if status in ("filled", "partially_filled"):
            filled_qty = float(od.get("filled_qty") or 0)
            if status == "filled":
                break
        await asyncio.sleep(0.7)

    await place_tp_sl_as_simple_sells(sym, filled_qty, tp, sl, is_crypto=False)
    return buy

# -------- COMMANDS --------
async def start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    stdef(u.effective_chat.id)
    await u.message.reply_text(
        "👋 Алпака-бот готовий.\n"
        "Крипта торгується 24/7; акції — коли ринок відкритий. Сканер/автотрейд може працювати у фоні.\n"
        "Увімкнути автотрейд: /alp_on  ·  Зупинити: /alp_off  ·  Стан: /alp_status\n"
        "Фоновий автоскан: /auto_on  ·  /auto_off  ·  /auto_status",
        reply_markup=kb()
    )

async def help_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "/signals_crypto — показати топ-N і (якщо Autotrade=ON) поставити ордери\n"
        "/trade_crypto — миттєво торгувати топ-N без додаткового звіту\n"
        "/signals_stocks — показати топ-N для акцій\n"
        "/trade_stocks — миттєво торгувати топ-N акцій\n"
        "/alp_on /alp_off /alp_status — автотрейд\n"
        "/auto_on /auto_off /auto_status — фоновий автоскан\n"
        "/long_mode /short_mode /both_mode — напрям\n"
        "/aggressive /scalp /default /swing /safe — профілі",
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
    await u.message.reply_text("Режим входів: SHORT (для крипти ігнорується)")

async def both_mode(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    st["side_mode"] = "both"
    await u.message.reply_text("Режим входів: BOTH (для крипти лише LONG)")

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
            f"AutoScan={'ON' if st.get('auto_scan') else 'OFF'} · "
            f"Side={st.get('side_mode','long')} · Notional=${ALPACA_NOTIONAL:.2f}"
        )
        await u.message.reply_text(txt)
    except Exception as e:
        await u.message.reply_text(f"🔴 alp_status error: {e}")

# ------- CRYPTO commands -------
async def signals_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        report, ranked = await scan_rank_crypto(st)
        await u.message.reply_text(report)

        if not st.get("autotrade") or not ranked:
            return

        picks = ranked[: _mode_conf(st)["top_n"]]
        for _, sym, arr in picks:
            side = "buy"
            px = float(arr[-1]["c"])
            conf = _mode_conf(st)
            tp, sl = calc_sl_tp(side, px, conf)

            if await has_open_long(sym):
                await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}")
                continue

            if skip_as_duplicate("CRYPTO", sym, side):
                await u.message.reply_text(f"⚪ SKIP (дубль): {sym} {side.upper()}")
                continue

            try:
                await place_bracket_notional_order_crypto(sym, side, ALPACA_NOTIONAL, tp, sl, px)
                await u.message.reply_text(
                    f"🟢 ORDER OK: {sym} BUY ${ALPACA_NOTIONAL:.2f}\n"
                    f"Entry≈{px:.6f} · TP:{tp:.6f} SL:{sl:.6f} (окремими ордерами)"
                )
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")

    except Exception as e:
        await u.message.reply_text(f"🔴 signals_crypto error: {e}")

async def trade_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE):
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
            tp, sl = calc_sl_tp(side, px, conf)

            if await has_open_long(sym):
                await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}")
                continue

            if skip_as_duplicate("CRYPTO", sym, side):
                await u.message.reply_text(f"⚪ SKIP (дубль): {sym} {side.upper()}")
                continue

            try:
                await place_bracket_notional_order_crypto(sym, side, ALPACA_NOTIONAL, tp, sl, px)
                await u.message.reply_text(
                    f"🟢 ORDER OK: {sym} BUY ${ALPACA_NOTIONAL:.2f}\n"
                    f"Entry≈{px:.6f} · TP:{tp:.6f} SL:{sl:.6f} (окремими ордерами)"
                )
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")
    except Exception as e:
        await u.message.reply_text(f"🔴 trade_crypto error: {e}")

# ------- STOCKS commands -------
async def signals_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        report, ranked = await scan_rank_stocks(st)
        await u.message.reply_text(report)

        if not st.get("autotrade") or not ranked:
            return

        # чи відкритий ринок
        try:
            clk = await alp_clock()
            market_open = bool(clk.get("is_open"))
        except Exception:
            market_open = True
        if not market_open:
            await u.message.reply_text("⏸ Ринок акцій закритий — ордери не виставляю.")
            return

        picks = ranked[: _mode_conf(st)["top_n"]]
        for _, sym, arr in picks:
            side = "buy"
            px = float(arr[-1]["c"])
            conf = _mode_conf(st)
            tp, sl = calc_sl_tp(side, px, conf)

            if await has_open_long(sym):
                await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}")
                continue

            if skip_as_duplicate("STOCK", sym, side):
                await u.message.reply_text(f"⚪ SKIP (дубль): {sym} {side.upper()}")
                continue

            try:
                await place_bracket_notional_order_stock(sym, side, ALPACA_NOTIONAL, tp, sl)
                await u.message.reply_text(
                    f"🟢 ORDER OK: {sym} BUY ${ALPACA_NOTIONAL:.2f}\n"
                    f"TP:{tp:.6f} SL:{sl:.6f} (окремими ордерами)"
                )
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")

    except Exception as e:
        await u.message.reply_text(f"🔴 signals_stocks error: {e}")

async def trade_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        _, ranked = await scan_rank_stocks(st)
        if not ranked:
            await u.message.reply_text("⚠️ Немає сигналів")
            return

        try:
            clk = await alp_clock()
            market_open = bool(clk.get("is_open"))
        except Exception:
            market_open = True
        if not market_open:
            await u.message.reply_text("⏸ Ринок акцій закритий — ордери не виставляю.")
            return

        picks = ranked[: _mode_conf(st)["top_n"]]
        for _, sym, arr in picks:
            side = "buy"
            px = float(arr[-1]["c"])
            conf = _mode_conf(st)
            tp, sl = calc_sl_tp(side, px, conf)

            if await has_open_long(sym):
                await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}")
                continue

            if skip_as_duplicate("STOCK", sym, side):
                await u.message.reply_text(f"⚪ SKIP (дубль): {sym} {side.upper()}")
                continue

            try:
                await place_bracket_notional_order_stock(sym, side, ALPACA_NOTIONAL, tp, sl)
                await u.message.reply_text(
                    f"🟢 ORDER OK: {sym} BUY ${ALPACA_NOTIONAL:.2f}\n"
                    f"TP:{tp:.6f} SL:{sl:.6f} (окремими ордерами)"
                )
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")
    except Exception as e:
        await u.message.reply_text(f"🔴 trade_stocks error: {e}")

# ======= AUTOSCAN (background) =======
async def _auto_scan_once_for_chat(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE):
    st = stdef(chat_id)
    if not st.get("auto_scan") or not st.get("autotrade"):
        return

    conf = _mode_conf(st)
    top_n = int(conf.get("top_n", max(1, ALPACA_TOP_N)))

    try:
        clk = await alp_clock()
        market_open = bool(clk.get("is_open"))
    except Exception:
        market_open = True

    try:
        crypto_report, crypto_ranked = await scan_rank_crypto(st)
    except Exception as e:
        crypto_report, crypto_ranked = (f"🔴 Крипто-скан помилка: {e}", [])

    try:
        stocks_report, stocks_ranked = await scan_rank_stocks(st)
    except Exception as e:
        stocks_report, stocks_ranked = (f"🔴 Скан акцій помилка: {e}", [])

    combined: List[Tuple[float, str, str, List[Dict[str, Any]]]] = []
    for sc, sym, arr in crypto_ranked:
        combined.append((sc, sym, "crypto", arr))
    for sc, sym, arr in stocks_ranked:
        combined.append((sc, sym, "stock", arr))
    combined.sort(reverse=True)
    picks = combined[:top_n]

    for score, sym, kind, arr in picks:
        if kind == "stock" and not market_open:
            continue
        if await has_open_long(sym):
            continue

        side = "buy"
        px = float(arr[-1]["c"])
        tp, sl = calc_sl_tp(side, px, conf)

        if skip_as_duplicate("STOCK" if kind == "stock" else "CRYPTO", sym, side):
            continue

        try:
            if kind == "stock":
                await place_bracket_notional_order_stock(sym, side, ALPACA_NOTIONAL, tp, sl)
            else:
                await place_bracket_notional_order_crypto(sym, side, ALPACA_NOTIONAL, tp, sl, px)
            await ctx.bot.send_message(
                chat_id,
                f"🟢 AUTO ORDER: {to_order_sym(sym)} BUY ${ALPACA_NOTIONAL:.2f} · TP:{(tp or 0):.6f} SL:{(sl or 0):.6f}"
            )
        except Exception as e:
            await ctx.bot.send_message(chat_id, f"🔴 AUTO ORDER FAIL {sym}: {e}")

async def periodic_auto_scan(ctx: ContextTypes.DEFAULT_TYPE):
    for chat_id in list(STATE.keys()):
        try:
            await _auto_scan_once_for_chat(chat_id, ctx)
        except Exception as e:
            try:
                await ctx.bot.send_message(chat_id, f"🔴 periodic autoscan error: {e}")
            except Exception:
                pass

# ------- AUTOSCAN commands -------
async def auto_on(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    st["auto_scan"] = True
    await u.message.reply_text(f"✅ AUTO-SCAN: ON (кожні {SCAN_INTERVAL_SEC}s)")

async def auto_off(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    st["auto_scan"] = False
    await u.message.reply_text("⛔ AUTO-SCAN: OFF")

async def auto_status(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    await u.message.reply_text(
        f"AutoScan={'ON' if st.get('auto_scan') else 'OFF'}; "
        f"Autotrade={'ON' if st.get('autotrade') else 'OFF'}; "
        f"Mode={st.get('mode','default')} · Side={st.get('side_mode','long')} · "
        f"Interval={SCAN_INTERVAL_SEC}s"
    )

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

    app.add_handler(CommandHandler("signals_stocks", signals_stocks))
    app.add_handler(CommandHandler("trade_stocks", trade_stocks))

    app.add_handler(CommandHandler("auto_on", auto_on))
    app.add_handler(CommandHandler("auto_off", auto_off))
    app.add_handler(CommandHandler("auto_status", auto_status))

    app.job_queue.run_repeating(periodic_auto_scan, interval=SCAN_INTERVAL_SEC, first=10)

    print("Bot started.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
