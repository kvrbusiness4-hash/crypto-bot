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

# інтервал фонового автоскану в секундах
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC") or 300)  # 5 хв за замовчуванням
DEDUP_COOLDOWN_MIN = int(os.getenv("DEDUP_COOLDOWN_MIN") or 240)  # хвилини антидубля

# балансний буфер та мінімальна сума на ордер для safe-buy
SAFE_BUF = float(os.getenv("SAFE_BUF", 0.98))  # використовуємо лише 98% доступних USD
MIN_ORDER_NOTIONAL = float(os.getenv("MIN_ORDER_NOTIONAL", 5))  # мінімальна сума на ордер
PRICE_DECIMALS = int(os.getenv("PRICE_DECIMALS") or 6)  # округлення для цін

# ====== GLOBAL STATE (per chat) ======

STATE: Dict[int, Dict[str, Any]] = {}

# ====== MODE PROFILES (таймфрейми, фільтри, ризики) ======

MODE_PARAMS = {
    "aggressive": {
        "bars": ("15Min", "30Min", "1Hour"),
        "rsi_buy": 55.0,
        "rsi_sell": 45.0,
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

# ====== CRYPTO WHITELIST (USD пари) ======

CRYPTO_USD_PAIRS = [
    "BTC/USD","ETH/USD","SOL/USD","LTC/USD","DOGE/USD","AVAX/USD","AAVE/USD","MKR/USD","DOT/USD",
    "LINK/USD","UNI/USD","PEPE/USD","XRP/USD","TRUMP/USD","CRV/USD","BCH/USD","BAT/USD","GRT/USD",
    "XTZ/USD","USDC/USD","USDT/USD","USDG/USD","YFI/USD","LDO/USD"
][:ALPACA_MAX_CRYPTO]

# ====== STOCKS UNIVERSE (можеш підредагувати) ======

STOCKS_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","ADBE","CRM","ORCL","AMD","AMAT","INTC","CSCO","QCOM",
    "BAC","JPM","GS","BRK.B","V","MA","KO","PEP","MCD","NKE",
    "SPY","QQQ","IWM","DIA","XLF","XLK","XLV","XLE","XLY","XLP",
][:ALPACA_MAX_STOCKS]

# ============ HELPERS (timeframe, symbols, dedup, http) ============

def map_tf(tf: str) -> str:
    """Alpaca data API не приймає 60Min — треба 1Hour."""
    t = (tf or "").strip()
    return "1Hour" if t.lower() in ("60min", "60", "1h", "60мин", "60мін") else t

def to_order_sym(sym: str) -> str:
    return sym.replace("/", "").upper()

def to_data_sym(sym: str) -> str:
    """BTC/USD -> BTC/USD; AAPL -> AAPL (для stocks залишаємо як є)."""
    s = (sym or "").replace(" ", "").upper()
    if "/" in s:
        return s
    if s.endswith("USD"):
        return s[:-3] + "/USD"
    return s

def is_crypto_sym(sym: str) -> bool:
    """Криптопара має вигляд 'BTC/USD' тощо."""
    return "/" in (sym or "")

def now_s() -> float:
    return time.time()

RECENT_TRADES: Dict[str, float] = {}  # "CRYPTO|AAVEUSD|buy" або "STOCK|AAPL|buy"
def skip_as_duplicate_cooldown(market: str, sym: str, side: str) -> bool:
    """Антидубль за часом (кулдаун)."""
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
    st.setdefault("auto_scan", False)  # фоновий автоскан
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

# ===== helper: account/clock =====

async def alp_clock() -> Dict[str, Any]:
    return await alp_get_json("/v2/clock")

async def usd_available() -> float:
    acc = await alp_get_json("/v2/account")
    return float(acc.get("cash") or 0.0)

def parse_available_from_error(msg: str) -> Optional[float]:
    try:
        jtxt = msg[msg.find("{"):]
        j = json.loads(jtxt)
        return float(j.get("available"))
    except Exception:
        return None

async def has_open_position(sym: str) -> bool:
    """Чи є відкрита довга/коротка позиція по символу."""
    try:
        # Для крипти — /v2/positions/{symbol без слеша}; AAVEUSD
        s = to_order_sym(sym)
        pos = await alp_get_json(f"/v2/positions/{s}")
        qty = float(pos.get("qty", 0) or 0)
        return abs(qty) > 0
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

def rank_score(
    c15: List[float], c30: List[float], c60: List[float],
    rsi_buy: float, rsi_sell: float,
    ema_fast_p: int, ema_slow_p: int
) -> float:
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
    # простий скор: сильніше за RSI на малій ТФ + тренд
    return bias * 100 + trend * 50 - abs(50.0 - r1)

# -------- SCAN (CRYPTO) --------

async def scan_rank_crypto(st: Dict[str, Any]) -> Tuple[str, List[Tuple[float, str, List[Dict[str, Any]]]]]:
    conf = _mode_conf(st)
    tf15, tf30, tf60 = map_tf(conf["bars"][0]), map_tf(conf["bars"][1]), map_tf(conf["bars"][2])

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

# -------- SCAN (STOCKS) --------

async def scan_rank_stocks(st: Dict[str, Any]) -> Tuple[str, List[Tuple[float, str, List[Dict[str, Any]]]]]:
    conf = _mode_conf(st)
    tf15, tf30, tf60 = map_tf(conf["bars"][0]), map_tf(conf["bars"][1]), map_tf(conf["bars"][2])

    symbols = STOCKS_UNIVERSE[:]

    bars15 = await get_bars_stocks(symbols, tf15, limit=120)
    bars30 = await get_bars_stocks(symbols, tf30, limit=120)
    bars60 = await get_bars_stocks(symbols, tf60, limit=120)

    ranked: List[Tuple[float, str, List[Dict[str, Any]]]] = []
    for sym in symbols:
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
        "📡 Сканер (акції):\n"
        f"• Символів у списку: {len(symbols)}\n"
        f"• Використаємо для торгівлі (лімітом): {min(conf['top_n'], len(ranked))}\n"
        + (f"• Перші 25: " + ", ".join([s for _, s, _ in ranked[:25]])
           if ranked else "• Немає сигналів")
    )
    return rep, ranked

# -------- RISK/TP/SL UTILS --------

def calc_sl_tp(side: str, px: float, conf: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    tp_pct = float(conf.get("tp_pct", 0.012))
    sl_pct = float(conf.get("sl_pct", 0.008))
    if side == "buy":
        tp = px * (1.0 + tp_pct)
        sl = px * (1.0 - sl_pct)
    else:
        tp = px * (1.0 - tp_pct)
        sl = px * (1.0 + sl_pct)
    return round(tp, PRICE_DECIMALS), round(sl, PRICE_DECIMALS)

# -------- SAFE BUY (MARKET, з урахуванням балансу) --------

async def safe_notional(base_notional: float) -> Optional[float]:
    avail = await usd_available()
    use = min(base_notional, avail * SAFE_BUF)
    return use if use >= MIN_ORDER_NOTIONAL else None

async def place_safe_market_buy_crypto(sym: str, base_notional: float) -> Dict[str, Any]:
    use = await safe_notional(base_notional)
    if use is None:
        raise RuntimeError("low balance for crypto buy")
    payload = {
        "symbol": to_order_sym(sym),
        "side": "buy",
        "type": "market",
        "time_in_force": "gtc",
        "notional": f"{use:.2f}",
    }
    try:
        return await alp_post_json("/v2/orders", payload)
    except RuntimeError as e:
        msg = str(e)
        if "403" in msg and "insufficient balance" in msg:
            avail = parse_available_from_error(msg)
            if avail and avail >= MIN_ORDER_NOTIONAL:
                payload["notional"] = f"{avail * SAFE_BUF:.2f}"
                return await alp_post_json("/v2/orders", payload)
        raise

async def place_safe_market_buy_stock(sym: str, base_notional: float) -> Dict[str, Any]:
    use = await safe_notional(base_notional)
    if use is None:
        raise RuntimeError("low balance for stock buy")
    payload = {
        "symbol": to_order_sym(sym),
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "notional": f"{use:.2f}",
    }
    try:
        return await alp_post_json("/v2/orders", payload)
    except RuntimeError as e:
        msg = str(e)
        if "403" in msg and "insufficient balance" in msg:
            avail = parse_available_from_error(msg)
            if avail and avail >= MIN_ORDER_NOTIONAL:
                payload["notional"] = f"{avail * SAFE_BUF:.2f}"
                return await alp_post_json("/v2/orders", payload)
        raise

# -------- CHILD ORDERS (TP/SL як окремі) --------

async def fetch_position_qty(sym: str) -> Optional[float]:
    """Повертає qty відкритої позиції після покупки (якщо вже з’явилась)."""
    try:
        pos = await alp_get_json(f"/v2/positions/{to_order_sym(sym)}")
        q = pos.get("qty")
        if q is None:
            return None
        return float(q)
    except Exception:
        return None

async def place_tp_sl_children(sym: str, side: str, entry_px: float, conf: Dict[str, Any]) -> List[str]:
    """
    Ставить два ордери: TP (limit) і SL (stop) під поточну позицію.
    Беремо фактичну qty з /v2/positions/{sym}; якщо не встигає з’явитись — приблизно рахуємо.
    """
    msgs = []
    # беремо фактичну кількість (даємо невеличкий час біржі оновити позицію)
    qty = None
    for _ in range(3):
        qty = await fetch_position_qty(sym)
        if qty and qty > 0:
            break
        await asyncio.sleep(0.7)

    if not qty or qty <= 0:
        # fallback: приблизна кількість за останньою ціною
        qty = round(ALPACA_NOTIONAL / max(1e-9, entry_px), 6)

    tp, sl = calc_sl_tp("buy", entry_px, conf)

    # TP: limit sell
    payload_tp = {
        "symbol": to_order_sym(sym),
        "side": "sell",
        "type": "limit",
        "time_in_force": "gtc",
        "limit_price": f"{tp:.{PRICE_DECIMALS}f}",
        "qty": f"{qty}",
    }
    try:
        await alp_post_json("/v2/orders", payload_tp)
        msgs.append(f"TP @ {tp:.{PRICE_DECIMALS}f}")
    except Exception as e:
        msgs.append(f"TP FAIL: {e}")

    # SL: stop (market при спрацюванні)
    payload_sl = {
        "symbol": to_order_sym(sym),
        "side": "sell",
        "type": "stop",
        "time_in_force": "gtc",
        "stop_price": f"{sl:.{PRICE_DECIMALS}f}",
        "qty": f"{qty}",
    }
    try:
        await alp_post_json("/v2/orders", payload_sl)
        msgs.append(f"SL @ {sl:.{PRICE_DECIMALS}f}")
    except Exception as e:
        msgs.append(f"SL FAIL: {e}")

    return msgs

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
        "/alp_on /alp_off /alp_status — автотрейд (дозвіл виставляти ордери)\n"
        "/auto_on /auto_off /auto_status — фоновий автоскан ринку\n"
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
            f"AutoScan={'ON' if st.get('auto_scan') else 'OFF'} · "
            f"Side={st.get('side_mode','long')}"
        )
        await u.message.reply_text(txt)
    except Exception as e:
        await u.message.reply_text(f"🔴 alp_status error: {e}")

# ------- CRYPTO commands -------

async def _do_crypto_picks(u: Update, st: Dict[str, Any], ranked):
    if not ranked:
        await u.message.reply_text("⚠️ Немає сигналів")
        return
    picks = ranked[: _mode_conf(st)["top_n"]]
    # локальний бюджет, щоб не спамити 403
    budget = await usd_available()

    for _, sym, arr in picks:
        if budget < MIN_ORDER_NOTIONAL:
            break
        if await has_open_position(sym):
            await u.message.reply_text(f"⚪ SKIP (позиція вже відкрита): {sym}")
            continue
        side = "buy"
        px = float(arr[-1]["c"])
        conf = _mode_conf(st)
        sl, tp = calc_sl_tp(side, px, conf)

        # антидубль за кулдауном
        if skip_as_duplicate_cooldown("CRYPTO", sym, side):
            await u.message.reply_text(f"⚪ SKIP (дубль): {sym} {side.upper()}")
            continue

        use_notional = min(ALPACA_NOTIONAL, budget * SAFE_BUF)
        if use_notional < MIN_ORDER_NOTIONAL:
            break

        try:
            resp = await place_safe_market_buy_crypto(sym, use_notional)
            budget -= use_notional
            # після купівлі — ставимо TP/SL як окремі ордери
            notes = await place_tp_sl_children(sym, side, px, conf)
            await u.message.reply_text(
                f"🟢 ORDER OK: {sym} BUY ${use_notional:.2f}\n"
                + (" · ".join(notes) if notes else f"TP:{tp:.{PRICE_DECIMALS}f} SL:{sl:.{PRICE_DECIMALS}f}")
            )
        except Exception as e:
            await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")

async def signals_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        report, ranked = await scan_rank_crypto(st)
        await u.message.reply_text(report)
        if st.get("autotrade"):
            await _do_crypto_picks(u, st, ranked)
    except Exception as e:
        await u.message.reply_text(f"🔴 signals_crypto error: {e}")

async def trade_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        _, ranked = await scan_rank_crypto(st)
        await _do_crypto_picks(u, st, ranked)
    except Exception as e:
        await u.message.reply_text(f"🔴 trade_crypto error: {e}")

# ------- STOCKS commands -------

async def _do_stock_picks(u: Update, st: Dict[str, Any], ranked):
    if not ranked:
        await u.message.reply_text("⚠️ Немає сигналів")
        return

    # Перевіряємо, чи ринок відкритий
    try:
        clk = await alp_clock()
        market_open = bool(clk.get("is_open"))
    except Exception:
        market_open = True

    if not market_open:
        await u.message.reply_text("⏸ Ринок акцій закритий — ордери не виставляю.")
        return

    picks = ranked[: _mode_conf(st)["top_n"]]
    budget = await usd_available()

    for _, sym, arr in picks:
        if budget < MIN_ORDER_NOTIONAL:
            break
        if await has_open_position(sym):
            await u.message.reply_text(f"⚪ SKIP (позиція вже відкрита): {sym}")
            continue
        side = "buy"
        px = float(arr[-1]["c"])
        conf = _mode_conf(st)
        sl, tp = calc_sl_tp(side, px, conf)

        if skip_as_duplicate_cooldown("STOCK", sym, side):
            await u.message.reply_text(f"⚪ SKIP (дубль): {sym} {side.upper()}")
            continue

        use_notional = min(ALPACA_NOTIONAL, budget * SAFE_BUF)
        if use_notional < MIN_ORDER_NOTIONAL:
            break

        try:
            resp = await place_safe_market_buy_stock(sym, use_notional)
            budget -= use_notional
            notes = await place_tp_sl_children(sym, side, px, conf)
            await u.message.reply_text(
                f"🟢 ORDER OK: {sym} BUY ${use_notional:.2f}\n"
                + (" · ".join(notes) if notes else f"TP:{tp:.{PRICE_DECIMALS}f} SL:{sl:.{PRICE_DECIMALS}f}")
            )
        except Exception as e:
            await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")

async def signals_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        report, ranked = await scan_rank_stocks(st)
        await u.message.reply_text(report)
        if st.get("autotrade"):
            await _do_stock_picks(u, st, ranked)
    except Exception as e:
        await u.message.reply_text(f"🔴 signals_stocks error: {e}")

async def trade_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        _, ranked = await scan_rank_stocks(st)
        await _do_stock_picks(u, st, ranked)
    except Exception as e:
        await u.message.reply_text(f"🔴 trade_stocks error: {e}")

# ======= AUTOSCAN (background) =======

async def _auto_scan_once_for_chat(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE):
    st = stdef(chat_id)
    if not st.get("auto_scan") or not st.get("autotrade"):
        return

    conf = _mode_conf(st)
    top_n = int(conf.get("top_n", max(1, ALPACA_TOP_N)))

    # clock для акцій
    try:
        clk = await alp_clock()
        market_open = bool(clk.get("is_open"))
    except Exception:
        market_open = True

    # скани
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

    budget = await usd_available()

    for score, sym, kind, arr in picks:
        if budget < MIN_ORDER_NOTIONAL:
            break
        if kind == "stock":
            if not market_open:
                continue
        if await has_open_position(sym):
            await ctx.bot.send_message(chat_id, f"⚪ SKIP (позиція вже відкрита): {to_order_sym(sym)}")
            continue

        side = "buy"
        px = float(arr[-1]["c"])
        sl, tp = calc_sl_tp(side, px, conf)

        if skip_as_duplicate_cooldown("STOCK" if kind == "stock" else "CRYPTO", sym, side):
            continue

        use_notional = min(ALPACA_NOTIONAL, budget * SAFE_BUF)
        if use_notional < MIN_ORDER_NOTIONAL:
            break

        try:
            if kind == "stock":
                await place_safe_market_buy_stock(sym, use_notional)
            else:
                await place_safe_market_buy_crypto(sym, use_notional)
            budget -= use_notional
            notes = await place_tp_sl_children(sym, side, px, conf)
            await ctx.bot.send_message(
                chat_id,
                f"🟢 AUTO ORDER: {to_order_sym(sym)} BUY ${use_notional:.2f} · " + " · ".join(notes)
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

    # Крипта
    app.add_handler(CommandHandler("signals_crypto", signals_crypto))
    app.add_handler(CommandHandler("trade_crypto", trade_crypto))
    # Акції
    app.add_handler(CommandHandler("signals_stocks", signals_stocks))
    app.add_handler(CommandHandler("trade_stocks", trade_stocks))

    # Автоскан
    app.add_handler(CommandHandler("auto_on", auto_on))
    app.add_handler(CommandHandler("auto_off", auto_off))
    app.add_handler(CommandHandler("auto_status", auto_status))

    # фоновий job раз у SCAN_INTERVAL_SEC
    app.job_queue.run_repeating(periodic_auto_scan, interval=SCAN_INTERVAL_SEC, first=10)

    print("Bot started.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
