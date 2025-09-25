# -*- coding: utf-8 -*-

import os, json, math, asyncio, time, re
from typing import Dict, Any, Tuple, List, Optional
from aiohttp import ClientSession, ClientTimeout
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes

# ========= ENV =========
TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_TOKEN") or "").strip()

ALPACA_API_KEY    = (os.getenv("ALPACA_API_KEY") or "").strip()
ALPACA_API_SECRET = (os.getenv("ALPACA_API_SECRET") or "").strip()
ALPACA_BASE_URL   = (os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_URL   = (os.getenv("ALPACA_DATA_URL") or "https://data.alpaca.markets").rstrip("/")

ALPACA_NOTIONAL   = float(os.getenv("ALPACA_NOTIONAL") or 50)
ALPACA_TOP_N      = int(os.getenv("ALPACA_TOP_N") or 2)
ALPACA_MAX_CRYPTO = int(os.getenv("ALPACA_MAX_CRYPTO") or 25)
ALPACA_MAX_STOCKS = int(os.getenv("ALPACA_MAX_STOCKS") or 50)

SCAN_INTERVAL_SEC   = int(os.getenv("SCAN_INTERVAL_SEC") or 300)
DEDUP_COOLDOWN_MIN  = int(os.getenv("DEDUP_COOLDOWN_MIN") or 240)
RISK_TICK_SEC       = int(os.getenv("RISK_TICK_SEC") or 15)   # як часто перевіряти відкриті позиції

# ====== GLOBAL STATE (per chat) ======
STATE: Dict[int, Dict[str, Any]] = {}

# Додатковий стан для трейлінгу/часткового виходу поміж чатами
TRAIL: Dict[str, Dict[str, float]] = {}   # key=symbol -> {"high":..., "took_partial":0/1, "entry":...}

# ====== MODE PROFILES ======
# Додано risk-параметри: max_spread_pct, min_book_ratio, partial_tp_pct, partial_qty_pct, trail_pct
MODE_PARAMS = {
    "aggressive": {
        "bars": ("15Min", "30Min", "1Hour"), "rsi_buy": 55.0, "rsi_sell": 45.0,
        "ema_fast": 15, "ema_slow": 30, "top_n": ALPACA_TOP_N,
        "tp_pct": 0.015, "sl_pct": 0.008,
        "max_spread_pct": 0.20/100, "min_book_ratio": 0.8,
        "partial_tp_pct": 0.012, "partial_qty_pct": 0.5,
        "trail_pct": 0.010
    },
    "scalp": {
        "bars": ("5Min", "15Min", "1Hour"), "rsi_buy": 58.0, "rsi_sell": 42.0,
        "ema_fast": 9, "ema_slow": 21, "top_n": ALPACA_TOP_N,
        "tp_pct": 0.010, "sl_pct": 0.006,
        "max_spread_pct": 0.12/100, "min_book_ratio": 1.0,
        "partial_tp_pct": 0.010, "partial_qty_pct": 0.5,
        "trail_pct": 0.007
    },
    "default": {
        "bars": ("15Min", "30Min", "1Hour"), "rsi_buy": 56.0, "rsi_sell": 44.0,
        "ema_fast": 12, "ema_slow": 26, "top_n": ALPACA_TOP_N,
        "tp_pct": 0.012, "sl_pct": 0.008,
        "max_spread_pct": 0.15/100, "min_book_ratio": 0.9,
        "partial_tp_pct": 0.011, "partial_qty_pct": 0.5,
        "trail_pct": 0.008
    },
    "swing": {
        "bars": ("30Min", "1Hour", "1Day"), "rsi_buy": 55.0, "rsi_sell": 45.0,
        "ema_fast": 20, "ema_slow": 40, "top_n": ALPACA_TOP_N,
        "tp_pct": 0.020, "sl_pct": 0.010,
        "max_spread_pct": 0.25/100, "min_book_ratio": 0.7,
        "partial_tp_pct": 0.015, "partial_qty_pct": 0.5,
        "trail_pct": 0.012
    },
    "safe": {
        "bars": ("15Min", "30Min", "1Hour"), "rsi_buy": 60.0, "rsi_sell": 40.0,
        "ema_fast": 15, "ema_slow": 35, "top_n": max(1, ALPACA_TOP_N-1),
        "tp_pct": 0.009, "sl_pct": 0.006,
        "max_spread_pct": 0.10/100, "min_book_ratio": 1.2,
        "partial_tp_pct": 0.009, "partial_qty_pct": 0.5,
        "trail_pct": 0.006
    },
}

# ====== CRYPTO WHITELIST (USD) ======
CRYPTO_USD_PAIRS = [
    "BTC/USD","ETH/USD","SOL/USD","LTC/USD","DOGE/USD","AVAX/USD","AAVE/USD","MKR/USD","DOT/USD",
    "LINK/USD","UNI/USD","PEPE/USD","XRP/USD","TRUMP/USD","CRV/USD","BCH/USD","BAT/USD","GRT/USD",
    "XTZ/USD","USDC/USD","USDT/USD","USDG/USD","YFI/USD","LDO/USD"
][:ALPACA_MAX_CRYPTO]

# ====== STOCKS UNIVERSE ======
STOCKS_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","ADBE","CRM","ORCL","AMD","AMAT","INTC","CSCO","QCOM",
    "BAC","JPM","GS","BRK.B","V","MA","KO","PEP","MCD","NKE","SPY","QQQ","IWM","DIA","XLF","XLK","XLV","XLE","XLY","XLP"
][:ALPACA_MAX_STOCKS]

# ============ HELPERS ============
def map_tf(tf: str) -> str:
    t = (tf or "").strip()
    return "1Hour" if t.lower() in ("60min","60","1h","60мин","60мін") else t

def to_order_sym(sym: str) -> str:
    return sym.replace("/", "").upper()

def to_data_sym(sym: str) -> str:
    s = (sym or "").replace(" ","").upper()
    if "/" in s: return s
    if s.endswith("USD"): return s[:-3]+"/USD"
    return s

def now_s() -> float:
    return time.time()

RECENT_TRADES: Dict[str, float] = {}
def skip_as_duplicate(market: str, sym: str, side: str) -> bool:
    key = f"{market}|{to_order_sym(sym)}|{side.lower()}"
    last = RECENT_TRADES.get(key, 0)
    if now_s() - last < DEDUP_COOLDOWN_MIN*60:
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
        ["/aggressive","/scalp","/default"],
        ["/swing","/safe","/help"],
        ["/signals_crypto","/trade_crypto"],
        ["/signals_stocks","/trade_stocks"],
        ["/alp_on","/alp_status","/alp_off"],
        ["/auto_on","/auto_status","/auto_off"],
        ["/long_mode","/short_mode","/both_mode"],
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

# ===== helper: account/clock/positions =====
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

# -------- DATA: bars / quotes / trades --------
async def get_bars_crypto(pairs: List[str], timeframe: str, limit: int = 120) -> Dict[str, Any]:
    tf = map_tf(timeframe)
    syms = ",".join([to_data_sym(p) for p in pairs])
    path = "/v1beta3/crypto/us/bars"
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
    path = "/v2/stocks/bars"
    params = {"symbols": syms, "timeframe": tf, "limit": str(limit), "sort": "asc"}
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        url = f"{ALPACA_DATA_URL}{path}"
        async with s.get(url, headers=_alp_headers(), params=params) as r:
            txt = await r.text()
            if r.status >= 400:
                raise RuntimeError(f"GET {url} {r.status}: {txt}")
            return json.loads(txt) if txt else {}

async def get_latest_quote_crypto(sym: str) -> Optional[Dict[str, Any]]:
    # NBBO quote (best bid/ask + sizes)
    dsym = to_data_sym(sym)
    path = "/v1beta3/crypto/us/quotes/latest"
    params = {"symbols": dsym}
    data = await alp_get_json(path, params)
    q = (data.get("quotes") or {}).get(dsym)
    return q[-1] if isinstance(q, list) and q else q

async def get_latest_trade_crypto(sym: str) -> Optional[float]:
    dsym = to_data_sym(sym)
    path = "/v1beta3/crypto/us/trades/latest"
    params = {"symbols": dsym}
    data = await alp_get_json(path, params)
    t = (data.get("trades") or {}).get(dsym)
    if isinstance(t, list) and t:
        t = t[-1]
    return float(t.get("p")) if t else None

# -------- INDICATORS --------
def ema(values: List[float], period: int) -> List[float]:
    if not values or period <= 0: return []
    k = 2.0 / (period + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out

def rsi(values: List[float], period: int) -> float:
    if len(values) < period + 1: return 50.0
    gains, losses = 0.0, 0.0
    for i in range(-period, 0):
        diff = values[i] - values[i - 1]
        if diff >= 0: gains += diff
        else: losses -= diff
    if losses == 0: return 70.0
    rs = gains / losses
    return 100.0 - (100.0 / (1 + rs))

def rank_score(c15: List[float], c30: List[float], c60: List[float],
               rsi_buy: float, rsi_sell: float, ema_fast_p: int, ema_slow_p: int) -> float:
    r1, r2, r3 = rsi(c15,14), rsi(c30,14), rsi(c60,14)
    e_fast, e_slow = ema(c60, ema_fast_p), ema(c60, ema_slow_p)
    trend = 0.0
    if e_fast and e_slow:
        trend = (e_fast[-1] - e_slow[-1]) / max(1e-9, abs(e_slow[-1]))
    bias_long = (1 if r1>=rsi_buy else 0)+(1 if r2>=rsi_buy else 0)+(1 if r3>=rsi_buy else 0)
    bias_short= (1 if r1<=rsi_sell else 0)+(1 if r2<=rsi_sell else 0)+(1 if r3<=rsi_sell else 0)
    bias = max(bias_long, bias_short)
    return bias*100.0 + trend*50.0 - abs(50.0 - r1)

def calc_sl_tp(side: str, price: float, conf: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    tp_pct = float(conf.get("tp_pct", 0.01))
    sl_pct = float(conf.get("sl_pct", 0.008))
    if side.lower() == "buy":
        return price*(1+tp_pct), price*(1-sl_pct)
    else:
        return price*(1-tp_pct), price*(1+sl_pct)

# -------- стакан-фільтр (NBBO) --------
async def orderbook_ok_for_buy(sym: str, conf: Dict[str, Any]) -> Tuple[bool, str]:
    q = await get_latest_quote_crypto(sym)
    if not q:
        return False, "no_quote"
    bid = float(q.get("bp") or 0)
    ask = float(q.get("ap") or 0)
    bs  = float(q.get("bs") or 0)
    asz = float(q.get("as") or 0)
    if bid<=0 or ask<=0 or ask<bid: 
        return False, "bad_quote"

    spread = (ask - bid) / ((ask + bid)/2.0)
    if spread > float(conf.get("max_spread_pct", 0.0015)):
        return False, f"spread>{conf.get('max_spread_pct')}"
    # для покупок хочемо, щоб підтримка покупців була не гірша за продавців
    if bs < float(conf.get("min_book_ratio",1.0)) * max(asz, 1e-9):
        return False, "weak_bid"
    return True, "ok"

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

# ======== ORDERS ========
def _round_stock_qty(qty: float) -> float:
    return round(qty, 3)

def _floor_qty(x: float, dec: int = 6) -> float:
    if x <= 0: return 0.0
    m = 10 ** dec
    return math.floor(x * m) / m

async def place_market_buy_notional(sym: str, notional: float) -> dict:
    safe_notional = max(1.0, float(notional) * 0.995)   # невеликий запас
    payload = {"symbol": to_order_sym(sym), "side": "buy", "type": "market",
               "time_in_force": "gtc", "notional": f"{safe_notional:.2f}"}
    return await alp_post_json("/v2/orders", payload)

async def place_market_buy_qty(sym: str, qty: float) -> dict:
    payload = {"symbol": to_order_sym(sym), "side": "buy", "type": "market",
               "time_in_force": "gtc", "qty": f"{_floor_qty(qty):.6f}"}
    return await alp_post_json("/v2/orders", payload)

async def place_market_sell_qty(sym: str, qty: float) -> dict:
    payload = {"symbol": to_order_sym(sym), "side": "sell", "type": "market",
               "time_in_force": "gtc", "qty": f"{_floor_qty(qty):.6f}"}
    return await alp_post_json("/v2/orders", payload)

async def get_order(order_id: str) -> dict:
    return await alp_get_json(f"/v2/orders/{order_id}")

async def place_bracket_notional_order_crypto(sym: str, side: str, notional: float) -> Any:
    if side.lower() != "buy":
        raise RuntimeError("crypto: лише long buy підтримано")

    # 1) спроба по нотіоналу із запасом
    try:
        buy = await place_market_buy_notional(sym, notional)
    except RuntimeError as e:
        msg = str(e)
        if "insufficient balance" in msg:
            m = re.search(r'"available"\s*:\s*"([\d\.]+)"', msg)
            if not m: raise
            available_qty = float(m.group(1))
            safe_qty = _floor_qty(available_qty - 1e-6, 6)
            if safe_qty <= 0: raise
            buy = await place_market_buy_qty(sym, safe_qty)
        else:
            raise

    # зачекати виконання та взяти filled_qty + avg_price
    order_id = buy.get("id", "")
    filled_qty, avg_price = 0.0, None
    for _ in range(10):
        od = await get_order(order_id)
        status = od.get("status")
        if status in ("filled","partially_filled"):
            filled_qty = float(od.get("filled_qty") or 0)
            avg_price  = float(od.get("filled_avg_price") or 0) or None
            if status == "filled": break
        await asyncio.sleep(0.6)

    # ініціалізуємо трейлінг-стан
    if avg_price:
        TRAIL[to_order_sym(sym)] = {"high": avg_price, "took_partial": 0.0, "entry": avg_price}
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
        "/signals_crypto — показати топ-N і (якщо Autotrade=ON) спробувати входи\n"
        "/trade_crypto — миттєво торгувати топ-N без звіту\n"
        "/signals_stocks — показати топ-N для акцій\n"
        "/trade_stocks — миттєво торгувати топ-N акцій\n"
        "/alp_on /alp_off /alp_status — автотрейд\n"
        "/auto_on /auto_off /auto_status — фоновий автоскан\n"
        "/long_mode /short_mode /both_mode — напрям (short для крипти ігнорується)\n"
        "/aggressive /scalp /default /swing /safe — профілі ризику",
        reply_markup=kb()
    )

async def set_mode(u: Update, c: ContextTypes.DEFAULT_TYPE, mode: str):
    st = stdef(u.effective_chat.id); st["mode"] = mode
    await u.message.reply_text(f"Режим встановлено: {mode.upper()}")

async def long_mode(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id); st["side_mode"] = "long"
    await u.message.reply_text("Режим входів: LONG")

async def short_mode(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id); st["side_mode"] = "short"
    await u.message.reply_text("Режим входів: SHORT (для крипти буде проігноровано)")

async def both_mode(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id); st["side_mode"] = "both"
    await u.message.reply_text("Режим входів: BOTH (для крипти застосуємо лише LONG)")

async def alp_on(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id); st["autotrade"] = True
    await u.message.reply_text("✅ Alpaca AUTOTRADE: ON")

async def alp_off(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id); st["autotrade"] = False
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
        conf = _mode_conf(st)

        for _, sym, arr in picks:
            side = "buy"
            if await has_open_long(sym):
                await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}")
                continue
            if skip_as_duplicate("CRYPTO", sym, side):
                await u.message.reply_text(f"⚪ SKIP (дубль): {sym} {side.upper()}")
                continue

            ok, reason = await orderbook_ok_for_buy(sym, conf)
            if not ok:
                await u.message.reply_text(f"⚪ SKIP {sym}: стакан-фільтр={reason}")
                continue

            try:
                entry = float(arr[-1]["c"])
                await place_bracket_notional_order_crypto(sym, side, ALPACA_NOTIONAL)
                await u.message.reply_text(
                    f"🟢 ORDER OK: {sym} BUY ${ALPACA_NOTIONAL:.2f}\n"
                    f"entry≈{entry:.6f}  (керування виходом у фоні)"
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
        conf = _mode_conf(st)

        for _, sym, arr in picks:
            side = "buy"
            if await has_open_long(sym):
                await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}")
                continue
            if skip_as_duplicate("CRYPTO", sym, side):
                await u.message.reply_text(f"⚪ SKIP (дубль): {sym} {side.upper()}")
                continue

            ok, reason = await orderbook_ok_for_buy(sym, conf)
            if not ok:
                await u.message.reply_text(f"⚪ SKIP {sym}: стакан-фільтр={reason}")
                continue

            try:
                entry = float(arr[-1]["c"])
                await place_bracket_notional_order_crypto(sym, side, ALPACA_NOTIONAL)
                await u.message.reply_text(
                    f"🟢 ORDER OK: {sym} BUY ${ALPACA_NOTIONAL:.2f}\n"
                    f"entry≈{entry:.6f}  (керування виходом у фоні)"
                )
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")
    except Exception as e:
        await u.message.reply_text(f"🔴 trade_crypto error: {e}")

# ------- STOCKS (без стакану й трейлу — як було) -------
async def signals_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        report, ranked = await scan_rank_stocks(st)
        await u.message.reply_text(report)
    except Exception as e:
        await u.message.reply_text(f"🔴 signals_stocks error: {e}")

async def trade_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("ℹ️ Автовходи для акцій залишив вимкненими (ринок не 24/7).")

# ======= AUTOSCAN (background) =======
async def _auto_scan_once_for_chat(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE):
    st = stdef(chat_id)
    if not st.get("auto_scan") or not st.get("autotrade"):
        return

    conf = _mode_conf(st)
    top_n = int(conf.get("top_n", max(1, ALPACA_TOP_N)))

    try:
        crypto_report, crypto_ranked = await scan_rank_crypto(st)
    except Exception:
        crypto_ranked = []

    combined: List[Tuple[float, str, List[Dict[str, Any]]]] = crypto_ranked
    combined.sort(reverse=True)
    picks = combined[:top_n]

    for score, sym, arr in picks:
        if await has_open_long(sym):
            continue
        side = "buy"

        ok, reason = await orderbook_ok_for_buy(sym, conf)
        if not ok:
            continue

        if skip_as_duplicate("CRYPTO", sym, side):
            continue

        try:
            entry = float(arr[-1]["c"])
            await place_bracket_notional_order_crypto(sym, side, ALPACA_NOTIONAL)
            await ctx.bot.send_message(
                chat_id,
                f"🟢 AUTO BUY: {to_order_sym(sym)} · ${ALPACA_NOTIONAL:.2f} · entry≈{entry:.6f}"
            )
        except Exception as e:
            await ctx.bot.send_message(chat_id, f"🔴 AUTO ORDER FAIL {sym}: {e}")

async def periodic_auto_scan(ctx: ContextTypes.DEFAULT_TYPE):
    for chat_id in list(STATE.keys()):
        try:
            await _auto_scan_once_for_chat(chat_id, ctx)
        except Exception as e:
            try: await ctx.bot.send_message(chat_id, f"🔴 periodic autoscan error: {e}")
            except Exception: pass

# ======= RISK MANAGER (трейлінг/часткові виходи/SL) =======
async def risk_manager_tick(ctx: ContextTypes.DEFAULT_TYPE):
    # читаємо всі позиції і застосовуємо правила лише для crypto
    try:
        positions = await alp_positions()
    except Exception:
        positions = []

    # Карта чатів, щоб надсилати нотифікацію (усім увімкненим)
    chat_ids = [cid for cid, st in STATE.items() if st.get("autotrade")]

    for p in positions or []:
        sym   = p.get("symbol","")
        cls   = p.get("asset_class","")
        if cls != "crypto": 
            continue
        qty   = float(p.get("qty") or 0)
        if qty <= 0: 
            continue

        entry = float(p.get("avg_entry_price") or 0)
        last  = await get_latest_trade_crypto(sym) or entry
        conf  = MODE_PARAMS.get(STATE[next(iter(STATE), None)].get("mode","scalp"), MODE_PARAMS["scalp"]) \
                if STATE else MODE_PARAMS["scalp"]

        # ініціалізувати стан
        tr = TRAIL.setdefault(sym, {"high": entry, "took_partial": 0.0, "entry": entry})
        tr["high"] = max(tr.get("high", entry), last)

        pnl_pct = (last/entry - 1.0)
        # Частковий тейк (один раз)
        if not tr.get("took_partial") and pnl_pct >= float(conf.get("partial_tp_pct",0.01)):
            sell_qty = qty * float(conf.get("partial_qty_pct",0.5))
            try:
                await place_market_sell_qty(sym, sell_qty)
                tr["took_partial"] = 1.0
                msg = f"✅ PARTIAL EXIT {sym}: +{pnl_pct*100:.2f}% · qty={sell_qty:.6f}"
                for cid in chat_ids: 
                    try: await ctx.bot.send_message(cid, msg)
                    except Exception: pass
            except Exception:
                pass

        # Трейлінг-стоп
        trail_pct = float(conf.get("trail_pct",0.008))
        trail_stop = tr["high"] * (1.0 - trail_pct)

        # Аварійний SL (жорсткий)
        hard_sl = entry * (1.0 - float(conf.get("sl_pct",0.008)))

        should_exit = False
        reason = ""
        if last <= hard_sl:
            should_exit, reason = True, "SL"
        elif last <= trail_stop and pnl_pct > 0:  # даємо трейл лише в плюс
            should_exit, reason = True, "TRAIL"

        if should_exit:
            try:
                await place_market_sell_qty(sym, qty)  # закрити залишок
                msg = f"☑️ EXIT {sym}: reason={reason} · avg={entry:.6f} → last={last:.6f} · PnL={pnl_pct*100:.2f}%"
                for cid in chat_ids:
                    try: await ctx.bot.send_message(cid, msg)
                    except Exception: pass
                # очистити стан
                TRAIL.pop(sym, None)
            except Exception:
                pass

# ------- AUTOSCAN commands -------
async def auto_on(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id); st["auto_scan"] = True
    await u.message.reply_text(f"✅ AUTO-SCAN: ON (кожні {SCAN_INTERVAL_SEC}s)")

async def auto_off(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id); st["auto_scan"] = False
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
async def scalp(u, c):      await set_mode(u, c, "scalp")
async def default(u, c):    await set_mode(u, c, "default")
async def swing(u, c):      await set_mode(u, c, "swing")
async def safe(u, c):       await set_mode(u, c, "safe")

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

    # фонові задачі
    app.job_queue.run_repeating(periodic_auto_scan, interval=SCAN_INTERVAL_SEC, first=10)
    app.job_queue.run_repeating(risk_manager_tick,  interval=RISK_TICK_SEC,  first=15)

    print("Bot started.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
