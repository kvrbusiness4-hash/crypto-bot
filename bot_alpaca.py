# bot_alpaca.py
# -*- coding: utf-8 -*-

import os
import json
import time
from typing import Dict, Any, Tuple, List

from aiohttp import ClientSession, ClientTimeout
from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ========= ENV =========
TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_TOKEN") or "").strip()

ALPACA_API_KEY = (os.getenv("ALPACA_API_KEY") or "").strip()
ALPACA_API_SECRET = (os.getenv("ALPACA_API_SECRET") or "").strip()

ALPACA_BASE_URL = (os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_URL = (os.getenv("ALPACA_DATA_URL") or "https://data.alpaca.markets").rstrip("/")

# торгуємо доларовим notional
ALPACA_NOTIONAL = float(os.getenv("ALPACA_NOTIONAL") or 25)

# скільки найкращих беремо для входу
ALPACA_TOP_N = int(os.getenv("ALPACA_TOP_N") or 3)

# верхні межі сканування, щоби не вбити ліміти
ALPACA_MAX_CRYPTO = int(os.getenv("ALPACA_MAX_CRYPTO") or 120)
ALPACA_MAX_STOCKS = int(os.getenv("ALPACA_MAX_STOCKS") or 400)

# антидубль
DEDUP_COOLDOWN_MIN = int(os.getenv("DEDUP_COOLDOWN_MIN") or 60)

# ====== GLOBAL STATE (per chat) ======
STATE: Dict[int, Dict[str, Any]] = {}
RECENT_TRADES: Dict[str, float] = {}  # "AAPL|buy" -> ts

# ====== MODE PROFILES ======
MODE_PARAMS = {
    "aggressive": {
        "bars": ("15Min", "30Min", "1Hour"),
        "rsi_buy": 55.0,
        "ema_fast": 12, "ema_slow": 26,
        "top_n": ALPACA_TOP_N,
        "tp_pct": 0.015, "sl_pct": 0.008,
        "min_liq_usd": 5_000_000,     # за 24h
        "max_spread_bps": 15,         # 0.15%
        "confirm_1h": False,
    },
    "scalp": {
        "bars": ("5Min", "15Min", "1Hour"),
        "rsi_buy": 58.0,
        "ema_fast": 9, "ema_slow": 21,
        "top_n": ALPACA_TOP_N,
        "tp_pct": 0.010, "sl_pct": 0.006,
        "min_liq_usd": 3_000_000,
        "max_spread_bps": 10,
        "confirm_1h": False,
    },
    "default": {
        "bars": ("15Min", "30Min", "1Hour"),
        "rsi_buy": 56.0,
        "ema_fast": 12, "ema_slow": 26,
        "top_n": ALPACA_TOP_N,
        "tp_pct": 0.012, "sl_pct": 0.008,
        "min_liq_usd": 8_000_000,
        "max_spread_bps": 12,
        "confirm_1h": True,
    },
    "swing": {
        "bars": ("30Min", "1Hour", "1Day"),
        "rsi_buy": 55.0,
        "ema_fast": 20, "ema_slow": 40,
        "top_n": ALPACA_TOP_N,
        "tp_pct": 0.020, "sl_pct": 0.010,
        "min_liq_usd": 10_000_000,
        "max_spread_bps": 20,
        "confirm_1h": True,
    },
    "safe": {
        "bars": ("15Min", "30Min", "1Hour"),
        "rsi_buy": 60.0,
        "ema_fast": 15, "ema_slow": 35,
        "top_n": max(1, ALPACA_TOP_N - 1),
        "tp_pct": 0.009, "sl_pct": 0.006,
        "min_liq_usd": 12_000_000,
        "max_spread_bps": 8,
        "confirm_1h": True,
    },
}

# ====== HELPERS ======
def now_s() -> float:
    return time.time()

def stdef(chat_id: int) -> Dict[str, Any]:
    st = STATE.setdefault(chat_id, {})
    st.setdefault("mode", "aggressive")
    st.setdefault("autotrade", False)
    return st

def kb() -> ReplyKeyboardMarkup:
    rows = [
        ["/aggressive", "/scalp", "/default"],
        ["/swing", "/safe", "/help"],
        ["/signals_crypto", "/trade_crypto"],
        ["/signals_stocks", "/trade_stocks"],
        ["/alp_on", "/alp_status", "/alp_off"],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def map_tf(tf: str) -> str:
    t = (tf or "").strip()
    return "1Hour" if t.lower() in ("60min", "60", "1h", "60мин", "60мін") else t

def _mode_conf(st: Dict[str, Any]) -> Dict[str, Any]:
    return MODE_PARAMS.get(st.get("mode") or "default", MODE_PARAMS["default"])

def is_crypto_sym(sym: str) -> bool:
    """Криптопара має вигляд 'BTC/USD' тощо."""
    return "/" in (sym or "")

def to_order_sym(sym: str) -> str:
    # Для ордерів у Alpaca: BTC/USD -> BTCUSD; акції залишаються як є
    if is_crypto_sym(sym):
        return sym.replace("/", "").upper()
    return sym.upper()

def to_data_crypto_sym(sym: str) -> str:
    s = sym.replace(" ", "").upper()
    if "/" in s:
        return s
    if s.endswith("USD"):
        return s[:-3] + "/USD"
    return s

def fmt_usd(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x:,.2f}"

def skip_as_duplicate(sym: str, side: str) -> bool:
    key = f"{to_order_sym(sym)}|{side.lower()}"
    last = RECENT_TRADES.get(key, 0)
    if now_s() - last < DEDUP_COOLDOWN_MIN * 60:
        return True
    RECENT_TRADES[key] = now_s()
    return False

# -------- math indicators --------
def ema(values: List[float], period: int) -> List[float]:
    if not values or period <= 0:
        return []
    k = 2.0 / (period + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out

def rsi_last(values: List[float], period: int = 14) -> float:
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

# -------- HTTP ----------
def _alp_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }

# Роутер: куди слати — trading API чи data API
DATA_PREFIXES = ("/v1beta", "/v2/stocks", "/v2/crypto")

async def alp_get_json(path: str, params: Dict[str, Any] | None = None) -> Any:
    # Все, що починається з /v1beta* або /v2/stocks|/v2/crypto — це DATA API
    use_data = path.startswith(DATA_PREFIXES)
    url = (ALPACA_DATA_URL if use_data else ALPACA_BASE_URL) + path
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        async with s.get(url, headers=_alp_headers(), params=params) as r:
            txt = await r.text()
            if r.status >= 400:
                raise RuntimeError(f"GET {url} {r.status}: {txt}")
            return json.loads(txt) if txt else {}

async def alp_post_json(path: str, payload: Dict[str, Any]) -> Any:
    # POST ми використовуємо лише для торгових операцій (trading API)
    url = f"{ALPACA_BASE_URL}{path}"
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        async with s.post(url, headers=_alp_headers(), data=json.dumps(payload)) as r:
            txt = await r.text()
            if r.status >= 400:
                raise RuntimeError(f"POST {url} {r.status}: {txt}")
            return json.loads(txt) if txt else {}

# -------- DATA: universe --------
async def list_assets(asset_class: str) -> List[str]:
    # asset_class: 'crypto' -> crypto; 'stocks' -> us_equity
    cls = "crypto" if asset_class == "crypto" else "us_equity"
    items = await alp_get_json(f"/v2/assets", params={"status": "active", "asset_class": cls})
    syms: List[str] = []
    for a in items or []:
        if not a.get("tradable", False):
            continue
        s = str(a.get("symbol") or "").upper().strip()
        if not s:
            continue
        if asset_class == "crypto":
            s = to_data_crypto_sym(s)  # зробити BTC/USD
        syms.append(s)
    return syms

# -------- DATA: bars --------
async def get_bars_crypto(pairs: List[str], timeframe: str, limit: int = 200) -> Dict[str, Any]:
    tf = map_tf(timeframe)
    batch = ",".join(pairs)
    path = f"/v1beta3/crypto/us/bars"
    params = {"symbols": batch, "timeframe": tf, "limit": str(limit), "sort": "asc"}
    return await alp_get_json(path, params=params)

async def get_bars_stocks(symbols: List[str], timeframe: str, limit: int = 200) -> Dict[str, Any]:
    tf = map_tf(timeframe)
    batch = ",".join(symbols)
    path = f"/v2/stocks/bars"
    params = {"symbols": batch, "timeframe": tf, "limit": str(limit), "sort": "asc", "adjustment": "split"}
    return await alp_get_json(path, params=params)

# -------- DATA: quotes (spread/liquidity proxy) --------
async def latest_quote_crypto(pairs: List[str]) -> Dict[str, Tuple[float, float]]:
    # returns { "BTC/USD": (bid, ask), ... }
    out: Dict[str, Tuple[float, float]] = {}
    path = f"/v1beta3/crypto/us/quotes/latest"
    # Alpaca дозволяє комою декілька symbols
    batch = ",".join(pairs)
    js = await alp_get_json(path, params={"symbols": batch})
    qmap = (js.get("quotes") or {})
    for sym in pairs:
        q = (qmap.get(sym) or {}).get("quote") or {}
        bp = float(q.get("bp", 0) or 0)
        ap = float(q.get("ap", 0) or 0)
        out[sym] = (bp, ap)
    return out

async def latest_quote_stock(symbols: List[str]) -> Dict[str, Tuple[float, float]]:
    # { "AAPL": (bp, ap) }
    out: Dict[str, Tuple[float, float]] = {}
    for sym in symbols:
        js = await alp_get_json("/v2/stocks/quotes/latest", params={"symbol": sym})
        q = js.get("quote") or {}
        bp = float(q.get("bp", 0) or 0)
        ap = float(q.get("ap", 0) or 0)
        out[sym] = (bp, ap)
    return out

def spread_bps(bid: float, ask: float) -> float:
    if ask <= 0 or bid <= 0 or ask < bid:
        return 99999.0
    return (ask - bid) / ask * 10000.0

# -------- scoring / confirmations --------
def basic_score(c15: List[float], c30: List[float], c60: List[float],
                rsi_buy: float, ema_fast: int, ema_slow: int) -> float:
    r1 = rsi_last(c15, 14)
    r2 = rsi_last(c30, 14)
    r3 = rsi_last(c60, 14)
    trend30 = 0.0
    e30f = ema(c30, ema_fast)
    e30s = ema(c30, ema_slow)
    if e30f and e30s:
        trend30 = 1.0 if e30f[-1] > e30s[-1] else -1.0
    trend60 = 0.0
    e60f = ema(c60, ema_fast)
    e60s = ema(c60, ema_slow)
    if e60f and e60s:
        trend60 = 1.0 if e60f[-1] > e60s[-1] else -1.0
    bias = sum(1 for r in (r1, r2, r3) if r >= rsi_buy)
    return bias*100 + 20.0*trend30 + 15.0*trend60 - abs(50.0 - r1)

def breakout_ok(c15: List[float]) -> bool:
    if len(c15) < 20:
        return False
    last = c15[-1]
    recent_high = max(c15[-20:-1])
    return last > recent_high  # простий пробій локального high

def liquidity_24h_value_usd(closes: List[float], vols: List[float]) -> float:
    # грубо: сума (close*volume) за ~96 барів 15m ~ 24 години
    if not closes or not vols:
        return 0.0
    n = min(len(closes), len(vols))
    take = min(n, 96)
    s = 0.0
    for i in range(-take, 0):
        try:
            s += float(closes[i]) * float(vols[i])
        except:
            pass
    return s

def accept_long(c15: List[float], c30: List[float], c60: List[float],
                conf: Dict[str, Any], sp_bps: float, liq_usd: float) -> bool:
    # спред та ліквідність
    if sp_bps > float(conf["max_spread_bps"]):
        return False
    if liq_usd < float(conf["min_liq_usd"]):
        return False
    # тренд і RSI
    sc = basic_score(c15, c30, c60, conf["rsi_buy"], conf["ema_fast"], conf["ema_slow"])
    if sc < 100:  # мінімум: 2 з 3 RSI вище порогу
        return False
    # підтвердження 30m (EMA fast>slow)
    e30f = ema(c30, conf["ema_fast"])
    e30s = ema(c30, conf["ema_slow"])
    if not (e30f and e30s and e30f[-1] > e30s[-1]):
        return False
    # для безпечних/дефолтних режимів — ще перевірка 1h trend up
    if conf.get("confirm_1h"):
        e60f = ema(c60, conf["ema_fast"])
        e60s = ema(c60, conf["ema_slow"])
        if not (e60f and e60s and e60f[-1] > e60s[-1]):
            return False
    # простий пробій локального high
    if not breakout_ok(c15):
        return False
    return True

def calc_sl_tp_from_pct(side: str, price: float, conf: Dict[str, Any]) -> Tuple[float | None, float | None]:
    tp_pct = float(conf.get("tp_pct", 0.012))
    sl_pct = float(conf.get("sl_pct", 0.008))
    if side == "buy":
        tp = price * (1 + tp_pct)
        sl = price * (1 - sl_pct)
    else:
        tp = sl = None
    return sl, tp

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

# -------- SCANS --------
async def scan_rank_crypto(st: Dict[str, Any]) -> Tuple[str, List[Tuple[float, str, Dict[str, Any]]]]:
    conf = _mode_conf(st)
    tf15, tf30, tf60 = [map_tf(x) for x in conf["bars"]]

    # 1) всесвіт крипти
    all_syms = await list_assets("crypto")
    if not all_syms:
        return "⚠️ Crypto universe empty.", []

    syms = all_syms[:ALPACA_MAX_CRYPTO]

    # 2) дані барів
    bars15 = await get_bars_crypto(syms, tf15, limit=150)
    bars30 = await get_bars_crypto(syms, tf30, limit=150)
    bars60 = await get_bars_crypto(syms, tf60, limit=150)

    # 3) котирування (спред)
    quotes = await latest_quote_crypto(syms)

    ranked: List[Tuple[float, str, Dict[str, Any]]] = []
    for sym in syms:
        raw15 = (bars15.get("bars") or {}).get(sym, [])
        raw30 = (bars30.get("bars") or {}).get(sym, [])
        raw60 = (bars60.get("bars") or {}).get(sym, [])
        if not (raw15 and raw30 and raw60):
            continue
        c15 = [float(x["c"]) for x in raw15]
        v15 = [float(x.get("v", 0)) for x in raw15]
        c30 = [float(x["c"]) for x in raw30]
        c60 = [float(x["c"]) for x in raw60]

        bid, ask = quotes.get(sym, (0.0, 0.0))
        sp = spread_bps(bid, ask)

        liq_usd = liquidity_24h_value_usd(c15, v15)

        if not accept_long(c15, c30, c60, conf, sp, liq_usd):
            continue

        # чим сильніший відрив EMA на 30м та RSI на 15м — тим вище
        e30f, e30s = ema(c30, conf["ema_fast"]), ema(c30, conf["ema_slow"])
        gap = 0.0
        if e30f and e30s and e30s[-1] != 0:
            gap = (e30f[-1] - e30s[-1]) / abs(e30s[-1])
        r = rsi_last(c15, 14)
        score = 1000*gap + (r - 50) - sp*0.2
        ranked.append((score, sym, {"c15": c15, "sp": sp}))

    ranked.sort(reverse=True)
    rep = (
        "🛰️ Сканер (крипта):\n"
        f"• Розглянуто: {len(syms)} пар\n"
        f"• Пройшло фільтри: {len(ranked)}\n"
        f"• Візьмемо до трейду: {min(conf['top_n'], len(ranked))}\n"
        + (("• Топ: " + ", ".join([s for _, s, _ in ranked[:min(20, len(ranked))]])) if ranked else "• Немає сигналів")
    )
    return rep, ranked

async def scan_rank_stocks(st: Dict[str, Any]) -> Tuple[str, List[Tuple[float, str, Dict[str, Any]]]]:
    conf = _mode_conf(st)
    tf15, tf30, tf60 = [map_tf(x) for x in conf["bars"]]

    # 1) всесвіт акцій
    all_syms = await list_assets("stocks")
    if not all_syms:
        return "⚠️ Stocks universe empty.", []
    syms = all_syms[:ALPACA_MAX_STOCKS]

    # 2) дані барів
    bars15 = await get_bars_stocks(syms, tf15, limit=150)
    bars30 = await get_bars_stocks(syms, tf30, limit=150)
    bars60 = await get_bars_stocks(syms, tf60, limit=150)

    # 3) котирування (спред)
    quotes = await latest_quote_stock(syms)

    ranked: List[Tuple[float, str, Dict[str, Any]]] = []
    bars_map15 = bars15.get("bars") or {}
    bars_map30 = bars30.get("bars") or {}
    bars_map60 = bars60.get("bars") or {}

    for sym in syms:
        raw15 = bars_map15.get(sym, [])
        raw30 = bars_map30.get(sym, [])
        raw60 = bars_map60.get(sym, [])
        if not (raw15 and raw30 and raw60):
            continue
        c15 = [float(x["c"]) for x in raw15]
        v15 = [float(x.get("v", 0)) for x in raw15]
        c30 = [float(x["c"]) for x in raw30]
        c60 = [float(x["c"]) for x in raw60]

        bid, ask = quotes.get(sym, (0.0, 0.0))
        sp = spread_bps(bid, ask)

        liq_usd = liquidity_24h_value_usd(c15, v15)

        if not accept_long(c15, c30, c60, conf, sp, liq_usd):
            continue

        e30f, e30s = ema(c30, conf["ema_fast"]), ema(c30, conf["ema_slow"])
        gap = 0.0
        if e30f and e30s and e30s[-1] != 0:
            gap = (e30f[-1] - e30s[-1]) / abs(e30s[-1])
        r = rsi_last(c15, 14)
        score = 1000*gap + (r - 50) - sp*0.2
        ranked.append((score, sym, {"c15": c15, "sp": sp}))

    ranked.sort(reverse=True)
    rep = (
        "🛰️ Сканер (акції):\n"
        f"• Розглянуто: {len(syms)} тікерів\n"
        f"• Пройшло фільтри: {len(ranked)}\n"
        f"• Візьмемо до трейду: {min(conf['top_n'], len(ranked))}\n"
        + (("• Топ: " + ", ".join([s for _, s, _ in ranked[:min(20, len(ranked))]])) if ranked else "• Немає сигналів")
    )
    return rep, ranked

# -------- EXECUTION WRAPPERS --------
async def execute_picks(u: Update, picks: List[Tuple[float, str, Dict[str, Any]]], what: str, st: Dict[str, Any]):
    if not picks:
        await u.message.reply_text(f"⚠️ {what}: немає відібраних сигналів")
        return
    conf = _mode_conf(st)
    take = picks[: conf["top_n"]]
    for _, sym, aux in take:
        side = "buy"
        price = float(aux["c15"][-1])
        sl, tp = calc_sl_tp_from_pct(side, price, conf)

        # антидубль
        if skip_as_duplicate(sym, side):
            await u.message.reply_text(f"⚪ SKIP (дубль): {sym}")
            continue

        try:
            await place_bracket_notional_order(sym, side, ALPACA_NOTIONAL, tp, sl)
            await u.message.reply_text(
                f"🟢 ORDER OK: {sym} BUY {fmt_usd(ALPACA_NOTIONAL)}\n"
                f"TP:{tp:.6f} SL:{sl:.6f}"
            )
        except Exception as e:
            await u.message.reply_text(f"🔴 ORDER FAIL {sym}: {e}")

# -------- COMMANDS --------
async def start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    stdef(u.effective_chat.id)
    await u.message.reply_text(
        "👋 Alpaca-бот готовий.\n"
        "Крипта торгується 24/7; акції — коли ринок відкритий, але сканер працює завжди.\n"
        "Увімкнути автотрейд: /alp_on  ·  Зупинити: /alp_off  ·  Стан: /alp_status",
        reply_markup=kb()
    )

async def help_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "/signals_crypto — показати найкращі крипто-сигнали (+ордери, якщо автотрейд ON)\n"
        "/trade_crypto   — одразу купити топ-N крипто\n"
        "/signals_stocks — показати найкращі акції (+ордери, якщо автотрейд ON)\n"
        "/trade_stocks   — одразу купити топ-N акцій\n"
        "/alp_on /alp_off /alp_status — автотрейд\n"
        "/aggressive /scalp /default /swing /safe — профілі режиму",
        reply_markup=kb()
    )

async def set_mode(u: Update, c: ContextTypes.DEFAULT_TYPE, mode: str):
    st = stdef(u.effective_chat.id)
    st["mode"] = mode
    await u.message.reply_text(f"Режим встановлено: {mode.upper()}")

async def aggressive(u, c): await set_mode(u, c, "aggressive")
async def scalp(u, c):      await set_mode(u, c, "scalp")
async def default(u, c):    await set_mode(u, c, "default")
async def swing(u, c):      await set_mode(u, c, "swing")
async def safe(u, c):       await set_mode(u, c, "safe")

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
            f"• cash={fmt_usd(float(acc.get('cash',0)))}\n"
            f"• buying_power={fmt_usd(float(acc.get('buying_power',0)))}\n"
            f"• equity={fmt_usd(float(acc.get('equity',0)))}\n"
            f"Mode={st.get('mode','default')} · Autotrade={'ON' if st.get('autotrade') else 'OFF'}"
        )
        await u.message.reply_text(txt)
    except Exception as e:
        await u.message.reply_text(f"🔴 alp_status error: {e}")

# --- Crypto ---
async def signals_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        report, ranked = await scan_rank_crypto(st)
        await u.message.reply_text(report)
        if st.get("autotrade"):
            await execute_picks(u, ranked, "CRYPTO", st)
    except Exception as e:
        await u.message.reply_text(f"🔴 signals_crypto error: {e}")

async def trade_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        _, ranked = await scan_rank_crypto(st)
        await execute_picks(u, ranked, "CRYPTO", st)
    except Exception as e:
        await u.message.reply_text(f"🔴 trade_crypto error: {e}")

# --- Stocks ---
async def signals_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        report, ranked = await scan_rank_stocks(st)
        await u.message.reply_text(report)
        if st.get("autotrade"):
            await execute_picks(u, ranked, "STOCKS", st)
    except Exception as e:
        await u.message.reply_text(f"🔴 signals_stocks error: {e}")

async def trade_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        _, ranked = await scan_rank_stocks(st)
        await execute_picks(u, ranked, "STOCKS", st)
    except Exception as e:
        await u.message.reply_text(f"🔴 trade_stocks error: {e}")

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

    app.add_handler(CommandHandler("alp_on", alp_on))
    app.add_handler(CommandHandler("alp_off", alp_off))
    app.add_handler(CommandHandler("alp_status", alp_status))

    app.add_handler(CommandHandler("signals_crypto", signals_crypto))
    app.add_handler(CommandHandler("trade_crypto", trade_crypto))

    app.add_handler(CommandHandler("signals_stocks", signals_stocks))
    app.add_handler(CommandHandler("trade_stocks", trade_stocks))

    print("Bot started.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
