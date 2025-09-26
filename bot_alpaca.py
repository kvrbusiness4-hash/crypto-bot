# -*- coding: utf-8 -*-

import os
import json
import math
import asyncio
import time
import re
from typing import Dict, Any, Tuple, List, Optional

from aiohttp import ClientSession, ClientTimeout
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes

# ========= ENV =========

TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_TOKEN") or "").strip()

ALPACA_API_KEY = (os.getenv("ALPACA_API_KEY") or "").strip()
ALPACA_API_SECRET = (os.getenv("ALPACA_API_SECRET") or "").strip()

ALPACA_BASE_URL = (os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_URL = (os.getenv("ALPACA_DATA_URL") or "https://data.alpaca.markets").rstrip("/")

ALPACA_NOTIONAL = float(os.getenv("ALPACA_NOTIONAL") or 50)   # USD на вхід (конвертуємо у qty)
ALPACA_TOP_N = int(os.getenv("ALPACA_TOP_N") or 2)
ALPACA_MAX_CRYPTO = int(os.getenv("ALPACA_MAX_CRYPTO") or 25)
ALPACA_MAX_STOCKS = int(os.getenv("ALPACA_MAX_STOCKS") or 50)

SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC") or 300)   # автоскан + менеджер
DEDUP_COOLDOWN_MIN = int(os.getenv("DEDUP_COOLDOWN_MIN") or 240)

# ====== GLOBAL STATE (per chat) ======
STATE: Dict[int, Dict[str, Any]] = {}
RECENT_TRADES: Dict[str, float] = {}

# ====== MODE PROFILES ======
MODE_PARAMS = {
    "aggressive": {"bars": ("15Min", "30Min", "1Hour"), "rsi_buy": 55.0, "rsi_sell": 45.0,
                   "ema_fast": 15, "ema_slow": 30, "top_n": ALPACA_TOP_N, "tp_pct": 0.015, "sl_pct": 0.010},
    "scalp": {"bars": ("5Min", "15Min", "1Hour"), "rsi_buy": 58.0, "rsi_sell": 42.0,
              "ema_fast": 9, "ema_slow": 21, "top_n": ALPACA_TOP_N, "tp_pct": 0.010, "sl_pct": 0.006},
    "default": {"bars": ("15Min", "30Min", "1Hour"), "rsi_buy": 56.0, "rsi_sell": 44.0,
                "ema_fast": 12, "ema_slow": 26, "top_n": ALPACA_TOP_N, "tp_pct": 0.012, "sl_pct": 0.008},
    "swing": {"bars": ("30Min", "1Hour", "1Day"), "rsi_buy": 55.0, "rsi_sell": 45.0,
              "ema_fast": 20, "ema_slow": 40, "top_n": ALPACA_TOP_N, "tp_pct": 0.020, "sl_pct": 0.012},
    "safe": {"bars": ("15Min", "30Min", "1Hour"), "rsi_buy": 60.0, "rsi_sell": 40.0,
             "ema_fast": 15, "ema_slow": 35, "top_n": max(1, ALPACA_TOP_N - 1), "tp_pct": 0.009, "sl_pct": 0.006},
}

# ====== CRYPTO WHITELIST (USD) ======
CRYPTO_USD_PAIRS = [
    "BTC/USD","ETH/USD","SOL/USD","LTC/USD","DOGE/USD","AVAX/USD","AAVE/USD","MKR/USD","DOT/USD",
    "LINK/USD","UNI/USD","PEPE/USD","XRP/USD","CRV/USD","BCH/USD","BAT/USD","GRT/USD",
    "XTZ/USD","USDC/USD","USDT/USD","USDG/USD","YFI/USD","LDO/USD"
][:ALPACA_MAX_CRYPTO]

# ====== STOCKS (довідково) ======
STOCKS_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","ADBE","CRM","ORCL","AMD","AMAT","INTC","CSCO","QCOM",
    "BAC","JPM","GS","BRK.B","V","MA","KO","PEP","MCD","NKE",
    "SPY","QQQ","IWM","DIA","XLF","XLK","XLV","XLE","XLY","XLP",
][:ALPACA_MAX_STOCKS]

# ============ HELPERS ============
def map_tf(tf: str) -> str:
    tf = (tf or "").strip()
    tl = tf.lower()
    if tl in ("60min","60","1h","60мин","60мін"):
        return "1Hour"
    m = {"1min":"1Min","5min":"5Min","15min":"15Min","30min":"30Min","1hour":"1Hour","1day":"1Day"}
    return m.get(tl, tf)

def to_order_sym(sym: str) -> str:
    return sym.replace("/", "").upper()

def to_data_sym(sym: str) -> str:
    s = (sym or "").replace(" ","").upper()
    if "/" in s: return s
    if s.endswith("USD"): return s[:-3]+"/USD"
    return s

def now_s() -> float:
    return time.time()

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

async def get_position(sym: str) -> Optional[Dict[str, Any]]:
    try:
        p = await alp_get_json(f"/v2/positions/{to_order_sym(sym)}")
        if p and (float(p.get("qty", 0) or 0) != 0):
            return p
        return None
    except Exception:
        return None

# -------- DATA: bars & orderbook ----------
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

async def get_orderbook_crypto(sym: str, depth: int = 10) -> Optional[Dict[str, Any]]:
    """
    Повертає {"bid_vol": float, "ask_vol": float, "mid": float} з ордербуку.
    Якщо ендпойнт недоступний — повертає None (логіка продовжить працювати без стакану).
    """
    try:
        data_sym = to_data_sym(sym)
        path = "/v1beta3/crypto/us/books"
        params = {"symbols": data_sym, "limit": str(depth)}
        async with ClientSession(timeout=ClientTimeout(total=10)) as s:
            url = f"{ALPACA_DATA_URL}{path}"
            async with s.get(url, headers=_alp_headers(), params=params) as r:
                if r.status >= 400:
                    # тихо ігноруємо (у деяких планах books може бути недоступний)
                    return None
                js = json.loads(await r.text()) if (await r.text()) else {}
        books = (js.get("books") or {}).get(data_sym, [])
        if not books:
            return None
        b = books[-1]  # останній снапшот
        bids = b.get("bids") or []
        asks = b.get("asks") or []
        bid_vol = sum(float(x.get("s", 0) or 0) for x in bids[:depth])
        ask_vol = sum(float(x.get("s", 0) or 0) for x in asks[:depth])
        best_bid = float(bids[0].get("p")) if bids else 0.0
        best_ask = float(asks[0].get("p")) if asks else 0.0
        mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else 0.0
        return {"bid_vol": bid_vol, "ask_vol": ask_vol, "mid": mid}
    except Exception:
        return None

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

# ======== ORDER UTILS ========
def _floor_qty(x: float, dec: int = 6) -> float:
    if x <= 0: return 0.0
    m = 10 ** dec
    return math.floor(x * m) / m

async def get_last_price_crypto(sym: str) -> float:
    data_sym = to_data_sym(sym)
    bars = await get_bars_crypto([data_sym], "5Min", limit=2)
    arr = (bars.get("bars") or {}).get(data_sym, [])
    if not arr:
        raise RuntimeError(f"no bars for {data_sym}")
    return float(arr[-1]["c"])

async def place_market_buy_crypto_qty(sym: str, qty: float) -> dict:
    payload = {
        "symbol": to_order_sym(sym),
        "side": "buy",
        "type": "market",
        "time_in_force": "gtc",
        "asset_class": "crypto",
        "qty": f"{_floor_qty(qty):.6f}",
    }
    return await alp_post_json("/v2/orders", payload)

async def place_market_sell_crypto_qty(sym: str, qty: float) -> dict:
    payload = {
        "symbol": to_order_sym(sym),
        "side": "sell",
        "type": "market",
        "time_in_force": "gtc",
        "asset_class": "crypto",
        "qty": f"{_floor_qty(qty):.6f}",
    }
    return await alp_post_json("/v2/orders", payload)

async def get_order(order_id: str) -> dict:
    return await alp_get_json(f"/v2/orders/{order_id}")

# ======== ENTRY & EXIT LOGIC ========

async def crypto_buy_by_usd(sym: str, usd_notional: float) -> Tuple[dict, float, float]:
    """
    Купівля крипти через qty = USD / price * 0.995 (запас).
    Повертає (raw_order, filled_qty, fill_price).
    Якщо 403 "insufficient balance" — автоматично зменшує qty і ретраїть.
    """
    px = await get_last_price_crypto(sym)
    raw_qty = (float(usd_notional) / max(1e-9, px)) * 0.995
    qty = _floor_qty(raw_qty, 6)
    if qty <= 0:
        raise RuntimeError("qty<=0 (too small notional)")

    async def _try(q: float) -> dict:
        return await place_market_buy_crypto_qty(sym, q)

    try:
        order = await _try(qty)
    except RuntimeError as e:
        s = str(e)
        if "insufficient balance" in s:
            # 1) спробуємо використати available з повідомлення (буває у crypto як qty)
            m = re.search(r'"available"\s*:\s*"([\d\.]+)"', s)
            if m:
                av = _floor_qty(float(m.group(1)) * 0.995, 6)
                if av > 0:
                    order = await _try(av)
                else:
                    raise
            else:
                # 2) або просто зменшимо qty на 2% і повторимо
                order = await _try(_floor_qty(qty * 0.98, 6))
        else:
            raise

    # чекаємо заповнення, щоб дістати fill average price
    order_id = order.get("id", "")
    filled_qty, fill_price = 0.0, px
    for _ in range(12):
        od = await get_order(order_id)
        status = od.get("status")
        if status in ("filled", "partially_filled"):
            filled_qty = float(od.get("filled_qty") or 0)
            try:
                fill_price = float(od.get("filled_avg_price") or px)
            except Exception:
                fill_price = px
            if status == "filled":
                break
        await asyncio.sleep(0.5)

    return order, filled_qty, fill_price

def should_exit_by_indicators(conf: Dict[str, Any], closes_short: List[float], closes_long: List[float]) -> bool:
    e_fast = ema(closes_long, conf["ema_fast"])
    e_slow = ema(closes_long, conf["ema_slow"])
    r = rsi(closes_short, 14)
    cross_down = (e_fast and e_slow and e_fast[-1] < e_slow[-1])
    weak_rsi = r < 50.0
    return bool(cross_down or weak_rsi)

def should_exit_by_orderbook(entry_price: float, last_price: float, ob: Optional[Dict[str, Any]]) -> bool:
    """
    Просте правило по стакану:
    - якщо bid/(bid+ask) < 0.45 (перевага продавців)
    - і ціна просідає відносно входу (last < entry або last < mid)
    → виходимо.
    Якщо ордербук недоступний — повертаємо False (фільтр не блокує інші умови).
    """
    if not ob:
        return False
    bid = float(ob.get("bid_vol") or 0.0)
    ask = float(ob.get("ask_vol") or 0.0)
    tot = bid + ask
    if tot <= 0:
        return False
    imbalance = bid / tot
    mid = float(ob.get("mid") or last_price)
    bad_flow = imbalance < 0.45
    price_worse = (last_price < entry_price) or (last_price < mid)
    return bool(bad_flow and price_worse)

async def try_manage_crypto_positions(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int):
    """Фоновий менеджер: TP/SL + EMA/RSI + стакан."""
    st = stdef(chat_id)
    conf = _mode_conf(st)
    tp_pct = float(conf.get("tp_pct", 0.01))
    sl_pct = float(conf.get("sl_pct", 0.008))

    positions = await alp_positions()
    crypto_positions = [p for p in positions if p.get("asset_class") == "crypto"]
    syms = [to_data_sym(p["symbol"]) for p in crypto_positions]
    if not syms:
        return

    bars_s = await get_bars_crypto(syms, map_tf(conf["bars"][0]), limit=120)
    bars_l = await get_bars_crypto(syms, map_tf(conf["bars"][1]), limit=120)

    for p in crypto_positions:
        sym_ord = p["symbol"]            # BTCUSD
        sym_data = to_data_sym(sym_ord)  # BTC/USD

        try:
            qty = float(p.get("qty") or 0)
            if qty <= 0:
                continue
            avg_entry = float(p.get("avg_entry_price") or 0)

            c_short = [float(x["c"]) for x in (bars_s.get("bars") or {}).get(sym_data, [])]
            c_long  = [float(x["c"]) for x in (bars_l.get("bars") or {}).get(sym_data, [])]
            if not c_short:
                continue
            last = c_short[-1]

            # ордербук (може бути None — тоді пропускаємо цю умову)
            ob = await get_orderbook_crypto(sym_ord, depth=10)

            take_profit = last >= avg_entry * (1.0 + tp_pct)
            stop_loss   = last <= avg_entry * (1.0 - sl_pct)
            by_filters  = should_exit_by_indicators(conf, c_short, c_long) if c_long else False
            by_book     = should_exit_by_orderbook(avg_entry, last, ob)

            if take_profit or stop_loss or by_filters or by_book:
                await place_market_sell_crypto_qty(sym_ord, qty)
                reason = "TP" if take_profit else ("SL" if stop_loss else ("SIGNAL" if by_filters else "ORDERBOOK"))
                pnl_pct = (last / avg_entry - 1.0) * 100.0 if avg_entry > 0 else 0.0
                extra = ""
                if ob:
                    bid = ob.get("bid_vol", 0.0); ask = ob.get("ask_vol", 0.0)
                    tot = bid + ask if (bid or ask) else 1.0
                    extra = f" · OB={bid/(tot):.2f}bid"
                await ctx.bot.send_message(
                    chat_id,
                    f"✅ EXIT {sym_ord}: reason={reason} · avg={avg_entry:.6f} → last={last:.6f} "
                    f"· PnL={pnl_pct:.2f}% · qty={qty:.6f}{extra}"
                )
        except Exception as e:
            try:
                await ctx.bot.send_message(chat_id, f"🔴 manage position error {sym_ord}: {e}")
            except Exception:
                pass

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

# -------- SCAN (STOCKS) — інформативно --------
async def scan_rank_stocks(st: Dict[str, Any]) -> Tuple[str, List[Tuple[float, str, List[Dict[str, Any]]]]]:
    conf = _mode_conf(st)
    tf15, tf30, tf60 = map_tf(conf["bars"][0]), map_tf(conf["bars"][1]), map_tf(conf["bars"][2])

    symbols = STOCKS_UNIVERSE[:]
    # лишаємо як є — тільки для звіту
    bars15 = await alp_get_json("/v2/stocks/bars", {"symbols": ",".join(symbols), "timeframe": tf15, "limit": "120", "sort": "asc"})
    bars30 = await alp_get_json("/v2/stocks/bars", {"symbols": ",".join(symbols), "timeframe": tf30, "limit": "120", "sort": "asc"})
    bars60 = await alp_get_json("/v2/stocks/bars", {"symbols": ",".join(symbols), "timeframe": tf60, "limit": "120", "sort": "asc"})

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
        "/signals_crypto — показати топ-N і (якщо Autotrade=ON) купити\n"
        "/trade_crypto — миттєво торгувати топ-N без звіту\n"
        "/signals_stocks — показати топ-N для акцій (інфо)\n"
        "/trade_stocks — інфо\n"
        "/alp_on /alp_off /alp_status — автотрейд\n"
        "/auto_on /auto_off /auto_status — фоновий автоскан + менеджер позицій\n"
        "/long_mode /short_mode /both_mode — напрям (short для крипти ігнорується)\n"
        "/aggressive /scalp /default /swing /safe — профілі ризику",
        reply_markup=kb()
    )

async def set_mode(u: Update, c: ContextTypes.DEFAULT_TYPE, mode: str):
    st = stdef(u.effective_chat.id)
    st["mode"] = mode
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
        tp_pct, sl_pct = conf["tp_pct"], conf["sl_pct"]

        for _, sym, _ in picks:
            if await get_position(sym):
                await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}")
                continue
            if skip_as_duplicate("CRYPTO", sym, "buy"):
                await u.message.reply_text(f"⚪ SKIP (дубль): {sym} BUY")
                continue

            try:
                _, filled_qty, fill_price = await crypto_buy_by_usd(sym, ALPACA_NOTIONAL)
                await u.message.reply_text(
                    f"🟢 BUY OK: {sym} · ${ALPACA_NOTIONAL:.2f}\n"
                    f"Entry={fill_price:.6f} · qty={filled_qty:.6f}\n"
                    f"Менеджер: TP={tp_pct*100:.2f}% · SL={sl_pct*100:.2f}% + EMA/RSI + OrderBook"
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
        tp_pct, sl_pct = conf["tp_pct"], conf["sl_pct"]

        for _, sym, _ in picks:
            if await get_position(sym):
                await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}")
                continue
            if skip_as_duplicate("CRYPTO", sym, "buy"):
                await u.message.reply_text(f"⚪ SKIP (дубль): {sym} BUY")
                continue

            try:
                _, filled_qty, fill_price = await crypto_buy_by_usd(sym, ALPACA_NOTIONAL)
                await u.message.reply_text(
                    f"🟢 BUY OK: {sym} · ${ALPACA_NOTIONAL:.2f}\n"
                    f"Entry={fill_price:.6f} · qty={filled_qty:.6f}\n"
                    f"Менеджер: TP={tp_pct*100:.2f}% · SL={sl_pct*100:.2f}% + EMA/RSI + OrderBook"
                )
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")
    except Exception as e:
        await u.message.reply_text(f"🔴 trade_crypto error: {e}")

# ------- STOCKS commands (інфо) -------
async def signals_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        report, _ = await scan_rank_stocks(st)
        await u.message.reply_text(report + "\n(Торгівля акціями у цій збірці вимкнена)")
    except Exception as e:
        await u.message.reply_text(f"🔴 signals_stocks error: {e}")

async def trade_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("ℹ️ У цій збірці автотрейд для акцій відключений. Фокус — крипта.")

# ======= AUTOSCAN (background) =======
async def _auto_scan_once_for_chat(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE):
    st = stdef(chat_id)

    # 1) менеджер позицій (виходи)
    try:
        await try_manage_crypto_positions(ctx, chat_id)
    except Exception as e:
        try:
            await ctx.bot.send_message(chat_id, f"🔴 manager error: {e}")
        except Exception:
            pass

    # 2) нові входи
    if not (st.get("auto_scan") and st.get("autotrade")):
        return

    try:
        report, ranked = await scan_rank_crypto(st)
        # показувати звіт кожного циклу не будемо, щоб не спамити
        _ = report
    except Exception as e:
        ranked = []
        await ctx.bot.send_message(chat_id, f"🔴 Крипто-скан помилка: {e}")

    conf = _mode_conf(st)
    picks = ranked[: int(conf.get("top_n", max(1, ALPACA_TOP_N)))]

    for _, sym, _ in picks:
        if await get_position(sym):
            continue
        if skip_as_duplicate("CRYPTO", sym, "buy"):
            continue
        try:
            _, qty, entry = await crypto_buy_by_usd(sym, ALPACA_NOTIONAL)
            await ctx.bot.send_message(
                chat_id,
                f"🟢 AUTO BUY: {to_order_sym(sym)} · ${ALPACA_NOTIONAL:.2f} · entry={entry:.6f} · qty={qty:.6f}"
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
    # Акції (інфо)
    app.add_handler(CommandHandler("signals_stocks", signals_stocks))
    app.add_handler(CommandHandler("trade_stocks", trade_stocks))

    # Автоскан + менеджер позицій
    app.add_handler(CommandHandler("auto_on", auto_on))
    app.add_handler(CommandHandler("auto_off", auto_off))
    app.add_handler(CommandHandler("auto_status", auto_status))

    app.job_queue.run_repeating(periodic_auto_scan, interval=SCAN_INTERVAL_SEC, first=10)

    print("Bot started.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
