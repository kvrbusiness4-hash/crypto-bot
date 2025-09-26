# -*- coding: utf-8 -*-
"""
Alpaca TG-бот (крипто-скальпінг + менеджер позицій).
Правки:
- Нема «шаблонних» пар (AAVE/AVAX) — список формуємо без фаворитів.
- SCAN_INTERVAL_SEC береться з ENV (SCAN_INTERVAL_SEC), значення за замовчуванням 300.
- Дедуп працює тільки на реально відправлені спроби, не блокує перший вхід.
- Фоновий менеджер закриває за TP/SL/сигналом. Дрібні позиції не намагається продати (qty→0).
"""

import os
import json
import math
import asyncio
import time
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

USD_PER_TRADE = float(os.getenv("ALPACA_NOTIONAL") or os.getenv("USD_PER_TRADE") or 50.0)

ALPACA_TOP_N = int(os.getenv("ALPACA_TOP_N") or 2)
ALPACA_MAX_CRYPTO = int(os.getenv("ALPACA_MAX_CRYPTO") or 25)
ALPACA_MAX_STOCKS = int(os.getenv("ALPACA_MAX_STOCKS") or 50)

# інтервал фонового автоскану в секундах (і менеджера позицій також) — з .env
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC") or 300)   # ← головна правка

# анти-дубль: скільки хвилин не дублювати одну й ту саму сторону по тому ж символу
DEDUP_COOLDOWN_MIN = int(os.getenv("DEDUP_COOLDOWN_MIN") or 240)

# ====== GLOBAL STATE (per chat) ======
STATE: Dict[int, Dict[str, Any]] = {}

# ====== MODE PROFILES ======
MODE_PARAMS = {
    # SCALP: швидкі таймфрейми, ціль 3%, стоп 2%
    "scalp": {
        "bars": ("1Min", "5Min", "15Min"),
        "rsi_buy": 58.0, "rsi_sell": 42.0,
        "ema_fast": 9, "ema_slow": 21,
        "top_n": ALPACA_TOP_N,
        "tp_pct": 0.03,     # 3%
        "sl_pct": 0.02      # 2%
    },

    # AGGRESSIVE: трохи довше тримаємо, ціль 5%, стоп 3%
    "aggressive": {
        "bars": ("5Min", "15Min", "1Hour"),
        "rsi_buy": 56.0, "rsi_sell": 44.0,
        "ema_fast": 12, "ema_slow": 26,
        "top_n": ALPACA_TOP_N,
        "tp_pct": 0.05,     # 5%
        "sl_pct": 0.03      # 3%
    },

    # DEFAULT: щось середнє між scalp/aggressive
    "default": {
        "bars": ("5Min", "15Min", "1Hour"),
        "rsi_buy": 57.0, "rsi_sell": 43.0,
        "ema_fast": 12, "ema_slow": 26,
        "top_n": ALPACA_TOP_N,
        "tp_pct": 0.04,     # 4%
        "sl_pct": 0.025     # 2.5%
    },

    # SWING: тримаємо довше, ціль 10%, стоп 3%
    "swing": {
        "bars": ("30Min", "1Hour", "4Hour"),
        "rsi_buy": 55.0, "rsi_sell": 45.0,
        "ema_fast": 20, "ema_slow": 40,
        "top_n": ALPACA_TOP_N,
        "tp_pct": 0.10,     # 10%
        "sl_pct": 0.03      # 3%
    },

    # SAFE: консервативно, ціль 3%, стоп 2% + жорсткіший RSI
    "safe": {
        "bars": ("5Min", "15Min", "1Hour"),
        "rsi_buy": 60.0, "rsi_sell": 40.0,
        "ema_fast": 12, "ema_slow": 26,
        "top_n": max(1, ALPACA_TOP_N - 1),
        "tp_pct": 0.03,     # 3%
        "sl_pct": 0.02      # 2%
    },
}

# ====== CRYPTO WHITELIST (USD) ======
# Без фаворитів — звичайний список ліквідних пар:
CRYPTO_USD_PAIRS = [
    "BTC/USD","ETH/USD","SOL/USD","LTC/USD","DOGE/USD","AVAX/USD","AAVE/USD","MKR/USD","DOT/USD",
    "LINK/USD","UNI/USD","PEPE/USD","XRP/USD","CRV/USD","BCH/USD","BAT/USD","GRT/USD",
    "XTZ/USD","USDC/USD","USDT/USD","YFI/USD","LDO/USD"
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

# Анти-дубль: ключі тільки якщо була реальна спроба трейду
RECENT_TRADES: Dict[str, float] = {}
def mark_trade(market: str, sym: str, side: str) -> None:
    key = f"{market}|{to_order_sym(sym)}|{side.lower()}"
    RECENT_TRADES[key] = now_s()

def too_recent(market: str, sym: str, side: str) -> bool:
    key = f"{market}|{to_order_sym(sym)}|{side.lower()}"
    last = RECENT_TRADES.get(key, 0)
    return now_s() - last < DEDUP_COOLDOWN_MIN * 60

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
# ===== Partial TP + Trailing state =====
# зберігаємо по кожному символу, чи взяли TP1 і який був максимум ціни після нього
POS_STATE: Dict[str, Dict[str, Any]] = {}   # { "BTCUSD": {"took_tp1": bool, "highest": float} }
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

# -------- DATA: /bars ----------
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

# -------- INDИКАТОРИ --------
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

# ======== ORDER UTILS ========

def _floor_qty(x: float, dec: int = 6) -> float:
    if x <= 0: return 0.0
    m = 10 ** dec
    return math.floor(x * m) / m

async def get_last_price_crypto(sym: str) -> float:
    """Остання ціна з 5Min барів (остання свічка)."""
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
    Купівля крипти через qty = USD / price * 0.995.
    Повертає (raw_order, filled_qty, fill_price).
    """
    px = await get_last_price_crypto(sym)
    raw_qty = (float(usd_notional) / max(1e-9, px)) * 0.995
    qty = _floor_qty(raw_qty, 6)
    if qty <= 0:
        raise RuntimeError("qty<=0 (too small notional)")
    order = await place_market_buy_crypto_qty(sym, qty)

    # дочекаємось заповнення
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

async def try_manage_crypto_positions(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int):
    """
    Фоновий менеджер:
    - SL завжди активний (sl_pct із MODE_PARAMS)
    - TP1 = tp_pct: продаємо 50% позиції
    - Після TP1 вмикаємо трейлінг на решту: фіксуємо максимум і закриваємо при падінні на trail_pct
    """
    st = stdef(chat_id)
    conf = _mode_conf(st)

    tp_pct = float(conf.get("tp_pct", 0.03))   # 3% дефолт
    sl_pct = float(conf.get("sl_pct", 0.02))   # 2% дефолт
    trail_pct = max(0.015, tp_pct / 2.0)       # напр., 1.5% або половина TP

    positions = await alp_positions()
    crypto_positions = [p for p in positions if (p.get("asset_class") == "crypto")]

    if not crypto_positions:
        return

    # Збираємо символи для одного запиту барів
    syms = [to_data_sym(p["symbol"]) for p in crypto_positions]
    bars_short = await get_bars_crypto(syms, map_tf(conf["bars"][0]), limit=2)

    for p in crypto_positions:
        sym_ord  = p["symbol"]            # BTCUSD
        sym_data = to_data_sym(sym_ord)   # BTC/USD

        try:
            qty = float(p.get("qty") or 0.0)
            if qty <= 0:
                continue

            avg_entry = float(p.get("avg_entry_price") or 0.0)
            arr = (bars_short.get("bars") or {}).get(sym_data, [])
            if not arr:
                continue
            last = float(arr[-1]["c"])

            # --- STOP LOSS завжди активний ---
            if avg_entry > 0 and (last <= avg_entry * (1.0 - sl_pct)):
                q = _floor_qty(qty, 6)
                if q > 0:
                    await place_market_sell_crypto_qty(sym_ord, q)
                    pnl_pct = (last / avg_entry - 1.0) * 100.0
                    await ctx.bot.send_message(
                        chat_id,
                        f"✅ EXIT {sym_ord}: reason=SL · avg={avg_entry:.6f} → last={last:.6f} · PnL={pnl_pct:.2f}% · qty={q:.6f}"
                    )
                continue  # позицію закрили, йдемо далі

            # Дістаємо/створюємо локальний стейт
            st_pos = POS_STATE.setdefault(sym_ord, {"took_tp1": False, "highest": avg_entry})

            # --- TP1: продаємо 50%, коли досягли tp_pct ---
            if not st_pos["took_tp1"] and avg_entry > 0 and (last >= avg_entry * (1.0 + tp_pct)):
                sell_qty = _floor_qty(qty * 0.5, 6)
                if sell_qty > 0:
                    await place_market_sell_crypto_qty(sym_ord, sell_qty)
                    st_pos["took_tp1"] = True
                    st_pos["highest"] = last   # починаємо трейлити від поточної ціни
                    pnl_pct = (last / avg_entry - 1.0) * 100.0
                    await ctx.bot.send_message(
                        chat_id,
                        f"✅ PARTIAL TP {sym_ord}: 50% · avg={avg_entry:.6f} → last={last:.6f} · PnL={pnl_pct:.2f}% · sold={sell_qty:.6f}"
                    )
                # не робимо continue, бо qty змінився тільки на біржі; наступний цикл підхопить залишок
                continue

            # --- ТРЕЙЛІНГ для решти після TP1 ---
            if st_pos["took_tp1"]:
                # оновлюємо максимум
                if last > float(st_pos.get("highest") or 0.0):
                    st_pos["highest"] = last

                peak = float(st_pos.get("highest") or last)
                # якщо просіли від піку на trail_pct — закриваємо решту
                if last <= peak * (1.0 - trail_pct):
                    q = _floor_qty(qty, 6)
                    if q > 0:
                        await place_market_sell_crypto_qty(sym_ord, q)
                        pnl_pct = (last / avg_entry - 1.0) * 100.0
                        await ctx.bot.send_message(
                            chat_id,
                            f"✅ EXIT {sym_ord}: reason=TRAIL · peak={peak:.6f} → last={last:.6f} · PnL={pnl_pct:.2f}% · qty={q:.6f}"
                        )
                    # очищаємо стейт на всяк випадок
                    POS_STATE.pop(sym_ord, None)
                    continue

            # --- Додатковий фільтр виходу за індикаторами (як і було) ---
            # (працює як страховка, якщо TP/SL/TRAIL не спрацювали)
            # Можна залишити або прибрати — на твій вибір:
            # e_fast<e_slow або RSI<50
            # — короткі бари вже є, додаємо довші
            # (щоб не робити зайвих запитів — лише коли нема TP/SL/Trail)
            # ------ OPTIONAL ------
            #bars_long = await get_bars_crypto([sym_data], map_tf(conf["bars"][1]), limit=60)
            #c_short = [float(x["c"]) for x in arr]
            #c_long  = [float(x["c"]) for x in (bars_long.get("bars") or {}).get(sym_data, [])]
            #if c_long and should_exit_by_indicators(conf, c_short, c_long):
            #    q = _floor_qty(qty, 6)
            #    if q > 0:
            #        await place_market_sell_crypto_qty(sym_ord, q)
            #        pnl_pct = (last / avg_entry - 1.0) * 100.0
            #        await ctx.bot.send_message(
            #            chat_id,
            #            f"✅ EXIT {sym_ord}: reason=SIGNAL · avg={avg_entry:.6f} → last={last:.6f} · PnL={pnl_pct:.2f}% · qty={q:.6f}"
            #        )
            #    POS_STATE.pop(sym_ord, None)

        except Exception as e:
            # захищаємося від нульових кількостей та інших дрібних багів
            try:
                await ctx.bot.send_message(chat_id, f"🔴 manage error {sym_ord}: {e}")
            except Exception:
                pass
       # -------- COMMANDS --------
async def start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    stdef(u.effective_chat.id)
    await u.message.reply_text(
        "👋 Алпака-бот готовий.\n"
        "Крипта 24/7; акції — у час біржі. Сканер/автотрейд може працювати у фоні.\n"
        "Увімкнути автотрейд: /alp_on  ·  Зупинити: /alp_off  ·  Стан: /alp_status\n"
        "Фоновий автоскан: /auto_on  ·  /auto_off  ·  /auto_status",
        reply_markup=kb()
    )

async def help_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "/signals_crypto — показати топ-N і (якщо Autotrade=ON) купити\n"
        "/trade_crypto — миттєво торгувати топ-N без звіту\n"
        "/signals_stocks — топ-N для акцій (інформативно)\n"
        "/alp_on /alp_off /alp_status — автотрейд\n"
        "/auto_on /auto_off /auto_status — фоновий автоскан\n"
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
    await u.message.reply_text("Режим входів: SHORT (для крипти ігнорується)")

async def both_mode(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id); st["side_mode"] = "both"
    await u.message.reply_text("Режим входів: BOTH (крипта все одно лише LONG)")

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
            f"Side={st.get('side_mode','long')} · USD_per_trade=${USD_PER_TRADE:.2f}"
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
            # side лише LONG для крипти
            if await get_position(sym):
                await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}")
                continue

            # дубль — тільки якщо ми вже пробували купувати нещодавно
            if too_recent("CRYPTO", sym, "buy"):
                await u.message.reply_text(f"⚪ SKIP (дубль): {sym} BUY")
                continue

            try:
                # маркуємо тільки перед реальною спробою
                mark_trade("CRYPTO", sym, "buy")
                order, filled_qty, fill_price = await crypto_buy_by_usd(sym, USD_PER_TRADE)
                await u.message.reply_text(
                    f"🟢 BUY OK: {sym} · ${USD_PER_TRADE:.2f}\n"
                    f"Entry={fill_price:.6f} · qty={filled_qty:.6f}\n"
                    f"План керування: TP={tp_pct*100:.2f}% · SL={sl_pct*100:.2f}% (фоном)"
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
            if too_recent("CRYPTO", sym, "buy"):
                await u.message.reply_text(f"⚪ SKIP (дубль): {sym} BUY")
                continue

            try:
                mark_trade("CRYPTO", sym, "buy")
                order, filled_qty, fill_price = await crypto_buy_by_usd(sym, USD_PER_TRADE)
                await u.message.reply_text(
                    f"🟢 BUY OK: {sym} · ${USD_PER_TRADE:.2f}\n"
                    f"Entry={fill_price:.6f} · qty={filled_qty:.6f}\n"
                    f"План керування: TP={tp_pct*100:.2f}% · SL={sl_pct*100:.2f}% (фоном)"
                )
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")
    except Exception as e:
        await u.message.reply_text(f"🔴 trade_crypto error: {e}")

# ------- STOCKS commands (інформативно) -------
async def signals_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        report, _ = await scan_rank_stocks(st)
        await u.message.reply_text(report + "\n(Торгівля акціями демо; модуль входів/виходів для акцій вимкнено)")
    except Exception as e:
        await u.message.reply_text(f"🔴 signals_stocks error: {e}")

async def trade_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("ℹ️ У цій збірці автотрейд для акцій відключений. Фокус — крипта.")

# ======= AUTOSCAN (background) =======
async def _auto_scan_once_for_chat(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE):
    st = stdef(chat_id)

    # 1) менеджер позицій
    try:
        await try_manage_crypto_positions(ctx, chat_id)
    except Exception as e:
        try:
            await ctx.bot.send_message(chat_id, f"🔴 manager error: {e}")
        except Exception:
            pass

    # 2) якщо увімкнено автоскар та автотрейд — шукаємо нові входи
    if not (st.get("auto_scan") and st.get("autotrade")):
        return

    try:
        report, ranked = await scan_rank_crypto(st)
    except Exception as e:
        ranked = []
        await ctx.bot.send_message(chat_id, f"🔴 Крипто-скан помилка: {e}")

    conf = _mode_conf(st)
    picks = ranked[: int(conf.get("top_n", max(1, ALPACA_TOP_N)))]

    for _, sym, _ in picks:
        if await get_position(sym):
            continue
        if too_recent("CRYPTO", sym, "buy"):
            continue
        try:
            mark_trade("CRYPTO", sym, "buy")
            _, qty, entry = await crypto_buy_by_usd(sym, USD_PER_TRADE)
            await ctx.bot.send_message(
                chat_id,
                f"🟢 AUTO BUY: {to_order_sym(sym)} · ${USD_PER_TRADE:.2f} · entry={entry:.6f} · qty={qty:.6f}"
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
    # Фоновий автоскан
    app.add_handler(CommandHandler("auto_on", auto_on))
    app.add_handler(CommandHandler("auto_off", auto_off))
    app.add_handler(CommandHandler("auto_status", auto_status))
    app.add_handler(CommandHandler("long_mode", long_mode))
    app.add_handler(CommandHandler("short_mode", short_mode))
    app.add_handler(CommandHandler("both_mode", both_mode))

    app.add_handler(CommandHandler("alp_on", alp_on))
    app.add_handler(CommandHandler("alp_off", alp_off))
    app.add_handler(CommandHandler("alp_status", alp_status))

    # Крипта
    app.add_handler(CommandHandler("signals_crypto", signals_crypto))
    app.add_handler(CommandHandler("trade_crypto", trade_crypto))
    # Акції (інформативно)
    app.add_handler(CommandHandler("signals_stocks", signals_stocks))
    app.add_handler(CommandHandler("trade_stocks", trade_stocks))

    # Фоновий job раз у SCAN_INTERVAL_SEC
    app.job_queue.run_repeating(periodic_auto_scan, interval=SCAN_INTERVAL_SEC, first=10)

    print("Bot started.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()         
                
