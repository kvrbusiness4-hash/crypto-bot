# -*- coding: utf-8 -*-

import os
import json
import math
import asyncio
import time
from typing import Dict, Any, Tuple, List, Optional

from aiohttp import ClientSession, ClientTimeout
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes

# ========= ENV / CONSTANTS =========

TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_TOKEN") or "").strip()

ALPACA_API_KEY = (os.getenv("ALPACA_API_KEY") or "").strip()
ALPACA_API_SECRET = (os.getenv("ALPACA_API_SECRET") or "").strip()

ALPACA_BASE_URL = (os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_URL = (os.getenv("ALPACA_DATA_URL") or "https://data.alpaca.markets").rstrip("/")

# скільки USD виділяємо на один крипто-вхід (через qty=usd/price)
USD_PER_TRADE = float(os.getenv("USD_PER_TRADE") or 50.0)

ALPACA_TOP_N = int(os.getenv("ALPACA_TOP_N") or 2)
ALPACA_MAX_CRYPTO = int(os.getenv("ALPACA_MAX_CRYPTO") or 25)
ALPACA_MAX_STOCKS = int(os.getenv("ALPACA_MAX_STOCKS") or 50)

# Частоти фон. задач
RANK_INTERVAL_SEC = int(os.getenv("RANK_INTERVAL_SEC") or 45)   # скан/ранжування
MANAGE_TICK_SEC   = int(os.getenv("MANAGE_TICK_SEC")   or 3)    # менеджер позицій

DEDUP_COOLDOWN_MIN = int(os.getenv("DEDUP_COOLDOWN_MIN") or 240)

# ====== GLOBAL STATE (per chat) ======
STATE: Dict[int, Dict[str, Any]] = {}
PEAKS: Dict[str, float] = {}   # локальні піки ціни по символам (для трейлінгу)

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

# ====== STOCKS UNIVERSE (інформативно) ======
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
    st.setdefault("usd_per_trade", USD_PER_TRADE)
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

# -------- HTTP with retry/backoff ----------
def _alp_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }

async def _with_retry(coro_fn, *args, **kwargs):
    delay = 0.0
    for _ in range(6):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "timeout" in msg.lower():
                delay = min(10.0, delay*2 or 1.5)
                await asyncio.sleep(delay)
                continue
            raise

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
    return await _with_retry(alp_get_json, "/v2/clock")

async def alp_positions() -> List[Dict[str, Any]]:
    return await _with_retry(alp_get_json, "/v2/positions")

async def get_position(sym: str) -> Optional[Dict[str, Any]]:
    try:
        p = await _with_retry(alp_get_json, f"/v2/positions/{to_order_sym(sym)}")
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
    bars = await _with_retry(get_bars_crypto, [data_sym], "5Min", 2)
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
    return await _with_retry(alp_post_json, "/v2/orders", payload)

async def place_market_sell_crypto_qty(sym: str, qty: float) -> dict:
    if qty <= 0:
        raise RuntimeError("qty must be > 0")
    payload = {
        "symbol": to_order_sym(sym),
        "side": "sell",
        "type": "market",
        "time_in_force": "gtc",
        "asset_class": "crypto",
        "qty": f"{_floor_qty(qty):.6f}",
    }
    return await _with_retry(alp_post_json, "/v2/orders", payload)

async def get_order(order_id: str) -> dict:
    return await _with_retry(alp_get_json, f"/v2/orders/{order_id}")

# ======== ENTRY & EXIT LOGIC ========

async def crypto_buy_by_usd(sym: str, usd_notional: float) -> Tuple[dict, float, float]:
    px = await get_last_price_crypto(sym)
    raw_qty = (float(usd_notional) / max(1e-9, px)) * 0.995
    qty = _floor_qty(raw_qty, 6)
    if qty <= 0:
        raise RuntimeError("qty<=0 (too small usd_notional)")
    order = await place_market_buy_crypto_qty(sym, qty)

    # чекаємо заповнення для середньої ціни
    order_id = order.get("id", "")
    filled_qty, fill_price = 0.0, px
    for _ in range(12):
        od = await get_order(order_id)
        status = od.get("status")
        if status in ("filled", "partially_filled"):
            try: filled_qty = float(od.get("filled_qty") or 0)
            except: filled_qty = 0.0
            try: fill_price = float(od.get("filled_avg_price") or px)
            except: fill_price = px
            if status == "filled":
                break
        await asyncio.sleep(0.5)

    # ініціалізуємо пік
    if filled_qty > 0:
        PEAKS[to_order_sym(sym)] = fill_price
    return order, filled_qty, fill_price

def should_exit_by_indicators(conf: Dict[str, Any], closes_short: List[float], closes_long: List[float]) -> bool:
    e_fast = ema(closes_long, conf["ema_fast"])
    e_slow = ema(closes_long, conf["ema_slow"])
    r = rsi(closes_short, 14)
    cross_down = (e_fast and e_slow and e_fast[-1] < e_slow[-1])
    weak_rsi = r < 50.0
    return bool(cross_down or weak_rsi)

def trail_should_exit(sym_ord: str, last: float, avg_entry: float) -> bool:
    """
    Динамічний трейл:
      - якщо прибуток >= 10%, фіксуємо пік і виходимо при відкаті ~1.5%
      - якщо прибуток < 10%, відкат чуть ширший ~3%
    """
    if avg_entry <= 0: 
        return False
    key = sym_ord.upper()
    peak = PEAKS.get(key, last)
    if last > peak: 
        PEAKS[key] = last
        peak = last

    gain = last/avg_entry - 1.0
    if gain >= 0.10:
        retrace = (peak - last) / peak
        return retrace >= 0.015
    else:
        retrace = (peak - last) / peak
        return retrace >= 0.03

async def try_manage_crypto_positions(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int):
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

            # базові умови
            take_profit = last >= avg_entry * (1.0 + tp_pct)
            stop_loss   = last <= avg_entry * (1.0 - sl_pct)
            by_filters  = should_exit_by_indicators(conf, c_short, c_long) if c_long else False
            by_trail    = trail_should_exit(sym_ord, last, avg_entry)

            if take_profit or stop_loss or by_filters or by_trail:
                # перевірити поточну позицію ще раз перед відправкою ордера
                fresh = await get_position(sym_ord)
                real_qty = float((fresh or {}).get("qty") or 0)
                if real_qty <= 0:
                    continue
                await place_market_sell_crypto_qty(sym_ord, real_qty)
                reason = "TP" if take_profit else ("SL" if stop_loss else ("TRAIL" if by_trail else "SIGNAL"))
                pnl_pct = (last / avg_entry - 1.0) * 100.0 if avg_entry > 0 else 0.0
                await ctx.bot.send_message(
                    chat_id,
                    f"✅ EXIT {sym_ord}: reason={reason} · avg={avg_entry:.6f} → last={last:.6f} "
                    f"· PnL={pnl_pct:.2f}% · qty={real_qty:.6f}"
                )
                # очистити пік після виходу
                PEAKS.pop(to_order_sym(sym_ord), None)
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

# ====== HIGH-LEVEL SCAN & ENTER ======
async def _scan_and_maybe_enter(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE):
    st = stdef(chat_id)
    report, ranked = await scan_rank_crypto(st)
    conf = _mode_conf(st)
    picks = ranked[: int(conf.get("top_n", max(1, ALPACA_TOP_N)))]

    for _, sym, _ in picks:
        try:
            if await get_position(sym):
                continue
            if skip_as_duplicate("CRYPTO", sym, "buy"):
                continue
            usd = float(st.get("usd_per_trade", USD_PER_TRADE))
            _, qty, entry = await crypto_buy_by_usd(sym, usd)
            await ctx.bot.send_message(
                chat_id,
                f"🟢 AUTO BUY: {to_order_sym(sym)} · ${usd:.2f} · entry={entry:.6f} · qty={qty:.6f}"
            )
        except Exception as e:
            await ctx.bot.send_message(chat_id, f"🔴 AUTO ORDER FAIL {sym}: {e}")

# -------- COMMANDS --------
async def start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    stdef(u.effective_chat.id)
    await u.message.reply_text(
        "👋 Алпака-бот готовий.\n"
        "Крипта торгується 24/7; акції — коли ринок відкритий. Сканер/автотрейд у фоні.\n"
        "Увімкнути автотрейд: /alp_on  ·  Зупинити: /alp_off  ·  Стан: /alp_status\n"
        "Фоновий автоскан/менеджер: /auto_on  ·  /auto_off  ·  /auto_status",
        reply_markup=kb()
    )

async def help_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "/signals_crypto — показати топ-N і (якщо Autotrade=ON) купити\n"
        "/trade_crypto — миттєво торгувати топ-N без звіту\n"
        "/signals_stocks — показати топ-N для акцій (інфо)\n"
        "/alp_on /alp_off /alp_status — автотрейд\n"
        "/auto_on /auto_off /auto_status — фонові задачі\n"
        "/long_mode /short_mode /both_mode — напрям (short для крипти ігнорується)\n"
        "/aggressive /scalp /default /swing /safe — профілі",
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
    await u.message.reply_text("Режим входів: SHORT (для крипти ігноруємо)")

async def both_mode(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id); st["side_mode"] = "both"
    await u.message.reply_text("Режим входів: BOTH (для крипти фактично LONG)")

async def alp_on(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id); st["autotrade"] = True
    await u.message.reply_text("✅ Alpaca AUTOTRADE: ON")

async def alp_off(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id); st["autotrade"] = False
    await u.message.reply_text("⛔ Alpaca AUTOTRADE: OFF")

async def alp_status(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        acc = await _with_retry(alp_get_json, "/v2/account")
        st = stdef(u.effective_chat.id)
        usd = float(st.get("usd_per_trade", USD_PER_TRADE))
        txt = (
            "📦 Alpaca:\n"
            f"• status={acc.get('status','UNKNOWN')}\n"
            f"• cash=${float(acc.get('cash',0)):.2f}\n"
            f"• buying_power=${float(acc.get('buying_power',0)):.2f}\n"
            f"• equity=${float(acc.get('equity',0)):.2f}\n"
            f"Mode={st.get('mode','default')} · Autotrade={'ON' if st.get('autotrade') else 'OFF'} · "
            f"AutoScan={'ON' if st.get('auto_scan') else 'OFF'} · "
            f"Side={st.get('side_mode','long')} · USD_per_trade=${usd:.2f}"
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
        usd = float(st.get("usd_per_trade", USD_PER_TRADE))

        for _, sym, _ in picks:
            if await get_position(sym):
                await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}")
                continue
            if skip_as_duplicate("CRYPTO", sym, "buy"):
                await u.message.reply_text(f"⚪ SKIP (дубль): {sym} BUY")
                continue

            try:
                _, filled_qty, fill_price = await crypto_buy_by_usd(sym, usd)
                await u.message.reply_text(
                    f"🟢 BUY OK: {sym} · ${usd:.2f}\n"
                    f"Entry={fill_price:.6f} · qty={filled_qty:.6f}\n"
                    f"План керування: TP={tp_pct*100:.2f}% · SL={sl_pct*100:.2f}% · TRAIL on"
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
        usd = float(st.get("usd_per_trade", USD_PER_TRADE))

        for _, sym, _ in picks:
            if await get_position(sym):
                await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}")
                continue
            if skip_as_duplicate("CRYPTO", sym, "buy"):
                await u.message.reply_text(f"⚪ SKIP (дубль): {sym} BUY")
                continue

            try:
                _, filled_qty, fill_price = await crypto_buy_by_usd(sym, usd)
                await u.message.reply_text(
                    f"🟢 BUY OK: {sym} · ${usd:.2f}\n"
                    f"Entry={fill_price:.6f} · qty={filled_qty:.6f}\n"
                    f"План керування: TP={tp_pct*100:.2f}% · SL={sl_pct*100:.2f}% · TRAIL on"
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
        await u.message.reply_text(report + "\n(Торгівля акціями у цій збірці вимкнена)")
    except Exception as e:
        await u.message.reply_text(f"🔴 signals_stocks error: {e}")

async def trade_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("ℹ️ У цій збірці автотрейд для акцій вимкнено. Фокус — крипта.")

# ======= BACKGROUND TASKS =======

async def ranking_tick(ctx: ContextTypes.DEFAULT_TYPE):
    for chat_id in list(STATE.keys()):
        st = stdef(chat_id)
        if not (st.get("auto_scan") and st.get("autotrade")):
            continue
        try:
            await _scan_and_maybe_enter(chat_id, ctx)
        except Exception as e:
            try: await ctx.bot.send_message(chat_id, f"🔴 ranking error: {e}")
            except: pass

async def manage_tick(ctx: ContextTypes.DEFAULT_TYPE):
    for chat_id in list(STATE.keys()):
        try:
            await try_manage_crypto_positions(ctx, chat_id)
        except Exception as e:
            try: await ctx.bot.send_message(chat_id, f"🔴 manager error: {e}")
            except: pass

# ------- AUTOSCAN commands -------
async def auto_on(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id); st["auto_scan"] = True
    await u.message.reply_text(f"✅ AUTO-SCAN: ON (ranking {RANK_INTERVAL_SEC}s, manage {MANAGE_TICK_SEC}s)")

async def auto_off(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id); st["auto_scan"] = False
    await u.message.reply_text("⛔ AUTO-SCAN: OFF")

async def auto_status(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    await u.message.reply_text(
        f"AutoScan={'ON' if st.get('auto_scan') else 'OFF'}; "
        f"Autotrade={'ON' if st.get('autotrade') else 'OFF'}; "
        f"Mode={st.get('mode','default')} · Side={st.get('side_mode','long')} · "
        f"Ranking={RANK_INTERVAL_SEC}s · Manage={MANAGE_TICK_SEC}s"
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
    # Акції (інформативно)
    app.add_handler(CommandHandler("signals_stocks", signals_stocks))
    app.add_handler(CommandHandler("trade_stocks", trade_stocks))

    # Авто-менеджер та скан розділено
    app.add_handler(CommandHandler("auto_on", auto_on))
    app.add_handler(CommandHandler("auto_off", auto_off))
    app.add_handler(CommandHandler("auto_status", auto_status))

    # Фонові задачі
    app.job_queue.run_repeating(manage_tick,  interval=MANAGE_TICK_SEC, first=5)
    app.job_queue.run_repeating(ranking_tick, interval=RANK_INTERVAL_SEC, first=7)

    print("Bot started.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
