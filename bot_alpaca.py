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

# ========= ENV =========

TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_TOKEN") or "").strip()

ALPACA_API_KEY = (os.getenv("ALPACA_API_KEY") or "").strip()
ALPACA_API_SECRET = (os.getenv("ALPACA_API_SECRET") or "").strip()

ALPACA_BASE_URL = (os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_URL = (os.getenv("ALPACA_DATA_URL") or "https://data.alpaca.markets").rstrip("/")

USD_PER_TRADE = float(os.getenv("ALPACA_NOTIONAL") or 50)   # скільки USD на один вхід
ALPACA_TOP_N   = int(os.getenv("ALPACA_TOP_N") or 2)
ALPACA_MAX_CRYPTO  = int(os.getenv("ALPACA_MAX_CRYPTO") or 25)
ALPACA_MAX_STOCKS  = int(os.getenv("ALPACA_MAX_STOCKS") or 50)

# інтервал фонового автоскану в секундах
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC") or 300)
DEDUP_COOLDOWN_MIN = int(os.getenv("DEDUP_COOLDOWN_MIN") or 240)

# ====== GLOBAL STATE (per chat) ======
STATE: Dict[int, Dict[str, Any]] = {}

# коротка пам’ять L2 для scalp
# STATE[chat]['l2_hist'][symbol] -> list of dicts [{'t':timestamp,'bp':...,'bs':...,'ap':...,'as':...,'last':...}] останні 6 знімків
L2_WINDOW = 6

# ====== MODE PROFILES ======
MODE_PARAMS = {
    # tp/sl — для менеджера позицій + трейлу
    "aggressive": {"bars": ("15Min","30Min","1Hour"), "rsi_buy":55.0,"rsi_sell":45.0,
                   "ema_fast":15,"ema_slow":30,"top_n":ALPACA_TOP_N, "tp_pct":0.15, "sl_pct":0.08},  # 15%/8% для довших
    "scalp":      {"bars": ("5Min","15Min","1Hour"), "rsi_buy":58.0,"rsi_sell":42.0,
                   "ema_fast":9,"ema_slow":21,"top_n":ALPACA_TOP_N, "tp_pct":0.20,"sl_pct":0.06},
    "default":    {"bars": ("15Min","30Min","1Hour"), "rsi_buy":56.0,"rsi_sell":44.0,
                   "ema_fast":12,"ema_slow":26,"top_n":ALPACA_TOP_N, "tp_pct":0.12,"sl_pct":0.08},
    "swing":      {"bars": ("30Min","1Hour","1Day"), "rsi_buy":55.0,"rsi_sell":45.0,
                   "ema_fast":20,"ema_slow":40,"top_n":ALPACA_TOP_N, "tp_pct":0.20,"sl_pct":0.10},
    "safe":       {"bars": ("15Min","30Min","1Hour"), "rsi_buy":60.0,"rsi_sell":40.0,
                   "ema_fast":15,"ema_slow":35,"top_n":max(1,ALPACA_TOP_N-1), "tp_pct":0.10,"sl_pct":0.06},
}

USE_ORDERBOOK_FOR_SCALP = True  # увімкнено твій запит: scalp працює через L2

# ====== CRYPTO WHITELIST (USD) ======
CRYPTO_USD_PAIRS = [
    "BTC/USD","ETH/USD","SOL/USD","LTC/USD","DOGE/USD","AVAX/USD","AAVE/USD","MKR/USD","DOT/USD",
    "LINK/USD","UNI/USD","PEPE/USD","XRP/USD","CRV/USD","BCH/USD","BAT/USD","GRT/USD",
    "XTZ/USD","USDC/USD","USDT/USD","USDG/USD","YFI/USD","LDO/USD"
][:ALPACA_MAX_CRYPTO]

# ====== STOCKS UNIVERSE (просто для відображення) ======
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
    st.setdefault("mode","aggressive")
    st.setdefault("autotrade", False)
    st.setdefault("auto_scan", False)
    st.setdefault("side_mode","long")
    st.setdefault("l2_hist", {})  # для scalp
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

# ===== helper: clock & positions =====
async def alp_clock() -> Dict[str, Any]:
    return await alp_get_json("/v2/clock")

async def alp_positions() -> List[Dict[str, Any]]:
    return await alp_get_json("/v2/positions")

async def get_position(sym: str) -> Optional[Dict[str, Any]]:
    try:
        p = await alp_get_json(f"/v2/positions/{to_order_sym(sym)}")
        if p and (float(p.get("qty",0) or 0) != 0):
            return p
        return None
    except Exception:
        return None

# ===== DATA: bars & snapshots =====
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

async def get_snapshots_crypto(pairs: List[str]) -> Dict[str, Any]:
    # містить latest trade & quote (bp/bs/ap/as)
    syms = ",".join([to_data_sym(p) for p in pairs])
    path = "/v1beta3/crypto/us/snapshots"
    params = {"symbols": syms}
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

# -------- L2 helpers (для scalp) --------
def _l2_push(st: Dict[str, Any], sym: str, bp: float, bs: float, ap: float, ask_s: float, last: float):
    hist = st.setdefault("l2_hist", {}).setdefault(sym, [])
    hist.append({"t": now_s(), "bp": bp, "bs": bs, "ap": ap, "as": ask_s, "last": last})
    if len(hist) > L2_WINDOW:
        del hist[:len(hist)-L2_WINDOW]

def _l2_imbalance(bs: float, ask_s: float) -> float:
    denom = max(1e-9, (bs + ask_s))
    return (bs - ask_s) / denom

def l2_long_entry_signal(st: Dict[str, Any], sym: str) -> bool:
    hist = st.get("l2_hist", {}).get(sym, [])
    if len(hist) < 3:  # потрібно кілька знімків
        return False
    # останні три
    h1, h2, h3 = hist[-3], hist[-2], hist[-1]
    im1 = _l2_imbalance(h1["bs"], h1["as"])
    im2 = _l2_imbalance(h2["bs"], h2["as"])
    im3 = _l2_imbalance(h3["bs"], h3["as"])
    # позитивний дисбаланс з наростанням
    return (im3 >= 0.20) and (im2 >= 0.10) and (im3 > im2 > im1)

def l2_exit_signal(st: Dict[str, Any], sym: str) -> bool:
    hist = st.get("l2_hist", {}).get(sym, [])
    if len(hist) < 3:
        return False
    h1, h2, h3 = hist[-3], hist[-2], hist[-1]
    im2 = _l2_imbalance(h2["bs"], h2["as"])
    im3 = _l2_imbalance(h3["bs"], h3["as"])
    # розворот дисбалансу в мінус
    return (im3 <= -0.10) and (im3 < im2)

# -------- ORDER UTILS --------
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
    px = await get_last_price_crypto(sym)
    raw_qty = (float(usd_notional) / max(1e-9, px)) * 0.995
    qty = _floor_qty(raw_qty, 6)
    if qty <= 0:
        raise RuntimeError("qty<=0 (too small notional)")
    order = await place_market_buy_crypto_qty(sym, qty)

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

def ema_cross_down(conf: Dict[str, Any], closes_long: List[float]) -> bool:
    e_fast = ema(closes_long, conf["ema_fast"])
    e_slow = ema(closes_long, conf["ema_slow"])
    return bool(e_fast and e_slow and e_fast[-1] < e_slow[-1])

def should_exit_by_indicators(conf: Dict[str, Any], closes_short: List[float], closes_long: List[float]) -> bool:
    r = rsi(closes_short, 14)
    return bool(ema_cross_down(conf, closes_long) or r < 50.0)

async def try_manage_crypto_positions(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int):
    st = stdef(chat_id)
    conf = _mode_conf(st)
    tp_pct = float(conf.get("tp_pct", 0.20 if st.get("mode")=="scalp" else 0.12))
    sl_pct = float(conf.get("sl_pct", 0.06))

    positions = await alp_positions()
    crypto_positions = [p for p in positions if p.get("asset_class") == "crypto"]

    syms = [to_data_sym(p["symbol"]) for p in crypto_positions]
    if not syms:
        return

    bars_s = await get_bars_crypto(syms, map_tf(conf["bars"][0]), limit=120)
    bars_l = await get_bars_crypto(syms, map_tf(conf["bars"][1]), limit=120)

    # також підтягнемо snapshots для L2-виходів (scalp)
    snaps = {}
    if USE_ORDERBOOK_FOR_SCALP and st.get("mode") == "scalp":
        try:
            snaps = await get_snapshots_crypto(syms)
        except Exception:
            snaps = {}

    for p in crypto_positions:
        sym_ord = p["symbol"]
        sym_data = to_data_sym(sym_ord)
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

            # трейлінг: коли профіт >10%, фіксувати при відкаті ~3%
            take_profit = last >= avg_entry * 1.10 and (last <= max(c_short[-5:]) * 0.97)
            hard_tp     = last >= avg_entry * (1.0 + tp_pct)
            stop_loss   = last <= avg_entry * (1.0 - sl_pct)

            l2_exit = False
            if USE_ORDERBOOK_FOR_SCALP and st.get("mode") == "scalp":
                snap = (snaps.get("snapshots") or {}).get(sym_data) or {}
                q = snap.get("latestQuote") or {}
                bp, bs, ap, ask_s = float(q.get("bp") or 0), float(q.get("bs") or 0), float(q.get("ap") or 0), float(q.get("as") or 0)
                last_trade = (snap.get("latestTrade") or {})
                last_px = float(last_trade.get("p") or last)
                _l2_push(st, sym_data, bp, bs, ap, ask_s, last_px)
                l2_exit = l2_exit_signal(st, sym_data)

            by_filters = bool(c_long) and should_exit_by_indicators(conf, c_short, c_long)

            if hard_tp or stop_loss or take_profit or l2_exit or by_filters:
                await place_market_sell_crypto_qty(sym_ord, qty)
                reason = (
                    "HARD_TP" if hard_tp else
                    ("SL" if stop_loss else ("TRAIL" if take_profit else ("L2" if l2_exit else "SIGNAL")))
                )
                pnl_pct = (last / avg_entry - 1.0) * 100.0 if avg_entry > 0 else 0.0
                await ctx.bot.send_message(
                    chat_id,
                    f"✅ EXIT {sym_ord}: reason={reason} · avg={avg_entry:.6f} → last={last:.6f} · PnL={pnl_pct:.2f}% · qty={qty:.6f}"
                )
        except Exception as e:
            try:
                await ctx.bot.send_message(chat_id, f"🔴 manage position error {sym_ord}: {e}")
            except Exception:
                pass

# -------- SCAN (CRYPTO) --------
async def scan_rank_crypto(st: Dict[str, Any]) -> Tuple[str, List[Tuple[float,str,List[Dict[str,Any]]]]]:
    conf = _mode_conf(st)
    tf1, tf2, tf3 = map_tf(conf["bars"][0]), map_tf(conf["bars"][1]), map_tf(conf["bars"][2])

    pairs = CRYPTO_USD_PAIRS[:]
    data_pairs = [to_data_sym(p) for p in pairs]

    bars1 = await get_bars_crypto(data_pairs, tf1, limit=120)
    bars2 = await get_bars_crypto(data_pairs, tf2, limit=120)
    bars3 = await get_bars_crypto(data_pairs, tf3, limit=120)

    ranked: List[Tuple[float,str,List[Dict[str,Any]]]] = []
    for sym in data_pairs:
        r1 = (bars1.get("bars") or {}).get(sym, [])
        r2 = (bars2.get("bars") or {}).get(sym, [])
        r3 = (bars3.get("bars") or {}).get(sym, [])
        if not r1 or not r2 or not r3:
            continue
        c1 = [float(x["c"]) for x in r1]
        c2 = [float(x["c"]) for x in r2]
        c3 = [float(x["c"]) for x in r3]
        score = rank_score(c1,c2,c3, conf["rsi_buy"],conf["rsi_sell"], conf["ema_fast"],conf["ema_slow"])
        ranked.append((score, sym, r1))

    ranked.sort(reverse=True)
    rep = (
        "🛰️ Сканер (крипта):\n"
        f"• Активних USD-пар: {len(data_pairs)}\n"
        f"• Використаємо для торгівлі (лімітом): {min(conf['top_n'], len(ranked))}\n"
        + (f"• Перші 25: " + ", ".join([s for _, s, _ in ranked[:25]]) if ranked else "• Немає сигналів")
    )
    return rep, ranked

# -------- SCALP entries via L2 --------
async def scalp_l2_pick_and_enter(st: Dict[str, Any], chat_id: int, ctx: ContextTypes.DEFAULT_TYPE):
    # беремо топ-N за індикаторним ранком, але підтверджуємо вхід L2
    report, ranked = await scan_rank_crypto(st)
    await ctx.bot.send_message(chat_id, report)

    if not ranked:
        return
    picks = ranked[: _mode_conf(st)["top_n"]]
    syms = [s for _, s, _ in picks]
    snaps = await get_snapshots_crypto(syms)

    for _, sym, _ in picks:
        if await get_position(sym):  # вже є позиція
            await ctx.bot.send_message(chat_id, f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}")
            continue
        if skip_as_duplicate("CRYPTO", sym, "buy"):
            await ctx.bot.send_message(chat_id, f"⚪ SKIP (дубль): {sym} BUY")
            continue

        snap = (snaps.get("snapshots") or {}).get(sym) or {}
        q = snap.get("latestQuote") or {}
        t = snap.get("latestTrade") or {}
        bp, bs = float(q.get("bp") or 0), float(q.get("bs") or 0)
        ap, ask_s = float(q.get("ap") or 0), float(q.get("as") or 0)
        last = float(t.get("p") or 0)
        _l2_push(st, sym, bp, bs, ap, ask_s, last)

        # Фільтри свічками (5m) + L2 тригер
        bars = await get_bars_crypto([sym], "5Min", limit=50)
        c = [float(x["c"]) for x in (bars.get("bars") or {}).get(sym, [])]
        if not c:
            continue
        ef = ema(c, _mode_conf(st)["ema_fast"])
        es = ema(c, _mode_conf(st)["ema_slow"])
        r = rsi(c, 14)
        trend_ok = bool(ef and es and ef[-1] > es[-1])
        rsi_ok = r > 50.0
        l2_ok = l2_long_entry_signal(st, sym)

        if trend_ok and rsi_ok and l2_ok:
            try:
                _, qty, entry = await crypto_buy_by_usd(sym, USD_PER_TRADE)
                await ctx.bot.send_message(
                    chat_id,
                    f"🟢 AUTO BUY: {to_order_sym(sym)} · ${USD_PER_TRADE:.2f} · entry={entry:.6f} · qty={qty:.6f}"
                )
            except Exception as e:
                await ctx.bot.send_message(chat_id, f"🔴 ORDER FAIL {sym} BUY: {e}")
        else:
            # не виконані умови L2/індикаторів — пропустимо цього разу
            pass

# -------- COMMANDS --------
async def start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    stdef(u.effective_chat.id)
    await u.message.reply_text(
        "👋 Алпака-бот готовий.\n"
        "Крипта 24/7. Режим /scalp працює через ‘стакан’ (bid/ask) + індикатори.\n"
        "Увімкнути автотрейд: /alp_on · Зупинити: /alp_off · Стан: /alp_status\n"
        "Фоновий автоскан+менеджер: /auto_on · /auto_off · /auto_status",
        reply_markup=kb()
    )

async def help_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "/signals_crypto — топ-N (для scalp теж підкачає L2 у фоні)\n"
        "/trade_crypto — миттєві входи за поточним режимом\n"
        "/signals_stocks — топ-N акцій (інфо)\n"
        "/alp_on /alp_off /alp_status — автотрейд\n"
        "/auto_on /auto_off /auto_status — автоскан+менеджер позицій\n"
        "/long_mode /short_mode /both_mode — напрям (крипта тільки LONG)\n"
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
            f"Side={st.get('side_mode','long')} · USD_per_trade=${USD_PER_TRADE:.2f}"
        )
        await u.message.reply_text(txt)
    except Exception as e:
        await u.message.reply_text(f"🔴 alp_status error: {e}")

# ------- CRYPTO commands -------
async def signals_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        if USE_ORDERBOOK_FOR_SCALP and st.get("mode") == "scalp":
            # поки що лише показ рангу; L2 підтвердження у фоні в autoscan/trade
            report, ranked = await scan_rank_crypto(st)
            await u.message.reply_text(report)
        else:
            report, ranked = await scan_rank_crypto(st)
            await u.message.reply_text(report)

        if not st.get("autotrade"):
            return

        if USE_ORDERBOOK_FOR_SCALP and st.get("mode") == "scalp":
            # у scalp входимо через L2
            await scalp_l2_pick_and_enter(st, u.effective_chat.id, c)
            return

        # інші режими — простий buy за top-N
        _, ranked = await scan_rank_crypto(st)
        if not ranked: return
        picks = ranked[: _mode_conf(st)["top_n"]]
        for _, sym, _ in picks:
            if await get_position(sym):
                await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}")
                continue
            if skip_as_duplicate("CRYPTO", sym, "buy"):
                await u.message.reply_text(f"⚪ SKIP (дубль): {sym} BUY")
                continue
            try:
                _, qty, entry = await crypto_buy_by_usd(sym, USD_PER_TRADE)
                await u.message.reply_text(f"🟢 BUY OK: {sym} · ${USD_PER_TRADE:.2f}\nEntry={entry:.6f} · qty={qty:.6f}")
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")

    except Exception as e:
        await u.message.reply_text(f"🔴 signals_crypto error: {e}")

async def trade_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        if USE_ORDERBOOK_FOR_SCALP and st.get("mode") == "scalp":
            await scalp_l2_pick_and_enter(st, u.effective_chat.id, c)
            return

        _, ranked = await scan_rank_crypto(st)
        if not ranked:
            await u.message.reply_text("⚠️ Немає сигналів")
            return
        picks = ranked[: _mode_conf(st)["top_n"]]
        for _, sym, _ in picks:
            if await get_position(sym):
                await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}")
                continue
            if skip_as_duplicate("CRYPTO", sym, "buy"):
                await u.message.reply_text(f"⚪ SKIP (дубль): {sym} BUY")
                continue
            try:
                _, qty, entry = await crypto_buy_by_usd(sym, USD_PER_TRADE)
                await u.message.reply_text(f"🟢 BUY OK: {sym} · ${USD_PER_TRADE:.2f}\nEntry={entry:.6f} · qty={qty:.6f}")
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")
    except Exception as e:
        await u.message.reply_text(f"🔴 trade_crypto error: {e}")

# ------- STOCKS commands (інформативно) -------
async def signals_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stdef(u.effective_chat.id)
    try:
        report, _ = await scan_rank_stocks(st)
        await u.message.reply_text(report + "\n(Торгівля акціями демо; фокус — крипта)")
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

    # 2) нові входи лише якщо AutoScan+Autotrade
    if not (st.get("auto_scan") and st.get("autotrade")):
        return

    try:
        if USE_ORDERBOOK_FOR_SCALP and st.get("mode") == "scalp":
            await scalp_l2_pick_and_enter(st, chat_id, ctx)
        else:
            report, ranked = await scan_rank_crypto(st)
            await ctx.bot.send_message(chat_id, report)
            conf = _mode_conf(st)
            picks = ranked[: int(conf.get("top_n", max(1, ALPACA_TOP_N)))]
            for _, sym, _ in picks:
                if await get_position(sym): continue
                if skip_as_duplicate("CRYPTO", sym, "buy"): continue
                try:
                    _, qty, entry = await crypto_buy_by_usd(sym, USD_PER_TRADE)
                    await ctx.bot.send_message(chat_id, f"🟢 AUTO BUY: {to_order_sym(sym)} · ${USD_PER_TRADE:.2f} · entry={entry:.6f} · qty={qty:.6f}")
                except Exception as e:
                    await ctx.bot.send_message(chat_id, f"🔴 AUTO ORDER FAIL {sym}: {e}")
    except Exception as e:
        try:
            await ctx.bot.send_message(chat_id, f"🔴 periodic autoscan error: {e}")
        except Exception:
            pass

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

    app.job_queue.run_repeating(periodic_auto_scan, interval=SCAN_INTERVAL_SEC, first=10)

    print("Bot started.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
