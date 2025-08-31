# -*- coding: utf-8 -*-
# bot_autotrade.py

import os, hmac, hashlib, time, math, asyncio, aiohttp, logging, json
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ============ ENV ============
TG_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BYBIT_KEY   = os.getenv("BYBIT_API_KEY", "").strip()
BYBIT_SEC   = os.getenv("BYBIT_API_SECRET", "").strip()
BYBIT_URL   = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").rstrip("/")
BYBIT_PROXY = os.getenv("BYBIT_PROXY", "").strip()  # http://user:pass@ip:port

# Параметри трейдингу
SIZE_USDT        = float(os.getenv("SIZE_USDT", "5"))
LEVERAGE         = int(os.getenv("LEVERAGE", "3"))
SL_PCT           = float(os.getenv("SL_PCT", "3"))
TP_PCT           = float(os.getenv("TP_PCT", "5"))
MAX_OPEN_POS     = int(os.getenv("MAX_OPEN_POS", "2"))
DEFAULT_AUTO_MIN = int(os.getenv("DEFAULT_AUTO_MIN", "15"))
TOP_N            = int(os.getenv("TOP_N", "2"))
TRADE_ENABLED    = os.getenv("TRADE_ENABLED", "ON").upper() == "ON"
AUTO_LEVERAGE    = os.getenv("AUTO_LEVERAGE", "ON").upper() == "ON"
HEDGE_MODE       = os.getenv("HEDGE_MODE", "OFF").upper() == "ON"

# Логи
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bot")

# ============ UI (компактна клавіатура) ============
KB = ReplyKeyboardMarkup(
    [
        ["/signals"],
        [f"/auto_on {DEFAULT_AUTO_MIN}", "/auto_off"],
        ["/status"],
    ],
    resize_keyboard=True
)

STATE: Dict[int, Dict[str, int | bool | float]] = {}

# ============ Helpers ============
def split_long(text: str, n: int = 3500) -> List[str]:
    out = []
    while len(text) > n:
        out.append(text[:n]); text = text[n:]
    out.append(text)
    return out

def utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

def _proxy_kwargs() -> dict:
    if BYBIT_PROXY.startswith(("http://", "https://")):
        return {"proxy": BYBIT_PROXY}
    return {}

# ============ Indicators ============
def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1:
        return series[:]
    k = 2 / (period + 1)
    out = [series[0]]
    for x in series[1:]:
        out.append(out[-1] + k * (x - out[-1]))
    return out

def rsi(series: List[float], period: int = 14) -> List[float]:
    if len(series) < period + 1: return []
    gains, losses = [], []
    for i in range(1, len(series)):
        d = series[i] - series[i-1]
        gains.append(max(0.0, d)); losses.append(max(0.0, -d))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [0.0] * period
    rsis.append(100.0 if avg_loss == 0 else 100.0 - 100.0/(1.0 + (avg_gain/(avg_loss+1e-9))))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        rsis.append(100.0 if avg_loss == 0 else 100.0 - 100.0/(1.0 + avg_gain/(avg_loss+1e-9)))
    return rsis

def macd(series: List[float], fast=12, slow=26, signal=9) -> Tuple[List[float], List[float]]:
    if len(series) < slow + signal: return [], []
    ef = ema(series, fast); es = ema(series, slow)
    macd_line = [a - b for a, b in zip(ef[-len(es):], es)]
    sig = ema(macd_line, signal)
    L = min(len(macd_line), len(sig))
    return macd_line[-L:], sig[-L:]

def votes_from_series(series: List[float]) -> Dict[str, int | float]:
    out = {"vote": 0, "rsi": None, "ema_trend": 0, "macd": None, "sig": None}
    if len(series) < 60: return out
    rr = rsi(series, 14)
    m, s = macd(series)
    e50 = ema(series, 50)
    e200 = ema(series, 200) if len(series) >= 200 else ema(series, max(100, len(series)//2))
    if rr:
        out["rsi"] = rr[-1]
        if rr[-1] <= 30: out["vote"] += 1
        if rr[-1] >= 70: out["vote"] -= 1
    if m and s:
        out["macd"], out["sig"] = m[-1], s[-1]
        if m[-1] > s[-1]: out["vote"] += 1
        if m[-1] < s[-1]: out["vote"] -= 1
    if e50 and e200:
        out["ema_trend"] = 1 if e50[-1] > e200[-1] else -1
        out["vote"] += 1 if e50[-1] > e200[-1] else -1
    return out

def decide_direction(v15:int, v30:int, v60:int) -> Optional[str]:
    total = v15 + v30 + v60
    pos = sum(1 for v in [v15, v30, v60] if v > 0)
    neg = sum(1 for v in [v15, v30, v60] if v < 0)
    if total >= 3 and pos >= 2: return "LONG"
    if total <= -3 and neg >= 2: return "SHORT"
    return None

def auto_sl_tp_by_vol(series: List[float], px: float) -> Tuple[float,float]:
    tail = series[-48:] if len(series) >= 48 else series
    if len(tail) >= 2:
        mean = sum(tail)/len(tail)
        var = sum((x-mean)**2 for x in tail)/len(tail)
        vol_pct = (math.sqrt(var)/px)*100.0
    else:
        vol_pct = 1.0
    sl = max(0.6, min(3.0, 0.7*vol_pct))
    tp = max(0.8, min(5.0, 1.2*vol_pct))
    return sl, tp

# ============ HTTP (public) ============
BYBIT = "https://api.bybit.com"

async def http_json(session: aiohttp.ClientSession, url: str, params: dict | None = None) -> dict:
    delay = 0.7
    for i in range(5):
        try:
            async with session.get(url, params=params, timeout=25, **_proxy_kwargs()) as r:
                r.raise_for_status()
                return await r.json()
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                await asyncio.sleep(delay); delay *= 1.8; continue
            if i == 4: raise
            await asyncio.sleep(delay); delay *= 1.5
        except Exception:
            if i == 4: raise
            await asyncio.sleep(delay); delay *= 1.5

async def bybit_top_symbols(session: aiohttp.ClientSession, top:int=15) -> List[dict]:
    data = await http_json(session, f"{BYBIT}/v5/market/tickers", {"category": "linear"})
    if not isinstance(data, dict):
        log.error("tickers: unexpected response type: %r", type(data)); return []
    result = data.get("result") or {}
    lst = result.get("list") or []
    if not isinstance(lst, list):
        log.error("tickers: 'list' not a list: %r", lst); return []
    cleaned = [x for x in lst if isinstance(x, dict) and str(x.get("symbol","")).endswith("USDT")]
    def _volume(x: dict) -> float:
        try: return float(x.get("turnover24h") or 0)
        except: return 0.0
    cleaned.sort(key=_volume, reverse=True)
    return cleaned[:top]

async def bybit_klines(session: aiohttp.ClientSession, symbol: str, interval: str, limit: int = 300) -> List[float]:
    data = await http_json(session, f"{BYBIT}/v5/market/kline", {
        "category":"linear","symbol":symbol,"interval":interval,"limit":str(limit)
    })
    rows = list(reversed(((data.get("result") or {}).get("list")) or []))
    closes = []
    for r in rows:
        try: closes.append(float(r[4]))
        except: pass
    return closes

# --- instruments info & qty/price normalization ---
async def get_instrument_info(session: aiohttp.ClientSession, symbol: str) -> dict:
    data = await http_json(session, f"{BYBIT}/v5/market/instruments-info",
                           {"category":"linear","symbol":symbol})
    if not isinstance(data, dict):
        log.error("No instrument info (type) for %s: %r", symbol, type(data)); return {}
    lst = ((data.get("result") or {}).get("list")) or []
    if not lst:
        log.error("Empty instruments list for %s: %s", symbol, data); return {}
    return lst[0]

def _round_step(value: float, step: float) -> float:
    if step <= 0: return value
    return math.floor(value / step) * step

def normalize_qty(symbol_info: dict, qty: float) -> float:
    lot = (symbol_info.get("lotSizeFilter") or {})
    try:
        step = float(lot.get("qtyStep") or 0)
        min_qty = float(lot.get("minOrderQty") or 0)
    except:
        step = 0.0; min_qty = 0.0
    q = qty
    if step > 0: q = _round_step(q, step)
    if min_qty > 0 and q < min_qty:
        q = _round_step(min_qty, step) if step > 0 else min_qty
    q = max(q, 0.0)
    return float(f"{q:.10f}")

def normalize_price(symbol_info: dict, price: float) -> float:
    pf = (symbol_info.get("priceFilter") or {})
    try:
        tick = float(pf.get("tickSize") or 0)
    except:
        tick = 0.0
    p = price
    if tick > 0: p = _round_step(p, tick)
    return float(f"{p:.10f}")

def calc_vol_pct(series: List[float], px: float) -> float:
    tail = series[-48:] if len(series) >= 48 else series
    if len(tail) < 2 or px <= 0: return 1.0
    mean = sum(tail)/len(tail)
    var = sum((x-mean)**2 for x in tail)/len(tail)
    return (math.sqrt(var)/px)*100.0

def choose_auto_leverage(symbol_info: dict, ch24_abs: float, vol_pct: float) -> int:
    levf = (symbol_info.get("leverageFilter") or {})
    try: max_lev = int(float(levf.get("maxLeverage") or 1))
    except: max_lev = 1
    if vol_pct < 1.5 and ch24_abs < 2: lev = 5
    elif vol_pct < 3.0 and ch24_abs < 5: lev = 3
    else: lev = 2
    return max(1, min(max_lev, lev))

# ============ PRIVATE (sign & post) ============
def sign_v5(params: Dict[str, str]) -> Dict[str, str]:
    ts = str(int(time.time()*1000))
    params["api_key"] = BYBIT_KEY
    params["timestamp"] = ts
    params["recv_window"] = "5000"
    qs = "&".join([f"{k}={params[k]}" for k in sorted(params.keys())])
    sig = hmac.new(BYBIT_SEC.encode(), qs.encode(), hashlib.sha256).hexdigest()
    params["sign"] = sig
    return params

async def private_post(session: aiohttp.ClientSession, path: str, params: Dict[str, str]) -> dict:
    url = f"{BYBIT_URL}{path}"
    p = sign_v5(params.copy())
    async with session.post(url, data=p, timeout=25, **_proxy_kwargs()) as r:
        txt = await r.text()
        try:
            r.raise_for_status()
            return json.loads(txt)
        except:
            raise RuntimeError(f"HTTP {r.status}: {txt[:400]}")

async def private_get(session: aiohttp.ClientSession, path: str, params: Dict[str, str]) -> dict:
    url = f"{BYBIT_URL}{path}"
    p = sign_v5(params.copy())
    async with session.get(url, params=p, timeout=25, **_proxy_kwargs()) as r:
        txt = await r.text()
        try:
            r.raise_for_status()
            return json.loads(txt)
        except:
            raise RuntimeError(f"HTTP {r.status}: {txt[:400]}")

# ============ Positions helper ============
async def get_open_positions(session: aiohttp.ClientSession) -> List[dict]:
    data = await private_get(session, "/v5/position/list", {"category":"linear"})
    return ((data.get("result") or {}).get("list")) or []

def symbol_in_positions(positions: List[dict], symbol: str) -> bool:
    for p in positions:
        if str(p.get("symbol")) == symbol:
            try: sz = abs(float(p.get("size") or 0))
            except: sz = 0
            if sz > 0: return True
    return False

async def ensure_leverage(session: aiohttp.ClientSession, symbol: str, lev: int):
    try:
        await private_post(session, "/v5/position/set-leverage", {
            "category":"linear", "symbol":symbol, "buyLeverage":str(lev), "sellLeverage":str(lev)
        })
    except Exception as e:
        log.warning("set-leverage fail %s: %s", symbol, e)

# ======= РИНКОВЕ ЗАМОВЛЕННЯ БЕЗ SL/TP =======
async def place_order_market(
    session: aiohttp.ClientSession,
    symbol: str,
    side: str,                # "Buy" або "Sell"
    size_usdt: float,
    px: float,
    k15_for_vol: Optional[List[float]] = None,
    ch24_abs: float = 0.0
):
    info = await get_instrument_info(session, symbol)
    if not info: raise RuntimeError(f"instrument info not found for {symbol}")

    raw_qty = size_usdt / max(px, 1e-12)
    qty = normalize_qty(info, raw_qty)
    try: min_qty = float((info.get("lotSizeFilter") or {}).get("minOrderQty") or 0)
    except: min_qty = 0.0
    if qty <= 0 or (min_qty > 0 and qty < min_qty):
        raise RuntimeError(f"qty too small for {symbol}: raw={raw_qty:.12f}, norm={qty}, min={min_qty}")

    lev_to_use = LEVERAGE
    if AUTO_LEVERAGE:
        vol_pct = calc_vol_pct(k15_for_vol or [], px)
        lev_to_use = choose_auto_leverage(info, abs(ch24_abs), vol_pct)
    await ensure_leverage(session, symbol, lev_to_use)

    params = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": f"{qty:.10f}",
        "timeInForce": "GoodTillCancel",
    }

    log.info("ORDER %s | side=%s lev=%s | px=%.10f qty=%s", symbol, side, lev_to_use, px, params["qty"])

    data = await private_post(session, "/v5/order/create", params)
    if str(data.get("retCode")) != "0":
        log.error("Order rejected for %s: %s", symbol, json.dumps(data))
        raise RuntimeError(f"Bybit error {data.get('retCode')}: {data.get('retMsg')} | resp={data}")
    return data, lev_to_use

# ============ Signals + Trade ============
async def build_signals_and_trade(chat_id: int) -> str:
    st = STATE.setdefault(chat_id, {"sl": SL_PCT, "tp": TP_PCT, "top_n": TOP_N})
    sl_pct = float(st.get("sl", SL_PCT))
    tp_pct = float(st.get("tp", TP_PCT))
    top_n  = int(st.get("top_n", TOP_N))
    report_lines = []

    async with aiohttp.ClientSession() as s:
        try:
            tickers = await bybit_top_symbols(s, 15)
        except Exception as e:
            return f"⚠️ Ринок недоступний: {e}"

        open_pos = []
        if TRADE_ENABLED and BYBIT_KEY and BYBIT_SEC:
            try: open_pos = await get_open_positions(s)
            except Exception as e: log.warning("get_open_positions fail: %s", e)

        scored: List[Tuple[float, str, str, float, str, float, float, float, List[float]]] = []

        for t in (tickers or []):
            if not isinstance(t, dict): continue
            sym = t.get("symbol","")
            try:
                px  = float(t.get("lastPrice") or 0.0)
                ch24 = float(t.get("price24hPcnt") or 0.0) * 100.0
            except:
                px, ch24 = 0.0, 0.0
            if not sym or px <= 0: continue

            try:
                k15 = await bybit_klines(s, sym, "15", 300); await asyncio.sleep(0.35)
                k30 = await bybit_klines(s, sym, "30", 300); await asyncio.sleep(0.35)
                k60 = await bybit_klines(s, sym, "60", 300)
            except:
                continue
            if not (k15 and k30 and k60): continue

            v15 = votes_from_series(k15)
            v30 = votes_from_series(k30)
            v60 = votes_from_series(k60)
            direction = decide_direction(v15["vote"], v30["vote"], v60["vote"])
            if not direction: continue

            if sl_pct <= 0 or tp_pct <= 0:
                base_sl, base_tp = auto_sl_tp_by_vol(k15, px)
            else:
                base_sl, base_tp = sl_pct, tp_pct

            score = v15["vote"] + v30["vote"] + v60["vote"]
            if v60["ema_trend"] == 1 and direction == "LONG": score += 1
            if v60["ema_trend"] == -1 and direction == "SHORT": score += 1
            score += min(2.0, abs(ch24)/10.0)

            def mark(v):
                r = v["rsi"]; rtxt = f"{r:.0f}" if isinstance(r,(int,float)) else "-"
                m = v["macd"]; sgn = v["sig"]
                mtxt = "↑" if (m is not None and sgn is not None and m > sgn) else ("↓" if (m is not None and sgn is not None and m < sgn) else "·")
                et = v["ema_trend"]; etxt = "↑" if et==1 else ("↓" if et==-1 else "·")
                return f"RSI:{rtxt} MACD:{mtxt} EMA:{etxt}"

            note = f"15m[{mark(v15)}] | 30m[{mark(v30)}] | 1h[{mark(v60)}]"
            scored.append((float(score), sym, direction, px, note, float(base_sl), float(base_tp), abs(ch24), k15))
            await asyncio.sleep(0.75)

        if not scored:
            return "⚠️ Узгоджених сильних сигналів немає."

        scored.sort(key=lambda x: x[0], reverse=True)
        picks = scored[:max(1, min(2, top_n))]

        opened = 0
        if TRADE_ENABLED and BYBIT_KEY and BYBIT_SEC:
            open_count = sum(1 for p in open_pos if float(p.get("size") or 0) > 0)
            can_open = max(0, MAX_OPEN_POS - open_count)

            for sc, sym, direction, px, note, bsl, btp, ch24_abs, k15 in picks:
                if opened >= can_open: break
                if symbol_in_positions(open_pos, sym):
                    report_lines.append(f"• {sym}: {direction} (пропущено — вже відкрита позиція)")
                    continue
                side = "Buy" if direction=="LONG" else "Sell"
                try:
                    resp, used_lev = await place_order_market(
                        s, sym, side, SIZE_USDT, px,
                        k15_for_vol=k15, ch24_abs=ch24_abs
                    )
                    opened += 1
                    oid = ((resp.get("result") or {}).get("orderId"))
                    if direction == "LONG":
                        slp = px*(1-bsl/100.0); tpp = px*(1+btp/100.0)
                    else:
                        slp = px*(1+bsl/100.0); tpp = px*(1-btp/100.0)
                    report_lines.append(
                        f"✅ ВІДКРИТО {sym} {direction} @ {px:.6f} · lev={used_lev}"
                        + (f" · id:{oid}" if oid else "")
                        + f"\n   Рекомендація: SL `{slp:.6f}` · TP `{tpp:.6f}` (без авто-SL/TP)\n   {note}"
                    )
                except Exception as e:
                    report_lines.append(f"❌ {sym} {direction}: {e}")

        if not report_lines:
            body = []
            for sc, sym, direction, px, note, bsl, btp, _c, _k in picks:
                if direction == "LONG":
                    slp = px*(1-bsl/100.0); tpp = px*(1+btp/100.0)
                else:
                    slp = px*(1+bsl/100.0); tpp = px*(1-btp/100.0)
                body.append(
                    f"• {sym}: *{direction}* @ {px:.6f}\n"
                    f"  SL: `{slp:.6f}` · TP: `{tpp:.6f}` · LEV: {LEVERAGE}{' (AUTO)' if AUTO_LEVERAGE else ''}\n"
                    f"  {note}"
                )
            return "📈 *Сильні сигнали:*\n\n" + "\n\n".join(body)
        else:
            header = "🤖 *Автотрейд виконано*:\n\n" if TRADE_ENABLED else "📈 Сигнали:\n\n"
            return header + "\n".join(report_lines)

# ============ Commands ============
async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    chat_id = u.effective_chat.id
    STATE.setdefault(chat_id, {"auto_on": False, "every": DEFAULT_AUTO_MIN, "sl": SL_PCT, "tp": TP_PCT, "top_n": TOP_N})
    await u.message.reply_text(
        "👋 Готовий. Головні команди:\n"
        f"• /signals\n• /auto_on {DEFAULT_AUTO_MIN} · /auto_off\n• /status\n\n"
        "Додатково (введи вручну): /trade_on /trade_off /set_size 5 /set_lev 3 /set_risk 3 5 /set_top 2",
        reply_markup=KB
    )

async def signals_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    txt = await build_signals_and_trade(u.effective_chat.id)
    for ch in split_long(txt):
        await u.message.reply_text(ch, parse_mode=ParseMode.MARKDOWN)

async def auto_on_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    chat_id = u.effective_chat.id
    st = STATE.setdefault(chat_id, {"auto_on": False, "every": DEFAULT_AUTO_MIN})
    minutes = st.get("every", DEFAULT_AUTO_MIN)
    if c.args:
        try: minutes = max(5, min(120, int(c.args[0])))
        except: pass
    st["auto_on"] = True; st["every"] = minutes
    name = f"auto_{chat_id}"
    for j in c.application.job_queue.get_jobs_by_name(name):
        j.schedule_removal()
    c.application.job_queue.run_repeating(
        auto_push_job, interval=minutes*60, first=5, name=name, data={"chat_id": chat_id}
    )
    await u.message.reply_text(f"✅ Автоскан/автотрейд: кожні {minutes} хв.", reply_markup=KB)

async def auto_off_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    chat_id = u.effective_chat.id
    st = STATE.setdefault(chat_id, {"auto_on": False, "every": DEFAULT_AUTO_MIN})
    st["auto_on"] = False
    name = f"auto_{chat_id}"
    for j in c.application.job_queue.get_jobs_by_name(name):
        j.schedule_removal()
    await u.message.reply_text("⏸ Авто вимкнено.", reply_markup=KB)

async def status_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(
        u.effective_chat.id,
        {"auto_on": False, "every": DEFAULT_AUTO_MIN, "sl": SL_PCT, "tp": TP_PCT, "top_n": TOP_N}
    )
    sl    = float((st.get("sl", SL_PCT) or SL_PCT) or 0.0)
    tp    = float((st.get("tp", TP_PCT) or TP_PCT) or 0.0)
    every = int((st.get("every", DEFAULT_AUTO_MIN) or DEFAULT_AUTO_MIN) or 0)
    topn  = int((st.get("top_n", TOP_N) or TOP_N) or 1)
    auto  = " (AUTO)" if AUTO_LEVERAGE else ""
    text = (
        f"Статус: {'ON' if st.get('auto_on') else 'OFF'} · кожні {every} хв\n"
        f"TRADE: {'ON' if TRADE_ENABLED else 'OFF'} · SIZE={SIZE_USDT:.2f} USDT · LEV={LEVERAGE}{auto}\n"
        f"SL={sl:.2f}% · TP={tp:.2f}% · TOP_N={topn}\n"
        f"HEDGE_MODE: {'ON' if HEDGE_MODE else 'OFF'} · UTC: {utc_now()}"
    )
    await u.message.reply_text(text)

async def trade_on_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    global TRADE_ENABLED
    TRADE_ENABLED = True
    await u.message.reply_text("✅ Автотрейд УВІМКНЕНО")

async def trade_off_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    global TRADE_ENABLED
    TRADE_ENABLED = False
    await u.message.reply_text("⏸ Автотрейд ВИМКНЕНО")

async def set_size_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    global SIZE_USDT
    try:
        v = float(c.args[0]); assert v > 0
        SIZE_USDT = v
        await u.message.reply_text(f"OK. SIZE_USDT={SIZE_USDT:.2f}")
    except:
        await u.message.reply_text("Формат: /set_size 5")

async def set_lev_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    global LEVERAGE
    try:
        v = int(c.args[0]); assert v >= 1
        LEVERAGE = v
        await u.message.reply_text(f"OK. LEVERAGE={LEVERAGE}")
    except:
        await u.message.reply_text("Формат: /set_lev 3")

async def set_risk_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, {"sl": SL_PCT, "tp": TP_PCT})
    try:
        sl = float(c.args[0]); tp = float(c.args[1]); assert sl >= 0 and tp >= 0
        st["sl"], st["tp"] = sl, tp
        await u.message.reply_text(f"OK. SL={sl:.2f}% · TP={tp:.2f}% (0 0 = авто)")
    except:
        await u.message.reply_text("Формат: /set_risk 3 5  (0 0 = авто)")

async def set_top_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, {"top_n": TOP_N})
    try:
        n = int(c.args[0]); assert 1 <= n <= 2
        st["top_n"] = n
        await u.message.reply_text(f"OK. TOP_N={n}")
    except:
        await u.message.reply_text("Формат: /set_top 1..2")

# ============ Job ============
async def auto_push_job(ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = ctx.job.data["chat_id"]
    try:
        txt = await build_signals_and_trade(chat_id)
        for ch in split_long(txt):
            await ctx.bot.send_message(chat_id=chat_id, text=ch, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        log.error("auto job error: %s", e)

# ============ Main ============
def main():
    if not TG_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN"); return
    if not BYBIT_KEY or not BYBIT_SEC:
        log.warning("BYBIT_API_KEY/SECRET not set — трейд не спрацює (лише сигнали).")

    print("Bot running: Bybit autotrade | TF=15/30/60 | top15 | max 2 pos | "
          f"AUTO_LEVERAGE={'ON' if AUTO_LEVERAGE else 'OFF'} | HEDGE_MODE={'ON' if HEDGE_MODE else 'OFF'}")
    app = Application.builder().token(TG_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))
    app.add_handler(CommandHandler("auto_on", auto_on_cmd))
    app.add_handler(CommandHandler("auto_off", auto_off_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("trade_on", trade_on_cmd))
    app.add_handler(CommandHandler("trade_off", trade_off_cmd))
    app.add_handler(CommandHandler("set_size", set_size_cmd))
    app.add_handler(CommandHandler("set_lev", set_lev_cmd))
    app.add_handler(CommandHandler("set_risk", set_risk_cmd))
    app.add_handler(CommandHandler("set_top", set_top_cmd))

    app.run_polling()

if __name__ == "__main__":
    main()
