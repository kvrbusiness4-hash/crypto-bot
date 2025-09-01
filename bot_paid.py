# -*- coding: utf-8 -*-
# bot_signals_only.py

import os, math, asyncio, aiohttp, logging
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from decimal import Decimal, getcontext
from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

getcontext().prec = 28

# ================== ENV / DEFAULTS ==================
TG_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BYBIT_URL   = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").rstrip("/")
BYBIT_PROXY = os.getenv("BYBIT_PROXY", "").strip()

DEFAULT_AUTO_MIN = int(os.getenv("DEFAULT_AUTO_MIN", "15"))
TOP_N_DEFAULT   = int(os.getenv("TOP_N", "3"))

# Рекомендоване плече: авто/ручне (використовується лише як порада в сигналі)
LEVERAGE_DEFAULT      = int(os.getenv("LEVERAGE", "3"))
AUTO_LEVERAGE_DEFAULT = os.getenv("AUTO_LEVERAGE", "ON").upper() == "ON"

# Ручні SL/TP (%), якщо авто ризик вимкнено
SL_PCT_DEFAULT = float(os.getenv("SL_PCT", "3"))
TP_PCT_DEFAULT = float(os.getenv("TP_PCT", "5"))

# Авто ризик через ATR
AUTO_RISK_DEFAULT = os.getenv("AUTO_RISK", "ON").upper() == "ON"
ATR_LEN_DEFAULT   = int(os.getenv("ATR_LEN", "14"))
SL_K_DEFAULT      = float(os.getenv("SL_K", "1.5"))   # SL = SL_K * ATR%
TP_K_DEFAULT      = float(os.getenv("TP_K", "2.5"))   # TP = TP_K * ATR%

# Фільтр шуму і сила тренду
NOISE_DEFAULT       = float(os.getenv("NOISE_PCT", "1.0"))   # мінімальний % для SL/TP
TREND_WEIGHT_DEFAULT= int(os.getenv("TREND_WEIGHT", "2"))    # 2..3

# Пороги індикаторів
RSI_LOW_DEFAULT     = int(os.getenv("RSI_LOW", "30"))
RSI_HIGH_DEFAULT    = int(os.getenv("RSI_HIGH", "70"))

# Логи
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("signals-bot")

# ================== UI (мінімальна клавіатура) ==================
KB = ReplyKeyboardMarkup(
    [
        ["/signals", "/status"],
        ["/set_auto_risk on", "/set_auto_risk off"],
        ["/set_atr 14", "/set_k 1.5 2.5"],
        ["/set_noise 1.0", "/set_trend 2"],
        ["/set_rsi 30 70", "/set_top 3"],
        ["/lev_auto on", "/lev_auto off"],
    ],
    resize_keyboard=True
)

STATE: Dict[int, Dict[str, int | bool | float]] = {}

# ================== Helpers ==================
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

def ensure_state(chat_id: int) -> Dict[str, float | int | bool]:
    return STATE.setdefault(chat_id, {
        "every": DEFAULT_AUTO_MIN,
        "top_n": TOP_N_DEFAULT,
        # ризик
        "auto_risk": AUTO_RISK_DEFAULT,
        "atr_len": ATR_LEN_DEFAULT,
        "sl_k": SL_K_DEFAULT,
        "tp_k": TP_K_DEFAULT,
        "sl": SL_PCT_DEFAULT,
        "tp": TP_PCT_DEFAULT,
        "noise": NOISE_DEFAULT,
        # індикатори
        "rsi_low": RSI_LOW_DEFAULT,
        "rsi_high": RSI_HIGH_DEFAULT,
        "trend_weight": TREND_WEIGHT_DEFAULT,   # 2..3
        # плече (рекомендація)
        "lev_auto": AUTO_LEVERAGE_DEFAULT,
        "lev_fixed": LEVERAGE_DEFAULT,
    })

# ================== Indicators ==================
def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1: return series[:]
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

def votes_from_series(series: List[float], rsi_low:int, rsi_high:int) -> Dict[str, int | float]:
    out = {"vote": 0, "rsi": None, "ema_trend": 0, "macd": None, "sig": None}
    if len(series) < 60: return out
    rr = rsi(series, 14); m, s = macd(series)
    e50 = ema(series, 50)
    e200 = ema(series, 200) if len(series) >= 200 else ema(series, max(100, len(series)//2))
    if rr:
        out["rsi"] = rr[-1]
        if rr[-1] <= rsi_low: out["vote"] += 1
        if rr[-1] >= rsi_high: out["vote"] -= 1
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

# ================== HTTP (public) ==================
BYBIT_PUBLIC = "https://api.bybit.com"

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

async def bybit_top_symbols(session: aiohttp.ClientSession, top:int=25) -> List[dict]:
    data = await http_json(session, f"{BYBIT_PUBLIC}/v5/market/tickers", {"category":"linear"})
    lst = ((data.get("result") or {}).get("list")) or []
    def _volume(x):
        try: return float(x.get("turnover24h") or 0)
        except: return 0.0
    lst.sort(key=_volume, reverse=True)
    return [x for x in lst if str(x.get("symbol","")).endswith("USDT")][:top]

async def bybit_klines(session: aiohttp.ClientSession, symbol: str, interval: str, limit: int = 300) -> List[float]:
    data = await http_json(session, f"{BYBIT_PUBLIC}/v5/market/kline", {
        "category":"linear","symbol":symbol,"interval":interval,"limit":str(limit)
    })
    rows = list(reversed(((data.get("result") or {}).get("list")) or []))
    closes = []
    for r in rows:
        try: closes.append(float(r[4]))
        except: pass
    return closes

async def bybit_klines_ohlc(session: aiohttp.ClientSession, symbol: str, interval: str, limit: int = 300):
    data = await http_json(session, f"{BYBIT_PUBLIC}/v5/market/kline", {
        "category":"linear","symbol":symbol,"interval":interval,"limit":str(limit)
    })
    rows = list(reversed(((data.get("result") or {}).get("list")) or []))
    out = []
    for r in rows:
        try:
            o = float(r[1]); h = float(r[2]); l = float(r[3]); c = float(r[4])
            out.append((o,h,l,c))
        except:
            pass
    return out  # [(o,h,l,c), ...]

# ================== ATR та ризик ==================
def atr_from_ohlc(ohlc: List[Tuple[float,float,float,float]], length: int = 14) -> List[float]:
    if len(ohlc) < length + 1:
        return []
    trs = []
    prev_close = ohlc[0][3]
    for _,h,l,c in ohlc[1:]:
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    atr = []
    for i in range(len(trs)):
        if i+1 < length:
            atr.append(None)
        else:
            window = trs[i+1-length:i+1]
            atr.append(sum(window)/length)
    return atr

def calc_vol_pct(series: List[float], px: float) -> float:
    tail = series[-48:] if len(series) >= 48 else series
    if len(tail) < 2 or px <= 0: return 1.0
    mean = sum(tail)/len(tail)
    var = sum((x-mean)**2 for x in tail)/len(tail)
    return (math.sqrt(var)/px)*100.0

def choose_auto_leverage(max_lev:int, ch24_abs: float, vol_pct: float) -> int:
    # проста евристика
    if vol_pct < 1.5 and ch24_abs < 2: lev = 5
    elif vol_pct < 3.0 and ch24_abs < 5: lev = 3
    else: lev = 2
    return max(1, min(max_lev, lev))

def compute_sl_tp_pct_auto(atr_abs: Optional[float], last_price: float, sl_k: float, tp_k: float, noise_min: float) -> Tuple[float,float]:
    if last_price <= 0 or atr_abs is None:
        return max(noise_min, SL_PCT_DEFAULT), max(noise_min*2, TP_PCT_DEFAULT)
    atr_pct = (atr_abs / last_price) * 100.0
    sl_pct = max(noise_min, atr_pct * sl_k)
    tp_pct = max(noise_min, atr_pct * tp_k)
    return sl_pct, tp_pct

# ================== Сигнали ==================
async def build_signals(chat_id: int) -> str:
    st = ensure_state(chat_id)
    top_n   = int(st.get("top_n", TOP_N_DEFAULT))
    rsi_low = int(st.get("rsi_low", RSI_LOW_DEFAULT))
    rsi_high= int(st.get("rsi_high", RSI_HIGH_DEFAULT))
    noise_min = float(st.get("noise", NOISE_DEFAULT))
    auto_risk = bool(st.get("auto_risk", AUTO_RISK_DEFAULT))
    atr_len   = int(st.get("atr_len", ATR_LEN_DEFAULT))
    sl_k      = float(st.get("sl_k", SL_K_DEFAULT))
    tp_k      = float(st.get("tp_k", TP_K_DEFAULT))
    trend_w   = int(st.get("trend_weight", TREND_WEIGHT_DEFAULT))
    lev_auto  = bool(st.get("lev_auto", AUTO_LEVERAGE_DEFAULT))
    lev_fixed = int(st.get("lev_fixed", LEVERAGE_DEFAULT))

    async with aiohttp.ClientSession() as s:
        try:
            tickers = await bybit_top_symbols(s, 25)
        except Exception as e:
            return f"⚠️ Ринок недоступний: {e}"

        scored: List[Tuple[float, str, str, float, str, float, float]] = []
        # (score, symbol, direction, px, note, ch24_abs, vol_pct)

        for t in tickers:
            sym = t.get("symbol","")
            try:
                px   = float(t.get("lastPrice") or 0.0)
                ch24 = float(t.get("price24hPcnt") or 0.0) * 100.0
                max_lev = int(float((t.get("fundingRate") and 5) or 5))  # заглушка; немає в тикері — тримаємо <=5
            except:
                px, ch24, max_lev = 0.0, 0.0, 5
            if px <= 0:
                continue

            try:
                k15 = await bybit_klines(s, sym, "15", 300)
                k30 = await bybit_klines(s, sym, "30", 300)
                k60 = await bybit_klines(s, sym, "60", 300)
                k15_ohlc = await bybit_klines_ohlc(s, sym, "15", 300)
            except:
                continue
            if not (k15 and k30 and k60 and k15_ohlc):
                continue

            v15 = votes_from_series(k15, rsi_low, rsi_high)
            v30 = votes_from_series(k30, rsi_low, rsi_high)
            v60 = votes_from_series(k60, rsi_low, rsi_high)
            direction = decide_direction(v15["vote"], v30["vote"], v60["vote"])
            if not direction:
                continue

            # ATR
            atr15 = atr_from_ohlc(k15_ohlc, atr_len)
            atr_abs = atr15[-1] if atr15 and atr15[-1] is not None else None
            vol_pct = calc_vol_pct(k15, px)

            # Рекомендоване плече
            lev_rec = choose_auto_leverage(5, abs(ch24), vol_pct) if lev_auto else lev_fixed

            # Оцінка (з урахуванням сили тренду)
            score = v15["vote"] + v30["vote"] + v60["vote"]
            trend_bonus = (trend_w - 1)
            if v60["ema_trend"] == 1 and direction == "LONG": score += trend_bonus
            if v60["ema_trend"] == -1 and direction == "SHORT": score += trend_bonus
            score += min(2.0, abs(ch24)/10.0)

            # SL/TP %
            if auto_risk:
                sl_pct, tp_pct = compute_sl_tp_pct_auto(atr_abs, px, sl_k, tp_k, noise_min)
                risk_tag = f"AUTO(ATR{atr_len}, k={sl_k:.2f}/{tp_k:.2f})"
            else:
                sl_pct = max(noise_min, float(st.get("sl", SL_PCT_DEFAULT)))
                tp_pct = max(noise_min, float(st.get("tp", TP_PCT_DEFAULT)))
                risk_tag = "MANUAL"

            if direction == "LONG":
                sl_price = px * (1 - sl_pct/100.0)
                tp_price = px * (1 + tp_pct/100.0)
            else:
                sl_price = px * (1 + sl_pct/100.0)
                tp_price = px * (1 - tp_pct/100.0)

            def mark(v):
                r = v["rsi"]; rtxt = f"{r:.0f}" if isinstance(r,(int,float)) else "-"
                m = v["macd"]; sgn = v["sig"]
                mtxt = "↑" if (m is not None and sgn is not None and m > sgn) else ("↓" if (m is not None and sgn is not None and m < sgn) else "·")
                et = v["ema_trend"]; etxt = "↑" if et==1 else ("↓" if et==-1 else "·")
                return f"RSI:{rtxt} MACD:{mtxt} EMA:{etxt}"

            note = f"15m[{mark(v15)}] | 30m[{mark(v30)}] | 1h[{mark(v60)}]"
            lines = (
                f"• {sym}: *{direction}* @ {px:.6f}\n"
                f"  SL: `{sl_price:.6f}` ({sl_pct:.2f}%) · TP: `{tp_price:.6f}` ({tp_pct:.2f}%) · {risk_tag}\n"
                f"  LEV (рек.): {lev_rec} · vol≈{vol_pct:.2f}% · 24hΔ≈{abs(ch24):.2f}%\n"
                f"  {note}"
            )
            scored.append((float(score), sym, direction, px, lines, abs(ch24), vol_pct))
            await asyncio.sleep(0.25)

        if not scored:
            return "⚠️ Узгоджених сильних сигналів немає."

        scored.sort(key=lambda x: x[0], reverse=True)
        picks = scored[:max(1, min(3, top_n))]

        body = [x[4] for x in picks]
        return "📈 *Сильні сигнали (аналіз)*\n\n" + "\n\n".join(body) + f"\n\nUTC: {utc_now()}"

# ================== Команди ==================
async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    ensure_state(u.effective_chat.id)
    await u.message.reply_text("👋 Готовий. Бот видає *аналізовані сигнали* без відкриття ордерів.\nНижче — корисні команди.", reply_markup=KB)

async def signals_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    txt = await build_signals(u.effective_chat.id)
    for ch in split_long(txt):
        await u.message.reply_text(ch, parse_mode=ParseMode.MARKDOWN)

async def status_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    text = (
        "🛠 *Налаштування*\n"
        f"TOP_N: {int(st['top_n'])} (1..3)\n"
        f"NOISE(min %): {st['noise']:.2f}\n"
        f"RSI_LOW/HIGH: {int(st['rsi_low'])}/{int(st['rsi_high'])}\n"
        f"TREND_WEIGHT: {int(st['trend_weight'])} (2..3)\n"
        f"AUTO_RISK: {'ON' if st['auto_risk'] else 'OFF'} · ATR={int(st['atr_len'])} · k={st['sl_k']:.2f}/{st['tp_k']:.2f}\n"
        f"LEV_AUTO: {'ON' if st['lev_auto'] else 'OFF'} · LEV_FIXED={int(st['lev_fixed'])}\n"
        f"UTC: {utc_now()}"
    )
    await u.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

# ---- setters
async def set_top_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    try:
        n = int(c.args[0]); assert 1 <= n <= 3
        st["top_n"] = n
        await u.message.reply_text(f"OK. TOP_N={n}")
    except:
        await u.message.reply_text("Формат: /set_top 1..3")

async def set_noise_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    try:
        v = float(c.args[0]); assert 0 <= v <= 5
        st["noise"] = v
        await u.message.reply_text(f"OK. NOISE(min %)={v:.2f}")
    except:
        await u.message.reply_text("Формат: /set_noise 1.0  (0..5)")

async def set_trend_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    try:
        v = int(c.args[0]); assert v in (2,3)
        st["trend_weight"] = v
        await u.message.reply_text(f"OK. TREND_WEIGHT={v}")
    except:
        await u.message.reply_text("Формат: /set_trend 2|3")

async def set_rsi_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    try:
        low = int(c.args[0]); high = int(c.args[1]); assert 5 <= low < high <= 95
        st["rsi_low"], st["rsi_high"] = low, high
        await u.message.reply_text(f"OK. RSI_LOW={low} · RSI_HIGH={high}")
    except:
        await u.message.reply_text("Формат: /set_rsi 30 70  (межі 5..95)")

async def set_auto_risk_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    arg = (c.args[0] if c.args else "").lower()
    if arg in ("on","off"):
        st["auto_risk"] = (arg == "on")
        await u.message.reply_text(f"OK. Авто SL/TP: {'ON' if st['auto_risk'] else 'OFF'}")
    else:
        await u.message.reply_text("Формат: /set_auto_risk on|off")

async def set_atr_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    try:
        n = int(c.args[0]); assert 5 <= n <= 50
        st["atr_len"] = n
        await u.message.reply_text(f"OK. ATR length = {n}")
    except:
        await u.message.reply_text("Формат: /set_atr 14  (5..50)")

async def set_k_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    try:
        slk = float(c.args[0]); tpk = float(c.args[1]); assert 0.5 <= slk <= 5 and 0.5 <= tpk <= 10
        st["sl_k"] = slk; st["tp_k"] = tpk
        await u.message.reply_text(f"OK. Коефіцієнти: SL_K={slk} · TP_K={tpk}")
    except:
        await u.message.reply_text("Формат: /set_k 1.5 2.5  (діапазони: SL_K 0.5..5, TP_K 0.5..10)")

async def lev_auto_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    arg = (c.args[0] if c.args else "").lower()
    if arg in ("on","off"):
        st["lev_auto"] = (arg == "on")
        await u.message.reply_text(f"OK. LEV_AUTO: {'ON' if st['lev_auto'] else 'OFF'}")
    else:
        await u.message.reply_text("Формат: /lev_auto on|off")

async def set_lev_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    try:
        v = int(c.args[0]); assert v >= 1
        st["lev_fixed"] = v
        await u.message.reply_text(f"OK. LEV_FIXED={v}")
    except:
        await u.message.reply_text("Формат: /set_lev 3")

# ================== Main ==================
def main():
    if not TG_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN"); return

    print("Signals bot running: Bybit | TF=15/30/60 | top up to 3 | NO AUTOTRADE")
    app = Application.builder().token(TG_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    app.add_handler(CommandHandler("set_top", set_top_cmd))
    app.add_handler(CommandHandler("set_noise", set_noise_cmd))
    app.add_handler(CommandHandler("set_trend", set_trend_cmd))
    app.add_handler(CommandHandler("set_rsi", set_rsi_cmd))

    app.add_handler(CommandHandler("set_auto_risk", set_auto_risk_cmd))
    app.add_handler(CommandHandler("set_atr", set_atr_cmd))
    app.add_handler(CommandHandler("set_k", set_k_cmd))

    app.add_handler(CommandHandler("lev_auto", lev_auto_cmd))
    app.add_handler(CommandHandler("set_lev", set_lev_cmd))

    app.run_polling()

if __name__ == "__main__":
    main()
