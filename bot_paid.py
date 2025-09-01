# -*- coding: utf-8 -*-
# bot_signals.py  — тільки сигнали (без автотрейду)

import os, math, asyncio, aiohttp, logging
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ============ ENV ============
TG_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BYBIT_PUBLIC = "https://api.bybit.com"
BYBIT_PROXY  = os.getenv("BYBIT_PROXY", "").strip()  # опціонально

# дефолтні параметри бота
DEF_TOP_N         = 3        # максимум 3 монети
DEF_STRENGTH_MIN  = 2        # 2 або 3
DEF_WEIGHTS       = {"rsi": 1, "macd": 1, "ema": 1}  # ваги індикаторів
DEF_SL            = 3.0
DEF_TP            = 5.0
DEF_LEV           = 3
DEF_MIN_TURNOVER  = 5_000_000     # фільтр шуму: мін. обіг за 24h (USDT)
DEF_MAX_VOL_PCT   = 12.0          # фільтр шуму: макс. волатильність (стд-відх. % за 48 свічок)

# Логи
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("signals-bot")

# ============ Клавіатура ============
def make_kb(st: dict) -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            ["/signals", "/status"],
            [f"/set_top {st.get('top_n', DEF_TOP_N)}", f"/set_strength {st.get('strength_min', DEF_STRENGTH_MIN)}"],
            [f"/set_weights {st['w']['rsi']} {st['w']['macd']} {st['w']['ema']}"],
            [f\"/set_risk {st['sl']:.0f} {st['tp']:.0f} {st['lev']}\", f\"/set_noise {int(st['min_turn'])} {st['max_vol']:.0f}\"],
        ],
        resize_keyboard=True
    )

# Памʼять користувача
STATE: Dict[int, Dict] = {}

# ============ Допоміжні ============
def utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

def _proxy_kwargs() -> dict:
    if BYBIT_PROXY.startswith(("http://", "https://")):
        return {"proxy": BYBIT_PROXY}
    return {}

def split_long(text: str, n: int = 3500) -> List[str]:
    out = []
    while len(text) > n:
        out.append(text[:n]); text = text[n:]
    out.append(text)
    return out

# ============ Індикатори ============
def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1: return series[:]
    k = 2 / (period + 1.0)
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

def series_vote(series: List[float], w: Dict[str,int]) -> Dict[str, float | int]:
    out = {"vote": 0.0, "rsi": None, "ema_trend": 0, "macd": None, "sig": None}
    if len(series) < 60: return out
    rr     = rsi(series, 14)
    m, s   = macd(series)
    e50    = ema(series, 50)
    e200   = ema(series, 200) if len(series) >= 200 else ema(series, max(100, len(series)//2))

    if rr:
        out["rsi"] = rr[-1]
        if rr[-1] <= 30: out["vote"] += 1 * w["rsi"]
        if rr[-1] >= 70: out["vote"] -= 1 * w["rsi"]

    if m and s:
        out["macd"], out["sig"] = m[-1], s[-1]
        if m[-1] > s[-1]: out["vote"] += 1 * w["macd"]
        if m[-1] < s[-1]: out["vote"] -= 1 * w["macd"]

    if e50 and e200:
        out["ema_trend"] = 1 if e50[-1] > e200[-1] else -1
        out["vote"] += (1 if e50[-1] > e200[-1] else -1) * w["ema"]

    return out

def decide_direction(v15:float, v30:float, v60:float, strength_min:int) -> Optional[str]:
    total = v15 + v30 + v60
    pos = sum(1 for v in [v15, v30, v60] if v > 0)
    neg = sum(1 for v in [v15, v30, v60] if v < 0)
    if total >= strength_min and pos >= 2:  return "LONG"
    if total <= -strength_min and neg >= 2: return "SHORT"
    return None

def calc_vol_pct(series: List[float], px: float) -> float:
    tail = series[-48:] if len(series) >= 48 else series
    if len(tail) < 2 or px <= 0: return 0.0
    mean = sum(tail)/len(tail)
    var = sum((x-mean)**2 for x in tail)/len(tail)
    return (math.sqrt(var)/px)*100.0

# ============ HTTP (public) ============
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

async def bybit_top_symbols(session: aiohttp.ClientSession, top:int=30) -> List[dict]:
    data = await http_json(session, f"{BYBIT_PUBLIC}/v5/market/tickers", {"category":"linear"})
    lst = ((data.get("result") or {}).get("list")) or []
    def _volume(x):
        try: return float(x.get("turnover24h") or 0.0)
        except: return 0.0
    lst.sort(key=_volume, reverse=True)
    return lst[:top]

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

# ============ Побудова сигналів ============
async def build_signals(chat_id: int) -> str:
    st = STATE.setdefault(chat_id, {
        "top_n": DEF_TOP_N,
        "strength_min": DEF_STRENGTH_MIN,
        "w": DEF_WEIGHTS.copy(),
        "sl": DEF_SL, "tp": DEF_TP, "lev": DEF_LEV,
        "min_turn": DEF_MIN_TURNOVER, "max_vol": DEF_MAX_VOL_PCT
    })

    w = st["w"]; top_n = int(st["top_n"])
    strength_min = int(st["strength_min"])

    async with aiohttp.ClientSession() as s:
        try:
            tickers = await bybit_top_symbols(s, 40)  # беремо більше, потім фільтруємо шум
        except Exception as e:
            return f"⚠️ Ринок недоступний: {e}"

        scored: List[Tuple[float, str, str, float, str]] = []  # score, symbol, dir, price, note

        for t in tickers:
            sym = t.get("symbol","")
            if not sym.endswith("USDT"):  # працюємо лише з USDT-перпами
                continue

            # фільтр шуму за оборотом
            try:
                turnover = float(t.get("turnover24h") or 0.0)
            except:
                turnover = 0.0
            if turnover < st["min_turn"]:
                continue

            try:
                px  = float(t.get("lastPrice") or 0.0)
                ch24 = float(t.get("price24hPcnt") or 0.0) * 100.0
            except:
                px, ch24 = 0.0, 0.0
            if px <= 0: 
                continue

            # таймфрейми
            try:
                k15 = await bybit_klines(s, sym, "15", 300); await asyncio.sleep(0.2)
                k30 = await bybit_klines(s, sym, "30", 300); await asyncio.sleep(0.2)
                k60 = await bybit_klines(s, sym, "60", 300)
            except:
                continue
            if not (k15 and k30 and k60): 
                continue

            # фільтр шуму за волатильністю
            vol_pct = calc_vol_pct(k15, px)
            if vol_pct > st["max_vol"]:
                continue

            v15 = series_vote(k15, w)
            v30 = series_vote(k30, w)
            v60 = series_vote(k60, w)
            direction = decide_direction(v15["vote"], v30["vote"], v60["vote"], strength_min)
            if not direction: 
                continue

            score = v15["vote"] + v30["vote"] + v60["vote"]
            # легкий бонус за збіг тренду старшого ТФ
            if v60["ema_trend"] == 1 and direction == "LONG":  score += 0.5
            if v60["ema_trend"] == -1 and direction == "SHORT": score += 0.5
            score += min(2.0, abs(ch24)/10.0)

            def mark(v):
                r = v["rsi"]; rtxt = f"{r:.0f}" if isinstance(r,(int,float)) else "-"
                m = v["macd"]; sgn = v["sig"]
                mtxt = "↑" if (m is not None and sgn is not None and m > sgn) else ("↓" if (m is not None and sgn is not None and m < sgn) else "·")
                et  = v["ema_trend"]; etxt = "↑" if et==1 else ("↓" if et==-1 else "·")
                return f"15mRSI:{rtxt} MACD:{mtxt} EMA:{etxt}"

            note = (
                f"{mark(v15)} | "
                f"30mRSI:{v30.get('rsi') and int(v30['rsi']) or '-'} MACD:{'↑' if (v30.get('macd') and v30.get('sig') and v30['macd']>v30['sig']) else ('↓' if (v30.get('macd') and v30.get('sig') and v30['macd']<v30['sig']) else '·')} EMA:{'↑' if v30['ema_trend']==1 else ('↓' if v30['ema_trend']==-1 else '·')} | "
                f"1hRSI:{v60.get('rsi') and int(v60['rsi']) or '-'} MACD:{'↑' if (v60.get('macd') and v60.get('sig') and v60['macd']>v60['sig']) else ('↓' if (v60.get('macd') and v60.get('sig') and v60['macd']<v60['sig']) else '·')} EMA:{'↑' if v60['ema_trend']==1 else ('↓' if v60['ema_trend']==-1 else '·')}"
            )

            scored.append((float(score), sym, direction, px, note))

            # невелика пауза, щоб не спамити API
            await asyncio.sleep(0.25)

        if not scored:
            return "⚠️ За поточними фільтрами/налаштуваннями сильних сигналів немає."

        # сортуємо за силою
        scored.sort(key=lambda x: x[0], reverse=True)
        picks = scored[:max(1, min(3, top_n))]

        # Формуємо відповідь
        lines = ["📈 *Сильні сигнали:*\n"]
        for score, sym, direction, px, note in picks:
            slp = px*(1 - st["sl"]/100.0) if direction=="LONG" else px*(1 + st["sl"]/100.0)
            tpp = px*(1 + st["tp"]/100.0) if direction=="LONG" else px*(1 - st["tp"]/100.0)
            lines.append(
                f"• {sym}: *{direction}* @ {px:.6f}  (score: `{score:.2f}`)\n"
                f"  SL: `{slp:.6f}` · TP: `{tpp:.6f}` · LEV: {st['lev']}\n"
                f"  {note}\n"
            )
        lines.append(f"UTC: {utc_now()}  |  налаштування: strength≥{strength_min}, weights(R/M/E)={w['rsi']}/{w['macd']}/{w['ema']}, "
                     f"noise(min turn={int(st['min_turn'])}, max vol={st['max_vol']:.0f}%)")
        return "\n".join(lines)

# ============ Команди ============
def ensure_state(chat_id: int) -> dict:
    return STATE.setdefault(chat_id, {
        "top_n": DEF_TOP_N,
        "strength_min": DEF_STRENGTH_MIN,
        "w": DEF_WEIGHTS.copy(),
        "sl": DEF_SL, "tp": DEF_TP, "lev": DEF_LEV,
        "min_turn": DEF_MIN_TURNOVER, "max_vol": DEF_MAX_VOL_PCT
    })

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    await u.message.reply_text("👋 Готовий. Працюю як аналітик-сигнальщик (без автотрейду).", reply_markup=make_kb(st))

async def status_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    w = st["w"]
    text = (
        f"🔧 Налаштування:\n"
        f"TOP_N: {st['top_n']}  •  STRENGTH_MIN: {st['strength_min']}\n"
        f"Weights (RSI/MACD/EMA): {w['rsi']}/{w['macd']}/{w['ema']}\n"
        f"Risk: SL={st['sl']:.2f}%  TP={st['tp']:.2f}%  LEV={st['lev']}\n"
        f"Noise filter: min_turnover={int(st['min_turn'])}  max_vol={st['max_vol']:.0f}%\n"
        f"UTC: {utc_now()}"
    )
    await u.message.reply_text(text, reply_markup=make_kb(st))

async def signals_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    txt = await build_signals(u.effective_chat.id)
    for chunk in split_long(txt):
        await u.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

# ---- setters
async def set_top_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    try:
        n = int(c.args[0]); assert 1 <= n <= 3
        st["top_n"] = n
        await u.message.reply_text(f"OK. TOP_N={n}", reply_markup=make_kb(st))
    except:
        await u.message.reply_text("Формат: /set_top 1..3")

async def set_strength_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    try:
        s = int(c.args[0]); assert s in (2,3)
        st["strength_min"] = s
        await u.message.reply_text(f"OK. STRENGTH_MIN={s}", reply_markup=make_kb(st))
    except:
        await u.message.reply_text("Формат: /set_strength 2|3")

async def set_weights_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    try:
        rsi_w  = int(c.args[0]); macd_w = int(c.args[1]); ema_w = int(c.args[2])
        # дозволимо діапазон -2..3 (нуль — вимкнути індикатор)
        for v in (rsi_w, macd_w, ema_w):
            assert -2 <= v <= 3
        st["w"] = {"rsi": rsi_w, "macd": macd_w, "ema": ema_w}
        await u.message.reply_text(f"OK. Weights set to RSI/MACD/EMA = {rsi_w}/{macd_w}/{ema_w}", reply_markup=make_kb(st))
    except:
        await u.message.reply_text("Формат: /set_weights rsi macd ema   (кожен у діапазоні -2..3)")

async def set_risk_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    try:
        sl = float(c.args[0]); tp = float(c.args[1]); lev = int(c.args[2])
        assert 0 <= sl <= 20 and 0 <= tp <= 30 and 1 <= lev <= 50
        st["sl"], st["tp"], st["lev"] = sl, tp, lev
        await u.message.reply_text(f"OK. SL={sl:.2f}% · TP={tp:.2f}% · LEV={lev}", reply_markup=make_kb(st))
    except:
        await u.message.reply_text("Формат: /set_risk 3 5 3   (SL% TP% LEV)")

async def set_noise_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = ensure_state(u.effective_chat.id)
    try:
        min_turn = float(c.args[0])          # у USDT
        max_vol  = float(c.args[1])          # у %
        assert min_turn >= 0 and 1 <= max_vol <= 50
        st["min_turn"], st["max_vol"] = min_turn, max_vol
        await u.message.reply_text(f"OK. Noise: min_turnover={int(min_turn)}, max_vol={max_vol:.0f}%", reply_markup=make_kb(st))
    except:
        await u.message.reply_text("Формат: /set_noise 5000000 12   (мін. оборот за 24h, макс. волатильність %)")

# ============ Main ============
def main():
    if not TG_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN"); return

    print("Signals-only bot running | TF=15/30/60 | top<=3 | RSI/MACD/EMA weights | noise filters")
    app = Application.builder().token(TG_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))

    app.add_handler(CommandHandler("set_top", set_top_cmd))
    app.add_handler(CommandHandler("set_strength", set_strength_cmd))
    app.add_handler(CommandHandler("set_weights", set_weights_cmd))
    app.add_handler(CommandHandler("set_risk", set_risk_cmd))
    app.add_handler(CommandHandler("set_noise", set_noise_cmd))

    app.run_polling()

if __name__ == "__main__":
    main()
