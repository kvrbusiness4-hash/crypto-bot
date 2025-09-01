# -*- coding: utf-8 -*-
# bot_signals.py — сигнали + аналітика, без автотрейду

import os, math, asyncio, aiohttp, logging
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ============ ENV ============
TG_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BYBIT_PUBLIC = "https://api.bybit.com"
BYBIT_PROXY  = os.getenv("BYBIT_PROXY", "").strip()

# ========= Базові дефолти користувача =========
DEFAULTS = {
    "top_n": 3,          # макс. монет у відповіді
    "strength": 2,       # 1..3 (мін. сила сигналу)
    "noise": 1.0,        # % ATR як фільтр шуму (0.5 .. 3.0)
    "sl": 3.0,           # базовий SL% (використ. якщо auto дає екстреми)
    "tp": 5.0,           # базовий TP%
    "lev_mode": "auto",  # 'auto' або 'manual'
    "lev_base": 3,       # базове плече у manual
}

# ========= Логи =========
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("signals-bot")

# ========= Держава по чатах =========
STATE: Dict[int, Dict[str, float | int | str]] = {}

# ========= UI (мінімум, тільки корисне) =========
def kb(st: Dict[str, float | int | str]) -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            ["/signals", "/status"],
            [f"/set_strength {st.get('strength',2)}", f"/set_top {st.get('top_n',3)}"],
            [f\"/set_noise {st.get('noise',1.0)}\", f\"/set_risk {int(st.get('sl',3))} {int(st.get('tp',5))}\"],
            [f\"/set_lev {st.get('lev_mode','auto')}\", f\"/set_lev_base {st.get('lev_base',3)}\"],
        ],
        resize_keyboard=True
    )

# ========= Helpers =========
def split_long(text: str, n: int = 3500) -> List[str]:
    out = []
    while len(text) > n:
        out.append(text[:n]); text = text[n:]
    out.append(text)
    return out

def utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

def _proxy_kwargs() -> dict:
    return {"proxy": BYBIT_PROXY} if BYBIT_PROXY.startswith(("http://","https://")) else {}

# ========= Indicators =========
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

def atr_from_klines(kl: List[List[str | float]], period: int = 14) -> List[float]:
    # kline формат Bybit v5: [start, open, high, low, close, volume, ...]
    if len(kl) < period + 1: return []
    trs = []
    prev_close = float(kl[0][4])
    for i in range(1, len(kl)):
        h = float(kl[i][2]); l = float(kl[i][3]); c_prev = prev_close
        tr = max(h-l, abs(h-c_prev), abs(l-c_prev))
        trs.append(tr)
        prev_close = float(kl[i][4])
    # простий EMA(TR) ~ ATR
    return ema(trs, period)

# ========= HTTP (public) =========
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
    def _turnover(x):
        try: return float(x.get("turnover24h") or 0)
        except: return 0.0
    lst.sort(key=_turnover, reverse=True)
    return [x for x in lst if str(x.get("symbol","")).endswith("USDT")][:top]

async def bybit_klines_raw(session: aiohttp.ClientSession, symbol: str, interval: str, limit: int = 300) -> List[List[str | float]]:
    data = await http_json(session, f"{BYBIT_PUBLIC}/v5/market/kline", {
        "category":"linear","symbol":symbol,"interval":interval,"limit":str(limit)
    })
    rows = ((data.get("result") or {}).get("list")) or []
    rows = list(reversed(rows))  # oldest -> newest
    return rows

# ========= Scoring =========
def vote_block(series: List[float]) -> Dict[str, float | int]:
    out = {"vote":0, "rsi":None, "ema_trend":0, "macd":None, "sig":None}
    if len(series) < 60: return out
    rr = rsi(series,14); m, s = macd(series)
    e50 = ema(series,50); e200 = ema(series,200) if len(series)>=200 else ema(series, max(100, len(series)//2))
    if rr:
        out["rsi"] = rr[-1]
        if rr[-1] <= 30: out["vote"] += 1
        if rr[-1] >= 70: out["vote"] -= 1
    if m and s:
        out["macd"], out["sig"] = m[-1], s[-1]
        out["vote"] += 1 if m[-1] > s[-1] else -1
    if e50 and e200:
        trend = 1 if e50[-1] > e200[-1] else -1
        out["ema_trend"] = trend
        out["vote"] += trend
    return out

def decide_direction(v15:int, v30:int, v60:int) -> Optional[str]:
    total = v15 + v30 + v60
    pos = sum(1 for v in [v15,v30,v60] if v>0)
    neg = sum(1 for v in [v15,v30,v60] if v<0)
    if total >= 3 and pos >= 2: return "LONG"
    if total <= -3 and neg >= 2: return "SHORT"
    return None

def lev_auto_from_vol(atr_pct: float) -> int:
    # проста шкала: менша волатильність -> більше плече
    if atr_pct < 1.0: return 5
    if atr_pct < 2.0: return 3
    return 2

def sl_tp_from_vol(direction:str, price:float, atr_pct:float, base_sl:float, base_tp:float) -> Tuple[float,float,float,float]:
    # орієнтуємося на ATR% (за 14), але не виходимо за 0.8×..1.8× бази
    sl_pct = max(0.8*base_sl, min(1.8*base_sl, max(0.7, atr_pct*0.8)))
    tp_pct = max(0.8*base_tp, min(1.8*base_tp, max(1.0, atr_pct*1.2)))
    if direction=="LONG":
        sl = price*(1-sl_pct/100.0); tp = price*(1+tp_pct/100.0)
    else:
        sl = price*(1+sl_pct/100.0); tp = price*(1-tp_pct/100.0)
    return sl, tp, sl_pct, tp_pct

# ========= Аналітика монети =========
def render_marks(v15, v30, v60) -> str:
    def mk(v):
        r = v["rsi"]; rtxt = f"{r:.0f}" if isinstance(r,(int,float)) else "-"
        m = v["macd"]; s = v["sig"]
        mtxt = "↑" if (m is not None and s is not None and m>s) else ("↓" if (m is not None and s is not None and m<s) else "·")
        et = v["ema_trend"]; etxt = "↑" if et==1 else ("↓" if et==-1 else "·")
        return f"RSI:{rtxt} MACD:{mtxt} EMA:{etxt}"
    return f"15m {mk(v15)} | 30m {mk(v30)} | 1h {mk(v60)}"

# ========= Головна побудова сигналів =========
async def build_signals(chat_id:int) -> str:
    st = STATE.setdefault(chat_id, DEFAULTS.copy())
    top_n     = int(st.get("top_n",3))
    strength  = int(st.get("strength",2))
    noise_pct = float(st.get("noise",1.0))
    base_sl   = float(st.get("sl",3.0))
    base_tp   = float(st.get("tp",5.0))
    lev_mode  = str(st.get("lev_mode","auto"))
    lev_base  = int(st.get("lev_base",3))

    async with aiohttp.ClientSession() as s:
        try:
            tickers = await bybit_top_symbols(s, 25)
        except Exception as e:
            return f"⚠️ Ринок недоступний: {e}"

        scored = []
        for t in tickers:
            sym = t.get("symbol","")
            try:
                px  = float(t.get("lastPrice") or 0.0)
                ch24 = float(t.get("price24hPcnt") or 0.0)*100.0
            except:
                continue
            if px <= 0: continue

            try:
                k15 = await bybit_klines_raw(s, sym, "15", 300); await asyncio.sleep(0.2)
                k30 = await bybit_klines_raw(s, sym, "30", 300); await asyncio.sleep(0.2)
                k60 = await bybit_klines_raw(s, sym, "60", 300)
            except: 
                continue
            if not (k15 and k30 and k60): 
                continue

            c15 = [float(x[4]) for x in k15]
            c30 = [float(x[4]) for x in k30]
            c60 = [float(x[4]) for x in k60]

            v15 = vote_block(c15)
            v30 = vote_block(c30)
            v60 = vote_block(c60)
            direction = decide_direction(v15["vote"], v30["vote"], v60["vote"])
            if not direction: 
                continue

            # Волатильність (ATR%) на 15м
            atr = atr_from_klines(k15, 14)
            if not atr: 
                continue
            atr_pct = (atr[-1] / px) * 100.0

            # Мінфільтр «шуму»
            if atr_pct < noise_pct:
                continue

            # Сила сигналу (0..5)
            score = v15["vote"] + v30["vote"] + v60["vote"]
            if (v60["ema_trend"]==1 and direction=="LONG") or (v60["ema_trend"]==-1 and direction=="SHORT"):
                score += 1
            score += min(2.0, abs(ch24)/10.0)

            # strength поріг (2 ~ помірно, 3 ~ високий)
            if score < strength:
                continue

            # SL/TP
            sl, tp, slp, tpp = sl_tp_from_vol(direction, px, atr_pct, base_sl, base_tp)

            # Плече
            lev = lev_auto_from_vol(atr_pct) if lev_mode=="auto" else max(1, int(lev_base))

            marks = render_marks(v15, v30, v60)
            scored.append((
                float(score), sym, direction, px, sl, tp, slp, tpp, lev, atr_pct, ch24, marks
            ))
            await asyncio.sleep(0.25)

        if not scored:
            return "⚠️ Узгоджених сильних сигналів не знайдено."

        scored.sort(key=lambda x: x[0], reverse=True)
        picks = scored[:max(1, min(3, top_n))]

        lines = []
        for sc, sym, dirn, px, sl, tp, slp, tpp, lev, atrp, ch24, marks in picks:
            lines.append(
                f"• *{sym}*: *{dirn}* @ `{px:.6f}`\n"
                f"  SL:`{sl:.6f}` ({slp:.1f}%) · TP:`{tp:.6f}` ({tpp:.1f}%) · LEV:`{lev}` · ATR%:`{atrp:.2f}` · 24h:`{ch24:.2f}%`\n"
                f"  {marks}"
            )

        header = "📈 *Сильні сигнали (аналітика, без автотрейду):*\n"
        footer = f"\n⚙️ strength:{strength} • noise:{noise_pct:.2f}% • max:{top_n} • lev:{lev_mode}({lev_base}) • UTC:{utc_now()}"
        return header + "\n\n".join(lines) + footer

# ========= Commands =========
async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, DEFAULTS.copy())
    await u.message.reply_text("👋 Готовий. Натисни /signals щоб отримати аналітику.", reply_markup=kb(st))

async def signals_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    txt = await build_signals(u.effective_chat.id)
    for ch in split_long(txt):
        await u.message.reply_text(ch, parse_mode=ParseMode.MARKDOWN)

async def status_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, DEFAULTS.copy())
    msg = (
        f"ℹ️ Статус:\n"
        f"- top_n: {st.get('top_n')}\n"
        f"- strength: {st.get('strength')}\n"
        f"- noise (ATR% фільтр): {st.get('noise')}\n"
        f"- risk (SL/TP): {st.get('sl')}% / {st.get('tp')}%\n"
        f"- leverage: {st.get('lev_mode')} (base={st.get('lev_base')})\n"
        f"- UTC: {utc_now()}"
    )
    await u.message.reply_text(msg, reply_markup=kb(st))

async def set_top_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, DEFAULTS.copy())
    try:
        n = int(c.args[0]); assert 1 <= n <= 3
        st["top_n"] = n
        await u.message.reply_text(f"OK. Показуватиму до {n} монет.", reply_markup=kb(st))
    except:
        await u.message.reply_text("Формат: /set_top 1..3")

async def set_strength_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, DEFAULTS.copy())
    try:
        v = int(c.args[0]); assert 1 <= v <= 3
        st["strength"] = v
        await u.message.reply_text(f"OK. Мінімальна сила сигналу = {v}.", reply_markup=kb(st))
    except:
        await u.message.reply_text("Формат: /set_strength 1..3")

async def set_noise_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, DEFAULTS.copy())
    try:
        v = float(c.args[0]); assert 0.3 <= v <= 5.0
        st["noise"] = v
        await u.message.reply_text(f"OK. Noise (ATR%% фільтр) = {v:.2f}%.", reply_markup=kb(st))
    except:
        await u.message.reply_text("Формат: /set_noise 0.3..5.0")

async def set_risk_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, DEFAULTS.copy())
    try:
        sl = float(c.args[0]); tp = float(c.args[1]); assert sl>=0 and tp>=0
        st["sl"], st["tp"] = sl, tp
        await u.message.reply_text(f"OK. База SL/TP = {sl:.1f}% / {tp:.1f}%.", reply_markup=kb(st))
    except:
        await u.message.reply_text("Формат: /set_risk 3 5")

async def set_levmode_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, DEFAULTS.copy())
    try:
        mode = str(c.args[0]).lower(); assert mode in ("auto","manual")
        st["lev_mode"] = mode
        await u.message.reply_text(f"OK. Режим плеча: {mode}.", reply_markup=kb(st))
    except:
        await u.message.reply_text("Формат: /set_lev auto|manual")

async def set_levbase_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, DEFAULTS.copy())
    try:
        v = int(c.args[0]); assert 1 <= v <= 20
        st["lev_base"] = v
        await u.message.reply_text(f"OK. Базове плече = {v}.", reply_markup=kb(st))
    except:
        await u.message.reply_text("Формат: /set_lev_base 1..20")

# ========= Main =========
def main():
    if not TG_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN"); return
    print("Signals bot running | TF=15/30/60 | top<=3 | RSI/EMA/MACD/ATR | no-trade")

    app = Application.builder().token(TG_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    app.add_handler(CommandHandler("set_top", set_top_cmd))
    app.add_handler(CommandHandler("set_strength", set_strength_cmd))
    app.add_handler(CommandHandler("set_noise", set_noise_cmd))
    app.add_handler(CommandHandler("set_risk", set_risk_cmd))
    app.add_handler(CommandHandler("set_lev", set_levmode_cmd))
    app.add_handler(CommandHandler("set_lev_base", set_levbase_cmd))

    app.run_polling()

if __name__ == "__main__":
    main()
