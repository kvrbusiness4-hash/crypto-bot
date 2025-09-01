# -*- coding: utf-8 -*-
# bot_paid.py — Bybit signals only (no trading), with scheduler & saved settings

import os, math, json, asyncio, aiohttp, logging
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ========= ENV =========
TG_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BYBIT_PUBLIC  = "https://api.bybit.com"
BYBIT_PROXY   = os.getenv("BYBIT_PROXY","").strip()  # e.g. http://user:pass@ip:port
STATE_PATH    = os.getenv("STATE_PATH", "state.json")  # де зберігати налаштування

# ========= LOGS ========
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("signals-bot")

# ========= DEFAULTS / STATE =========
DEFAULTS: Dict[str, float | int | str | bool] = {
    "top_n": 3,          # 1..3 — скільки монет показувати
    "strength": 2,       # 2 або 3 — вимогливість тренду (3 — суворіше)
    "noise": 1.0,        # 0.2..5.0 — мін. волатильність (%) за ~48 свічок (шум-фільтр)
    "sl": 3.0,           # SL у %
    "tp": 5.0,           # TP у %
    "lev_mode": "auto",  # 'auto' або 'manual' (лише рекомендація)
    "lev_base": 3,       # базове плече, якщо manual
    "use_rsi": True,
    "use_macd": True,
    "use_ema": True,
    # авто-розсилка
    "auto_on": False,
    "every": 15,         # хвилин
}

STATE: Dict[int, Dict[str, float | int | str | bool]] = {}

# ========= PERSIST =========
def load_state() -> None:
    global STATE
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # ключі з файлу — рядки; приводимо до int
        STATE = {int(k): v for k, v in raw.items()}
        log.info("State loaded: %d chats", len(STATE))
    except Exception:
        STATE = {}

def save_state() -> None:
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in STATE.items()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("State save failed: %s", e)

def get_st(chat_id: int) -> Dict[str, float | int | str | bool]:
    st = STATE.get(chat_id)
    if not st:
        st = DEFAULTS.copy()
        STATE[chat_id] = st
        save_state()
    return st

# ========= UI =========
def kb(st: Dict[str, float | int | str | bool]) -> ReplyKeyboardMarkup:
    rows = [
        ["/signals", "/status"],
        [f"/set_strength {st.get('strength',2)}", f"/set_top {st.get('top_n',3)}"],
        [f"/set_noise {st.get('noise',1.0)}", f"/set_risk {int(st.get('sl',3))} {int(st.get('tp',5))}"],
        [f"/set_lev {st.get('lev_mode','auto')}", f"/set_lev_base {st.get('lev_base',3)}"],
        [f"/set_rsi {'on' if st.get('use_rsi',True) else 'off'}",
         f"/set_macd {'on' if st.get('use_macd',True) else 'off'}"],
        [f"/set_ema {'on' if st.get('use_ema',True) else 'off'}"],
        [f"/auto_on {st.get('every',15)}", "/auto_off"],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

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
    k = 2.0 / (period + 1.0)
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
    ef, es = ema(series, fast), ema(series, slow)
    macd_line = [a - b for a, b in zip(ef[-len(es):], es)]
    sig = ema(macd_line, signal)
    L = min(len(macd_line), len(sig))
    return macd_line[-L:], sig[-L:]

def votes_from_series(series: List[float], st: Dict) -> Dict[str, float | int]:
    out = {"vote": 0, "rsi": None, "ema_trend": 0, "macd": None, "sig": None}
    if len(series) < 60: return out

    if st.get("use_rsi", True):
        rr = rsi(series, 14)
        if rr:
            out["rsi"] = rr[-1]
            if rr[-1] <= 30: out["vote"] += 1
            if rr[-1] >= 70: out["vote"] -= 1

    if st.get("use_macd", True):
        m, s = macd(series)
        if m and s:
            out["macd"], out["sig"] = m[-1], s[-1]
            if m[-1] > s[-1]: out["vote"] += 1
            if m[-1] < s[-1]: out["vote"] -= 1

    if st.get("use_ema", True):
        e50 = ema(series, 50)
        e200 = ema(series, 200) if len(series) >= 200 else ema(series, max(100, len(series)//2))
        if e50 and e200:
            out["ema_trend"] = 1 if e50[-1] > e200[-1] else -1
            out["vote"] += 1 if e50[-1] > e200[-1] else -1

    return out

def decide_direction(v15:int, v30:int, v60:int, strength:int) -> Optional[str]:
    total = v15 + v30 + v60
    pos = sum(1 for v in [v15, v30, v60] if v > 0)
    neg = sum(1 for v in [v15, v30, v60] if v < 0)
    if strength <= 2:
        if total >= 2 and pos >= 2: return "LONG"
        if total <= -2 and neg >= 2: return "SHORT"
    else:
        if total >= 3 and pos >= 2: return "LONG"
        if total <= -3 and neg >= 2: return "SHORT"
    return None

def calc_vol_pct(series: List[float], px: float) -> float:
    tail = series[-48:] if len(series) >= 48 else series
    if len(tail) < 2 or px <= 0: return 0.0
    mean = sum(tail)/len(tail)
    var = sum((x-mean)**2 for x in tail)/len(tail)
    return (math.sqrt(var)/px)*100.0

def suggest_leverage(ch24_abs: float, vol_pct: float, base: int, mode: str) -> int:
    if mode != "auto":
        return max(1, int(base))
    if vol_pct < 1.5 and ch24_abs < 2: return 5
    if vol_pct < 3.0 and ch24_abs < 5: return 3
    return 2

# ========= HTTP =========
async def http_json(session: aiohttp.ClientSession, url: str, params: dict | None = None) -> dict:
    delay = 0.7
    for i in range(5):
        try:
            async with session.get(url, params=params, timeout=25, **_proxy_kwargs()) as r:
                r.raise_for_status()
                return await r.json()
        except aiohttp.ClientResponseError as e:
            if e.status == 429 and i < 4:
                await asyncio.sleep(delay); delay *= 1.8; continue
            raise
        except Exception:
            if i == 4: raise
            await asyncio.sleep(delay); delay *= 1.5

async def bybit_top_symbols(session: aiohttp.ClientSession, top:int=40) -> List[dict]:
    data = await http_json(session, f"{BYBIT_PUBLIC}/v5/market/tickers", {"category":"linear"})
    lst = ((data.get("result") or {}).get("list")) or []
    def _vol(x):
        try: return float(x.get("turnover24h") or 0.0)
        except: return 0.0
    lst.sort(key=_vol, reverse=True)
    return [x for x in lst if str(x.get("symbol","")).endswith("USDT")][:top]

async def bybit_klines(session: aiohttp.ClientSession, symbol: str, interval: str, limit: int = 300) -> List[float]:
    data = await http_json(session, f"{BYBIT_PUBLIC}/v5/market/kline", {
        "category":"linear","symbol":symbol,"interval":interval,"limit":str(limit)
    })
    res = data.get("result") or {}
    rows = list(reversed(res.get("list") or []))
    closes: List[float] = []
    for r in rows:
        try: closes.append(float(r[4]))
        except: pass
    return closes

# ========= Core signals =========
async def build_signals(chat_id: int) -> str:
    st = get_st(chat_id)
    top_n    = int(max(1, min(3, st.get("top_n", 3))))
    strength = int(3 if st.get("strength",2) >= 3 else 2)
    noise    = float(max(0.2, min(5.0, st.get("noise",1.0))))
    sl_pct   = float(max(0.0, st.get("sl", 3.0)))
    tp_pct   = float(max(0.0, st.get("tp", 5.0)))

    async with aiohttp.ClientSession() as s:
        try:
            tickers = await bybit_top_symbols(s, 40)
        except Exception as e:
            return f"⚠️ Ринок недоступний: {e}"

        scored = []  # (score, sym, dir, px, note, ch_abs, k15, vol_pct)
        for t in tickers:
            sym = t.get("symbol","")
            try:
                px = float(t.get("lastPrice") or 0.0)
                ch24 = float(t.get("price24hPcnt") or 0.0) * 100.0
            except:
                continue
            if px <= 0: 
                continue

            try:
                k15 = await bybit_klines(s, sym, "15", 300); await asyncio.sleep(0.12)
                k30 = await bybit_klines(s, sym, "30", 300); await asyncio.sleep(0.12)
                k60 = await bybit_klines(s, sym, "60", 300)
            except:
                continue
            if not (k15 and k30 and k60):
                continue

            vol_pct = calc_vol_pct(k15, px)
            if vol_pct < noise:
                continue

            v15 = votes_from_series(k15, st)
            v30 = votes_from_series(k30, st)
            v60 = votes_from_series(k60, st)
            direction = decide_direction(int(v15["vote"]), int(v30["vote"]), int(v60["vote"]), strength)
            if not direction:
                continue

            score = v15["vote"]*1 + v30["vote"]*1.5 + v60["vote"]*2
            if v60["ema_trend"] == 1 and direction=="LONG": score += 1
            if v60["ema_trend"] == -1 and direction=="SHORT": score += 1
            score += min(2.0, abs(ch24)/10.0)

            def mark(tag, v):
                r = v.get("rsi"); rtxt = f"{int(r)}" if isinstance(r,(int,float)) else "-"
                m, sgn = v.get("macd"), v.get("sig")
                mtxt = "↑" if (m is not None and sgn is not None and m > sgn) else ("↓" if (m is not None and sgn is not None and m < sgn) else "·")
                et = v.get("ema_trend",0); etxt = "↑" if et==1 else ("↓" if et==-1 else "·")
                return f"{tag}RSI:{rtxt} MACD:{mtxt} EMA:{etxt}"

            note = f"{mark('15m',v15)} | {mark('30m',v30)} | {mark('1h',v60)}"
            scored.append((float(score), sym, direction, px, note, abs(ch24), k15, vol_pct))
            await asyncio.sleep(0.2)

        if not scored:
            return "⚠️ Узгоджених сильних сигналів не знайдено з поточними фільтрами."

        scored.sort(key=lambda x: x[0], reverse=True)
        picks = scored[:top_n]

        lines = ["📈 *Сильні сигнали:*\n"]
        for sc, sym, direction, px, note, ch_abs, k15, vol_pct in picks:
            if direction == "LONG":
                slp = px*(1-sl_pct/100.0); tpp = px*(1+tp_pct/100.0)
            else:
                slp = px*(1+sl_pct/100.0); tpp = px*(1-tp_pct/100.0)
            lev = suggest_leverage(ch_abs, vol_pct, int(st.get("lev_base",3)), str(st.get("lev_mode","auto")))
            lines.append(
                f"• *{sym}*: *{direction}* @ `{px:.6f}`\n"
                f"  SL: `{slp:.6f}` · TP: `{tpp:.6f}` · LEV: `{lev}` (vol≈{vol_pct:.2f}% | 24hΔ≈{ch_abs:.2f}%)\n"
                f"  {note}\n"
            )

        return "\n".join(lines).strip()

# ========= Commands =========
async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = get_st(u.effective_chat.id)
    await u.message.reply_text(
        "👋 Привіт! Я надсилаю *аналітичні сигнали* Bybit (без торгів).\n"
        "Натисни /signals — отримаєш найкращі сетапи зараз.\n"
        "Можна ввімкнути автопостинг: /auto_on 15",
        parse_mode=ParseMode.MARKDOWN, reply_markup=kb(st)
    )

async def status_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = get_st(u.effective_chat.id)
    text = (
        f"⚙️ *Налаштування*\n"
        f"Монет: *{st['top_n']}* · Сила тренду: *{st['strength']}* · Шум: *{st['noise']:.2f}%*\n"
        f"SL/TP: *{st['sl']:.1f}%* / *{st['tp']:.1f}%*\n"
        f"Leverage: *{st['lev_mode']}* (base={st['lev_base']})\n"
        f"RSI: *{'on' if st['use_rsi'] else 'off'}* · "
        f"MACD: *{'on' if st['use_macd'] else 'off'}* · "
        f"EMA: *{'on' if st['use_ema'] else 'off'}*\n"
        f"Автопостинг: *{'ON' if st['auto_on'] else 'OFF'}* кожні *{st['every']}* хв\n"
        f"UTC: {utc_now()}"
    )
    await u.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=kb(st))

async def signals_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    txt = await build_signals(u.effective_chat.id)
    for ch in split_long(txt):
        await u.message.reply_text(ch, parse_mode=ParseMode.MARKDOWN)

# setters
async def set_top_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = get_st(u.effective_chat.id)
    try:
        n = int(c.args[0]); assert 1 <= n <= 3
        st["top_n"] = n; save_state()
        await u.message.reply_text(f"OK. Кількість монет: {n}", reply_markup=kb(st))
    except:
        await u.message.reply_text("Формат: /set_top 1..3")

async def set_strength_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = get_st(u.effective_chat.id)
    try:
        s = int(c.args[0]); assert s in (2,3)
        st["strength"] = s; save_state()
        await u.message.reply_text(f"OK. Сила тренду: {s}", reply_markup=kb(st))
    except:
        await u.message.reply_text("Формат: /set_strength 2|3")

async def set_noise_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = get_st(u.effective_chat.id)
    try:
        v = float(c.args[0]); assert 0.2 <= v <= 5.0
        st["noise"] = v; save_state()
        await u.message.reply_text(f"OK. Фільтр шуму: {v:.2f}%", reply_markup=kb(st))
    except:
        await u.message.reply_text("Формат: /set_noise 0.2..5.0 (у %)")

async def set_risk_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = get_st(u.effective_chat.id)
    try:
        sl = float(c.args[0]); tp = float(c.args[1]); assert sl >= 0 and tp >= 0
        st["sl"], st["tp"] = sl, tp; save_state()
        await u.message.reply_text(f"OK. SL={sl:.1f}% · TP={tp:.1f}%", reply_markup=kb(st))
    except:
        await u.message.reply_text("Формат: /set_risk 3 5")

async def set_lev_mode_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = get_st(u.effective_chat.id)
    try:
        mode = str(c.args[0]).lower(); assert mode in ("auto","manual")
        st["lev_mode"] = mode; save_state()
        await u.message.reply_text(f"OK. Режим плеча: {mode}", reply_markup=kb(st))
    except:
        await u.message.reply_text("Формат: /set_lev auto|manual")

async def set_lev_base_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = get_st(u.effective_chat.id)
    try:
        base = int(c.args[0]); assert 1 <= base <= 10
        st["lev_base"] = base; save_state()
        await u.message.reply_text(f"OK. Базове плече: {base}", reply_markup=kb(st))
    except:
        await u.message.reply_text("Формат: /set_lev_base 1..10")

async def set_rsi_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = get_st(u.effective_chat.id)
    try:
        on = str(c.args[0]).lower() in ("on","true","1")
        st["use_rsi"] = on; save_state()
        await u.message.reply_text(f"OK. RSI: {'on' if on else 'off'}", reply_markup=kb(st))
    except:
        await u.message.reply_text("Формат: /set_rsi on|off")

async def set_macd_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = get_st(u.effective_chat.id)
    try:
        on = str(c.args[0]).lower() in ("on","true","1")
        st["use_macd"] = on; save_state()
        await u.message.reply_text(f"OK. MACD: {'on' if on else 'off'}", reply_markup=kb(st))
    except:
        await u.message.reply_text("Формат: /set_macd on|off")

async def set_ema_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = get_st(u.effective_chat.id)
    try:
        on = str(c.args[0]).lower() in ("on","true","1")
        st["use_ema"] = on; save_state()
        await u.message.reply_text(f"OK. EMA: {'on' if on else 'off'}", reply_markup=kb(st))
    except:
        await u.message.reply_text("Формат: /set_ema on|off")

# ========= Scheduler =========
async def auto_job(ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = ctx.job.data["chat_id"]
    try:
        text = await build_signals(chat_id)
        for ch in split_long(text):
            await ctx.bot.send_message(chat_id=chat_id, text=ch, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        log.error("auto_job error: %s", e)

async def auto_on_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = get_st(u.effective_chat.id)
    minutes = st.get("every", 15)
    if c.args:
        try:
            minutes = max(5, min(180, int(c.args[0])))
        except:
            pass
    st["every"] = minutes
    st["auto_on"] = True
    save_state()
    name = f"auto_{u.effective_chat.id}"
    # очистити попередні
    for j in c.application.job_queue.get_jobs_by_name(name):
        j.schedule_removal()
    c.application.job_queue.run_repeating(
        auto_job, interval=minutes*60, first=5, name=name, data={"chat_id": u.effective_chat.id}
    )
    await u.message.reply_text(f"✅ Автопостинг увімкнено: кожні {minutes} хв.", reply_markup=kb(st))

async def auto_off_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = get_st(u.effective_chat.id)
    st["auto_on"] = False
    save_state()
    name = f"auto_{u.effective_chat.id}"
    for j in c.application.job_queue.get_jobs_by_name(name):
        j.schedule_removal()
    await u.message.reply_text("⏸ Автопостинг вимкнено.", reply_markup=kb(st))

# ========= Main =========
def main():
    if not TG_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN"); return
    load_state()

    print("Signals bot running | TF 15/30/60 | top≤3 | scheduler | no trading")
    app = Application.builder().token(TG_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))

    app.add_handler(CommandHandler("set_top", set_top_cmd))
    app.add_handler(CommandHandler("set_strength", set_strength_cmd))
    app.add_handler(CommandHandler("set_noise", set_noise_cmd))
    app.add_handler(CommandHandler("set_risk", set_risk_cmd))
    app.add_handler(CommandHandler("set_lev", set_lev_mode_cmd))
    app.add_handler(CommandHandler("set_lev_base", set_lev_base_cmd))
    app.add_handler(CommandHandler("set_rsi", set_rsi_cmd))
    app.add_handler(CommandHandler("set_macd", set_macd_cmd))
    app.add_handler(CommandHandler("set_ema", set_ema_cmd))

    app.add_handler(CommandHandler("auto_on", auto_on_cmd))
    app.add_handler(CommandHandler("auto_off", auto_off_cmd))

    app.run_polling()

if __name__ == "__main__":
    main()
