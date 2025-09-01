# -*- coding: utf-8 -*-
# bot_signals.py — Bybit Signals (NO autotrade)

import os, math, asyncio, aiohttp, logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone, time as dtime

from telegram import Update, ReplyKeyboardMarkup, BotCommand
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ========= ENV =========
TG_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BYBIT_PROXY = os.getenv("BYBIT_PROXY", "").strip()              # http://user:pass@host:port (optional)

DEFAULT_AUTO_MIN = int(os.getenv("DEFAULT_AUTO_MIN", "15"))     # період автопостингу, хв
TOP_N            = int(os.getenv("TOP_N", "3"))                 # до N монет у сигналі

# ========= LOGS =========
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("signals")

# ========= PROFILES (нове) =========
PROFILES = {
    "scalp": {
        "top_n": 5,
        "noise": 1.0,
        "trend_weight": 3,
        "atr_len": 10,
        "sl_k": 1.2,
        "tp_k": 1.8,
        "rr_min": 1.5,
        "min_turnover": 100.0,
        "max_spread_bps": 8,
        "max_24h_change": 25.0,
        "cooldown_min": 60,
        "every": 15,
    },
    "default": {
        "top_n": 3,
        "noise": 1.8,
        "trend_weight": 3,
        "atr_len": 14,
        "sl_k": 1.5,
        "tp_k": 2.5,
        "rr_min": 1.8,
        "min_turnover": 150.0,
        "max_spread_bps": 5,
        "max_24h_change": 15.0,
        "cooldown_min": 180,
        "every": 15,
    },
    "swing": {
        "top_n": 3,
        "noise": 3.0,
        "trend_weight": 4,
        "atr_len": 20,
        "sl_k": 2.0,
        "tp_k": 3.5,
        "rr_min": 2.0,
        "min_turnover": 150.0,
        "max_spread_bps": 5,
        "max_24h_change": 20.0,
        "cooldown_min": 360,
        "every": 15,
    },
}

# ========= UI =========
def _kb(_: Dict[str, object]) -> ReplyKeyboardMarkup:
    # мінімальне меню: профілі + базові команди
    return ReplyKeyboardMarkup(
        [
            ["/scalp", "/default", "/swing"],
            ["/signals", "/status"],
        ],
        resize_keyboard=True
    )

# ========= STATE =========
STATE: Dict[int, Dict[str, object]] = {}

def default_state() -> Dict[str, object]:
    return {
        # фільтри ринку
        "min_turnover": 150.0,     # млн USDT за 24h
        "max_spread_bps": 5,       # базисні пункти (0.05%)
        "max_24h_change": 15.0,    # %
        "whitelist": set(),        # якщо пусто — не застосовується
        "blacklist": set({"TRUMPUSDT","PUMPFUNUSDT","FARTCOINUSDT","IPUSDT","ENAUSDT"}),
        # логіка тренду/шуму
        "noise": 1.8,              # % мін. амплітуда очікуваного руху
        "trend_weight": 3,         # суворість голосування 15/30/60
        # ATR/ризики
        "atr_len": 14,
        "sl_k": 1.5,
        "tp_k": 2.5,
        "rr_min": 1.8,
        # сигнали
        "top_n": TOP_N,
        "every": DEFAULT_AUTO_MIN,
        "auto_on": False,
        # таймінги
        "sess_from": 12,           # 12:00–20:00 UTC
        "sess_to": 20,
        "cooldown_min": 180,       # хвилин між повторними сигналами по тій самій монеті
        "_last_sig_ts": {},        # symbol -> ts
    }

# ========= Helpers =========
def utc_now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

def in_session(st) -> bool:
    now = datetime.now(timezone.utc).time()
    f, t = int(st["sess_from"]), int(st["sess_to"])
    if f <= t:
        return dtime(f,0) <= now <= dtime(t,0)
    return now >= dtime(f,0) or now <= dtime(t,0)

def _proxy_kwargs() -> dict:
    return {"proxy": BYBIT_PROXY} if BYBIT_PROXY.startswith(("http://","https://")) else {}

def split_long(text: str, n: int = 3500) -> List[str]:
    out = []
    while len(text) > n:
        out.append(text[:n]); text = text[n:]
    out.append(text)
    return out

# ========= Indicators =========
def ema(xs: List[float], p: int) -> List[float]:
    if not xs: return []
    k = 2/(p+1)
    out = [xs[0]]
    for x in xs[1:]:
        out.append(out[-1] + k*(x - out[-1]))
    return out

def rsi(xs: List[float], p: int = 14) -> List[float]:
    if len(xs) < p+1: return []
    gains, losses = [], []
    for i in range(1, len(xs)):
        d = xs[i]-xs[i-1]
        gains.append(max(0,d)); losses.append(max(0,-d))
    ag = sum(gains[:p])/p; al = sum(losses[:p])/p
    out = [0.0]*p
    out.append(100.0 if al==0 else 100-100/(1+ag/(al+1e-9)))
    for i in range(p, len(gains)):
        ag = (ag*(p-1)+gains[i])/p
        al = (al*(p-1)+losses[i])/p
        out.append(100.0 if al==0 else 100-100/(1+ag/(al+1e-9)))
    return out

def macd(xs: List[float], fast=12, slow=26, signal=9) -> Tuple[List[float], List[float]]:
    if len(xs) < slow+signal: return [], []
    ef, es = ema(xs, fast), ema(xs, slow)
    m = [a-b for a,b in zip(ef[-len(es):], es)]
    s = ema(m, signal)
    L = min(len(m), len(s))
    return m[-L:], s[-L:]

def atr(high: List[float], low: List[float], close: List[float], n: int=14) -> float:
    if len(close) < n+1 or len(high)!=len(low)!=len(close): return 0.0
    trs=[]
    for i in range(1,len(close)):
        tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        trs.append(tr)
    if len(trs)<n: return 0.0
    return sum(trs[-n:])/n

# ========= HTTP =========
BASE = "https://api.bybit.com"

async def http_json(session: aiohttp.ClientSession, url: str, params: dict=None) -> dict:
    delay=0.6
    for i in range(5):
        try:
            async with session.get(url, params=params, timeout=25, **_proxy_kwargs()) as r:
                r.raise_for_status()
                return await r.json()
        except Exception:
            if i==4: raise
            await asyncio.sleep(delay); delay*=1.6

async def get_tickers(session) -> List[dict]:
    data = await http_json(session, f"{BASE}/v5/market/tickers", {"category":"linear"})
    return ((data.get("result") or {}).get("list")) or []

async def get_klines(session, symbol: str, interval: str, limit: int=300):
    data = await http_json(session, f"{BASE}/v5/market/kline",
                           {"category":"linear","symbol":symbol,"interval":interval,"limit":str(limit)})
    lst = list(reversed(((data.get("result") or {}).get("list")) or []))
    opens=[]; highs=[]; lows=[]; closes=[]
    for r in lst:
        try:
            opens.append(float(r[1])); highs.append(float(r[2])); lows.append(float(r[3])); closes.append(float(r[4]))
        except: pass
    return opens, highs, lows, closes

async def get_orderbook_spread_bps(session, symbol: str) -> float:
    data = await http_json(session, f"{BASE}/v5/market/orderbook", {"category":"linear","symbol":symbol,"limit":"1"})
    res = data.get("result") or {}
    bids = res.get("b") or []
    asks = res.get("a") or []
    if not bids or not asks: return 9999.0
    bid = float(bids[0][0]); ask=float(asks[0][0])
    if ask<=0: return 9999.0
    return (ask-bid)/ask*10000.0  # bps

# ========= Scoring / Filters =========
def votes_from_series(closes: List[float]) -> Dict[str, float]:
    out = {"vote":0,"rsi":None,"ema_trend":0,"macd":None,"sig":None}
    if len(closes)<60: return out
    rr = rsi(closes,14); m,s = macd(closes); e50=ema(closes,50); e200=ema(closes,200 if len(closes)>=200 else max(100,len(closes)//2))
    if rr:
        out["rsi"]=rr[-1]
        if rr[-1]<=30: out["vote"]+=1
        if rr[-1]>=70: out["vote"]-=1
    if m and s:
        out["macd"],out["sig"]=m[-1],s[-1]
        out["vote"] += 1 if m[-1]>s[-1] else -1
    if e50 and e200:
        et = 1 if e50[-1]>e200[-1] else -1
        out["ema_trend"]=et
        out["vote"] += 1 if et==1 else -1
    return out

def decide_direction(v15:int,v30:int,v60:int, need:int) -> Optional[str]:
    total=v15+v30+v60
    pos=sum(1 for v in [v15,v30,v60] if v>0)
    neg=sum(1 for v in [v15,v30,v60] if v<0)
    if total>=need and pos>=2: return "LONG"
    if total<=-need and neg>=2: return "SHORT"
    return None

def rr_ok(entry: float, sl: float, tp: float, rr_min: float) -> bool:
    risk = abs(entry-sl); reward=abs(tp-entry)
    if risk<=0: return False
    return (reward/risk) >= rr_min

# ========= Signals builder =========
async def build_signals(st: Dict[str,object]) -> str:
    if not in_session(st):
        return f"⏳ Поза торговою сесією (UTC {st['sess_from']:02.0f}:00–{st['sess_to']:02.0f}:00)."

    last_ts: Dict[str,float] = st["_last_sig_ts"]
    now_ts = datetime.utcnow().timestamp()

    async with aiohttp.ClientSession() as s:
        tickers = await get_tickers(s)

        # первинний відсів
        cands=[]
        for t in tickers:
            sym = str(t.get("symbol",""))
            if st["whitelist"] and sym not in st["whitelist"]:
                continue
            if sym in st["blacklist"]:
                continue
            try:
                vol = float(t.get("turnover24h") or 0)/1e6  # млн USDT
                ch24= float(t.get("price24hPcnt") or 0)*100.0
                px  = float(t.get("lastPrice") or 0)
            except:
                continue
            if vol < float(st["min_turnover"]):
                continue
            if abs(ch24) > float(st["max_24h_change"]):
                continue
            if px<=0:
                continue
            cands.append((sym, px, ch24))

        scored=[]
        for sym, px, ch24 in cands:
            sp_bps = await get_orderbook_spread_bps(s, sym)
            if sp_bps > float(st["max_spread_bps"]):
                continue

            o15,h15,l15,c15 = await get_klines(s, sym, "15", 300)
            await asyncio.sleep(0.15)
            o30,h30,l30,c30 = await get_klines(s, sym, "30", 300)
            await asyncio.sleep(0.15)
            o60,h60,l60,c60 = await get_klines(s, sym, "60", 300)
            if not (c15 and c30 and c60):
                continue

            v15=votes_from_series(c15)
            v30=votes_from_series(c30)
            v60=votes_from_series(c60)
            direction = decide_direction(v15["vote"], v30["vote"], v60["vote"], int(st["trend_weight"]))
            if not direction:
                continue

            atr_val = atr(h15,l15,c15,int(st["atr_len"]))
            if atr_val<=0:
                continue
            sl_k, tp_k = float(st["sl_k"]), float(st["tp_k"])
            if direction=="LONG":
                sl = px - sl_k*atr_val
                tp = px + tp_k*atr_val
            else:
                sl = px + sl_k*atr_val
                tp = px - tp_k*atr_val

            # фільтр шуму + R:R
            if abs(tp-px)/px*100.0 < float(st["noise"]):
                continue
            if not rr_ok(px, sl, tp, float(st["rr_min"])):
                continue

            # кулдаун
            lt = last_ts.get(sym, 0.0)
            if now_ts - lt < float(st["cooldown_min"])*60.0:
                continue

            score = (v15["vote"]+v30["vote"]+v60["vote"]) \
                    + (1 if (v60["ema_trend"]==1 and direction=="LONG") else 0) \
                    + (1 if (v60["ema_trend"]==-1 and direction=="SHORT") else 0)
            scored.append((score, sym, direction, px, sl, tp, sp_bps, ch24, atr_val, (v15,v30,v60)))

        if not scored:
            return "⚠️ Якісних сигналів не знайдено (фільтри відсіяли все)."

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max(1, int(st["top_n"]))]

        for _, sym, *_ in top:
            st["_last_sig_ts"][sym] = datetime.utcnow().timestamp()

        def mark(v):
            r = v["rsi"]; rtxt = f"{r:.0f}" if isinstance(r,(int,float)) else "-"
            m=v["macd"]; s=v["sig"]
            mtxt = "↑" if (m is not None and s is not None and m>s) else ("↓" if (m is not None and s is not None and m<s) else "·")
            et = v["ema_trend"]; etxt = "↑" if et==1 else ("↓" if et==-1 else "·")
            return f"RSI:{rtxt} MACD:{mtxt} EMA:{etxt}"

        body=[]
        for sc, sym, direc, px, sl, tp, sp_bps, ch24, atr_v, (v15,v30,v60) in top:
            rr = abs(tp-px)/max(1e-9,abs(px-sl))
            body.append(
                f"• *{sym}*: *{direc}* @ `{px:.6f}`\n"
                f"  SL:`{sl:.6f}` · TP:`{tp:.6f}` · ATR:`{atr_v:.6f}` · RR:`{rr:.2f}`\n"
                f"  spread:{sp_bps:.2f}bps · 24hΔ:{ch24:+.2f}% · noise≥{st['noise']:.2f}%\n"
                f"  15m {mark(v15)} | 30m {mark(v30)} | 1h {mark(v60)}"
            )
        return "📈 *Сильні сигнали:*\n\n" + "\n\n".join(body) + f"\n\nUTC: {utc_now_str()}"

# ========= Auto helpers (нове) =========
async def _start_autoposting(chat_id: int, app, st: Dict[str, object], minutes: int):
    st["every"] = minutes
    st["auto_on"] = True
    name = f"auto_{chat_id}"
    for j in app.job_queue.get_jobs_by_name(name):
        j.schedule_removal()
    app.job_queue.run_repeating(auto_job, interval=minutes*60, first=5, name=name, data={"chat_id": chat_id})

async def _scan_now_and_send(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(chat_id, default_state())
    txt = await build_signals(st)
    for ch in split_long(txt):
        await context.bot.send_message(chat_id=chat_id, text=ch, parse_mode=ParseMode.MARKDOWN)

async def _apply_profile_and_scan(u: Update, c: ContextTypes.DEFAULT_TYPE, key: str):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    p = PROFILES[key]

    # застосувати параметри
    st.update({
        "top_n": p["top_n"],
        "noise": p["noise"],
        "trend_weight": p["trend_weight"],
        "atr_len": p["atr_len"],
        "sl_k": p["sl_k"],
        "tp_k": p["tp_k"],
        "rr_min": p["rr_min"],
        "min_turnover": p["min_turnover"],
        "max_spread_bps": p["max_spread_bps"],
        "max_24h_change": p["max_24h_change"],
        "cooldown_min": p["cooldown_min"],
    })

    # увімкнути автопостинг і показати мінімальне меню
    await _start_autoposting(u.effective_chat.id, c.application, st, p["every"])
    await u.message.reply_text(
        f"✅ Профіль *{key}* застосовано. Автоскан кожні {p['every']} хв.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=_kb(st)
    )

    # миттєвий одноразовий скан
    await _scan_now_and_send(u.effective_chat.id, c)

# ========= Commands =========
async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    # поставимо коротке меню команд у клієнті Telegram
    try:
        await c.bot.set_my_commands([
            BotCommand("scalp", "Агресивний режим (скальпінг)"),
            BotCommand("default", "Стандартний режим"),
            BotCommand("swing", "Середньостроковий режим"),
            BotCommand("signals", "Сканувати зараз"),
            BotCommand("status", "Поточні налаштування"),
        ])
    except Exception:
        pass

    await u.message.reply_text(
        "👋 Готово. Бот видає *сигнали без автотрейду*.\nОберіть режим нижче.",
        parse_mode=ParseMode.MARKDOWN, reply_markup=_kb(st)
    )

async def signals_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await _scan_now_and_send(u.effective_chat.id, c)

async def status_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    text = (
        f"Автопостинг: {'ON' if st['auto_on'] else 'OFF'} кожні {st['every']} хв\n"
        f"TOP_N={st['top_n']} · noise={st['noise']}% · trend_weight={st['trend_weight']}\n"
        f"ATR(len={st['atr_len']}) · SL_k={st['sl_k']} · TP_k={st['tp_k']} · RR_min={st['rr_min']}\n"
        f"turnover≥{st['min_turnover']}M · spread≤{st['max_spread_bps']}bps · 24hΔ≤{st['max_24h_change']}%\n"
        f"session UTC {st['sess_from']:02.0f}-{st['sess_to']:02.0f} · cooldown={st['cooldown_min']}м\n"
        f"whitelist: {', '.join(sorted(st['whitelist'])) or '—'}\n"
        f"blacklist: {', '.join(sorted(st['blacklist'])) or '—'}\n"
        f"UTC: {utc_now_str()}"
    )
    await u.message.reply_text(text, reply_markup=_kb(st))

# --- сеттери (залишив працюючими, але їх немає у меню)
async def set_noise_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v=float(c.args[0]); assert 0.5<=v<=5
        st["noise"]=v
        await u.message.reply_text(f"OK. Фільтр шуму: {v:.2f}%.")
    except:
        await u.message.reply_text("Формат: /set_noise 1.8  (0.5..5)")

async def set_trend_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st=STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v=int(c.args[0]); assert v in (2,3,4)
        st["trend_weight"]=v
        await u.message.reply_text(f"OK. Суворість тренду: {v}.")
    except:
        await u.message.reply_text("Формат: /set_trend 2|3|4")

async def set_rr_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st=STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v=float(c.args[0]); assert 1.2<=v<=3.0
        st["rr_min"]=v
        await u.message.reply_text(f"OK. Мін. R:R = {v:.2f}.")
    except:
        await u.message.reply_text("Формат: /set_rr 1.8")

async def set_risk_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st=STATE.setdefault(u.effective_chat.id, default_state())
    try:
        slk=float(c.args[0]); tpk=float(c.args[1]); assert 0.5<=slk<=3 and 1.0<=tpk<=5
        st["sl_k"]=slk; st["tp_k"]=tpk
        await u.message.reply_text(f"OK. Auto SL={slk}×ATR, TP={tpk}×ATR.")
    except:
        await u.message.reply_text("Формат: /set_risk 1.5 2.5")

async def set_liq_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st=STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v=float(c.args[0]); assert 20<=v<=1000
        st["min_turnover"]=v
        await u.message.reply_text(f"OK. Мін. обіг 24h = {v:.0f}M USDT.")
    except:
        await u.message.reply_text("Формат: /set_liq 150   (в млн USDT)")

async def set_spread_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st=STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v=int(c.args[0]); assert 1<=v<=30
        st["max_spread_bps"]=v
        await u.message.reply_text(f"OK. Максимальний спред = {v} bps.")
    except:
        await u.message.reply_text("Формат: /set_spread 5  (bps)")

async def set_24h_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st=STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v=float(c.args[0]); assert 5<=v<=50
        st["max_24h_change"]=v
        await u.message.reply_text(f"OK. Максимальна 24hΔ = {v:.1f}%.")
    except:
        await u.message.reply_text("Формат: /set_24h 15")

async def set_session_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st=STATE.setdefault(u.effective_chat.id, default_state())
    try:
        f=int(c.args[0]); t=int(c.args[1]); assert 0<=f<=23 and 0<=t<=23
        st["sess_from"]=f; st["sess_to"]=t
        await u.message.reply_text(f"OK. Сесія UTC {f:02d}-{t:02d}.")
    except:
        await u.message.reply_text("Формат: /set_session 12 20")

async def set_cooldown_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st=STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v=int(c.args[0]); assert 30<=v<=1440
        st["cooldown_min"]=v
        await u.message.reply_text(f"OK. Кулдаун: {v} хв.")
    except:
        await u.message.reply_text("Формат: /set_cooldown 180")

async def wl_add_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st=STATE.setdefault(u.effective_chat.id, default_state())
    syms=[s.strip().upper() for s in c.args]
    for s in syms:
        if s: st["whitelist"].add(s)
    await u.message.reply_text(f"OK. whitelist += {', '.join(syms) or '—'}")

async def wl_clear_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st=STATE.setdefault(u.effective_chat.id, default_state())
    st["whitelist"].clear()
    await u.message.reply_text("OK. whitelist очищено.")

async def bl_add_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st=STATE.setdefault(u.effective_chat.id, default_state())
    syms=[s.strip().upper() for s in c.args]
    for s in syms:
        if s: st["blacklist"].add(s)
    await u.message.reply_text(f"OK. blacklist += {', '.join(syms) or '—'}")

async def bl_clear_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st=STATE.setdefault(u.effective_chat.id, default_state())
    st["blacklist"].clear()
    await u.message.reply_text("OK. blacklist очищено.")

# автопостинг
async def auto_job(ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = ctx.job.data["chat_id"]
    st = STATE.setdefault(chat_id, default_state())
    try:
        txt = await build_signals(st)
        for ch in split_long(txt):
            await ctx.bot.send_message(chat_id=chat_id, text=ch, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        log.error("auto job err: %s", e)

async def auto_on_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st=STATE.setdefault(u.effective_chat.id, default_state())
    minutes = st.get("every", DEFAULT_AUTO_MIN)
    if c.args:
        try: minutes=max(5, min(180, int(c.args[0])))
        except: pass
    await _start_autoposting(u.effective_chat.id, c.application, st, minutes)
    await u.message.reply_text(f"✅ Автопостинг ON кожні {minutes} хв.", reply_markup=_kb(st))

async def auto_off_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st=STATE.setdefault(u.effective_chat.id, default_state())
    st["auto_on"]=False
    name=f"auto_{u.effective_chat.id}"
    for j in c.application.job_queue.get_jobs_by_name(name):
        j.schedule_removal()
    await u.message.reply_text("⏸ Автопостинг OFF.", reply_markup=_kb(st))

# Профіль-команди (нове)
async def scalp_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await _apply_profile_and_scan(u, c, "scalp")

async def default_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await _apply_profile_and_scan(u, c, "default")

async def swing_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await _apply_profile_and_scan(u, c, "swing")

# ========= Main =========
def main():
    if not TG_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN"); return
    app = Application.builder().token(TG_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    # профілі
    app.add_handler(CommandHandler("scalp", scalp_cmd))
    app.add_handler(CommandHandler("default", default_cmd))
    app.add_handler(CommandHandler("swing", swing_cmd))

    # сеттери (залишились доступні вручну, але не у меню)
    app.add_handler(CommandHandler("set_noise", set_noise_cmd))
    app.add_handler(CommandHandler("set_trend", set_trend_cmd))
    app.add_handler(CommandHandler("set_rr", set_rr_cmd))
    app.add_handler(CommandHandler("set_risk", set_risk_cmd))
    app.add_handler(CommandHandler("set_liq", set_liq_cmd))
    app.add_handler(CommandHandler("set_spread", set_spread_cmd))
    app.add_handler(CommandHandler("set_24h", set_24h_cmd))
    app.add_handler(CommandHandler("set_session", set_session_cmd))
    app.add_handler(CommandHandler("set_cooldown", set_cooldown_cmd))

    app.add_handler(CommandHandler("wl_add", wl_add_cmd))
    app.add_handler(CommandHandler("wl_clear", wl_clear_cmd))
    app.add_handler(CommandHandler("bl_add", bl_add_cmd))
    app.add_handler(CommandHandler("bl_clear", bl_clear_cmd))

    app.add_handler(CommandHandler("auto_on", auto_on_cmd))
    app.add_handler(CommandHandler("auto_off", auto_off_cmd))

    app.run_polling()

if __name__ == "__main__":
    main()
