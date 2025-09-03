# -*- coding: utf-8 -*-
# Bybit Signals (NO autotrade) — clean formatted version

import os
import math
import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone, time as dtime

from telegram import Update, ReplyKeyboardMarkup, BotCommand
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# =============== ENV ===============
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BYBIT_PROXY = os.getenv("BYBIT_PROXY", "").strip()

DEFAULT_AUTO_MIN = int(os.getenv("DEFAULT_AUTO_MIN", "15"))
TOP_N = int(os.getenv("TOP_N", "3"))

# =============== LOGS ===============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("signals")

# =============== PROFILES ===============
PROFILES = {
    "scalp": {
        "top_n": 5, "noise": 1.0, "trend_weight": 3, "atr_len": 10,
        "sl_k": 1.2, "rr_k": 2.6, "min_adx": 20, "vol_mult": 1.0,
        "min_turnover": 100.0, "max_spread_bps": 8, "max_24h_change": 25.0,
        "cooldown_min": 60, "every": 15, "trail_k": 1.0,
    },
    "default": {
        "top_n": 3, "noise": 1.6, "trend_weight": 3, "atr_len": 14,
        "sl_k": 1.5, "rr_k": 2.2, "min_adx": 18, "vol_mult": 1.0,
        "min_turnover": 150.0, "max_spread_bps": 6, "max_24h_change": 18.0,
        "cooldown_min": 180, "every": 15, "trail_k": 1.2,
    },
    "swing": {
        "top_n": 3, "noise": 1.2, "trend_weight": 4, "atr_len": 20,
        "sl_k": 2.0, "rr_k": 3.0, "min_adx": 18, "vol_mult": 0.9,
        "min_turnover": 150.0, "max_spread_bps": 12, "max_24h_change": 20.0,
        "cooldown_min": 360, "every": 30, "trail_k": 1.5,
    },
}

# =============== UI ===============
def _kb(_: Dict[str, object]) -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [["/scalp", "/default", "/swing"], ["/signals", "/status", "/help"]],
        resize_keyboard=True,
    )

# =============== STATE ===============
STATE: Dict[int, Dict[str, object]] = {}

def default_state() -> Dict[str, object]:
    return {
        "min_turnover": 150.0,
        "max_spread_bps": 6,
        "max_24h_change": 18.0,
        "whitelist": set(),
        "blacklist": set({"TRUMPUSDT", "PUMPFUNUSDT", "FARTCOINUSDT", "IPUSDT", "ENAUSDT"}),
        "noise": 1.6,
        "trend_weight": 3,
        "min_adx": 18,
        "vol_mult": 1.0,
        "atr_len": 14,
        "sl_k": 1.5,
        "rr_k": 2.2,
        "top_n": TOP_N,
        "every": DEFAULT_AUTO_MIN,
        "auto_on": False,
        "sess_from": 12,
        "sess_to": 20,
        "cooldown_min": 180,
        "_last_sig_ts": {},
        "trail_k": 1.2,
    }

# =============== HELPERS ===============
BASE = "https://api.bybit.com"

def utc_now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

def in_session(st: Dict[str, object]) -> bool:
    now = datetime.now(timezone.utc).time()
    f, t = int(st["sess_from"]), int(st["sess_to"])
    if f <= t:
        return dtime(f, 0) <= now <= dtime(t, 0)
    return now >= dtime(f, 0) or now <= dtime(t, 0)

def _proxy_kwargs() -> dict:
    return {"proxy": BYBIT_PROXY} if BYBIT_PROXY.startswith(("http://", "https://")) else {}

def split_long(text: str, n: int = 3500) -> List[str]:
    out: List[str] = []
    while len(text) > n:
        out.append(text[:n])
        text = text[n:]
    out.append(text)
    return out

# =============== HTTP ===============
async def http_json(session: aiohttp.ClientSession, url: str, params: dict | None = None) -> dict:
    delay = 0.6
    for i in range(5):
        try:
            async with session.get(url, params=params, timeout=25, **_proxy_kwargs()) as r:
                r.raise_for_status()
                return await r.json()
        except Exception:
            if i == 4:
                raise
            await asyncio.sleep(delay)
            delay *= 1.6

async def get_tickers(session) -> List[dict]:
    data = await http_json(session, f"{BASE}/v5/market/tickers", {"category": "linear"})
    return ((data.get("result") or {}).get("list")) or []

async def get_orderbook_spread_bps(session, symbol: str) -> float:
    data = await http_json(session, f"{BASE}/v5/market/orderbook",
                           {"category": "linear", "symbol": symbol, "limit": "1"})
    res = data.get("result") or {}
    bids = res.get("b") or []
    asks = res.get("a") or []
    if not bids or not asks:
        return 9999.0
    bid = float(bids[0][0])
    ask = float(asks[0][0])
    if ask <= 0:
        return 9999.0
    return (ask - bid) / ask * 10000.0  # bps

async def get_klines(session, symbol: str, interval: str, limit: int = 300):
    data = await http_json(
        session,
        f"{BASE}/v5/market/kline",
        {"category": "linear", "symbol": symbol, "interval": interval, "limit": str(limit)},
    )
    lst = list(reversed(((data.get("result") or {}).get("list")) or []))
    opens, highs, lows, closes, volumes = [], [], [], [], []
    for r in lst:
        try:
            opens.append(float(r[1]))
            highs.append(float(r[2]))
            lows.append(float(r[3]))
            closes.append(float(r[4]))
            volumes.append(float(r[5]))
        except Exception:
            pass
    return opens, highs, lows, closes, volumes

# =============== INDICATORS ===============
def ema(xs: List[float], p: int) -> List[float]:
    if not xs:
        return []
    k = 2 / (p + 1)
    out = [xs[0]]
    for x in xs[1:]:
        out.append(out[-1] + k * (x - out[-1]))
    return out

def sma_series(xs: List[float], p: int) -> List[Optional[float]]:
    if p <= 0:
        return []
    out: List[Optional[float]] = [None] * (p - 1)
    if len(xs) < p:
        return out
    s = sum(xs[:p])
    out.append(s / p)
    for i in range(p, len(xs)):
        s += xs[i] - xs[i - p]
        out.append(s / p)
    return out

def rsi(xs: List[float], p: int = 14) -> List[float]:
    if len(xs) < p + 1:
        return []
    gains, losses = [], []
    for i in range(1, len(xs)):
        d = xs[i] - xs[i - 1]
        gains.append(max(0.0, d))
        losses.append(max(0.0, -d))
    ag = sum(gains[:p]) / p
    al = sum(losses[:p]) / p
    out = [0.0] * p
    out.append(100.0 if al == 0 else 100 - 100 / (1 + ag / (al + 1e-9)))
    for i in range(p, len(gains)):
        ag = (ag * (p - 1) + gains[i]) / p
        al = (al * (p - 1) + losses[i]) / p
        out.append(100.0 if al == 0 else 100 - 100 / (1 + ag / (al + 1e-9)))
    return out

def macd(xs: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float]]:
    if len(xs) < slow + signal:
        return [], []
    ef, es = ema(xs, fast), ema(xs, slow)
    m = [a - b for a, b in zip(ef[-len(es):], es)]
    s = ema(m, signal)
    L = min(len(m), len(s))
    return m[-L:], s[-L:]

def atr(high: List[float], low: List[float], close: List[float], n: int = 14) -> float:
    if len(close) < n + 1 or not (len(high) == len(low) == len(close)):
        return 0.0
    trs: List[float] = []
    for i in range(1, len(close)):
        trs.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
    if len(trs) < n:
        return 0.0
    return sum(trs[-n:]) / n

def adx_last(high: List[float], low: List[float], close: List[float], n: int = 14) -> float:
    if len(close) < n + 1:
        return 0.0
    plus_dm, minus_dm, tr = [], [], []
    for i in range(1, len(close)):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm.append(up if (up > dn and up > 0) else 0.0)
        minus_dm.append(dn if (dn > up and dn > 0) else 0.0)
        tr.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
    # просте SMA-згладжування
    def rmean(xs: List[float], p: int) -> List[float]:
        if len(xs) < p:
            return []
        out: List[float] = []
        s = sum(xs[:p])
        out.append(s / p)
        for i in range(p, len(xs)):
            s += xs[i] - xs[i - p]
            out.append(s / p)
        return out
    atr_n = rmean(tr, n)
    if not atr_n:
        return 0.0
    last = len(atr_n) - 1
    atrv = atr_n[last] or 1e-9
    pdi = 100.0 * (sum(plus_dm[-n:]) / n) / atrv
    mdi = 100.0 * (sum(minus_dm[-n:]) / n) / atrv
    dx = 100.0 * abs(pdi - mdi) / (pdi + mdi + 1e-9)
    # ще раз згладимо
    dx_series = rmean([dx] * n, n)
    return dx_series[-1] if dx_series else dx

# =============== SCORING ===============
def votes_from_series(closes: List[float]) -> Dict[str, float]:
    out = {"vote": 0, "rsi": None, "ema_trend": 0, "macd": None, "sig": None}
    if len(closes) < 60:
        return out
    rr = rsi(closes, 14)
    m, s = macd(closes)
    e50 = ema(closes, 50)
    e200 = ema(closes, 200 if len(closes) >= 200 else max(100, len(closes) // 2))
    if rr:
        out["rsi"] = rr[-1]
        if rr[-1] <= 30:
            out["vote"] += 1
        if rr[-1] >= 70:
            out["vote"] -= 1
    if m and s:
        out["macd"], out["sig"] = m[-1], s[-1]
        out["vote"] += 1 if m[-1] > s[-1] else -1
    if e50 and e200:
        et = 1 if e50[-1] > e200[-1] else -1
        out["ema_trend"] = et
        out["vote"] += 1 if et == 1 else -1
    return out

def decide_direction(v15: int, v30: int, v60: int, need: int) -> Optional[str]:
    total = v15 + v30 + v60
    pos = sum(1 for v in (v15, v30, v60) if v > 0)
    neg = sum(1 for v in (v15, v30, v60) if v < 0)
    if total >= need and pos >= 2:
        return "LONG"
    if total <= -need and neg >= 2:
        return "SHORT"
    return None

# =============== SIGNALS ===============
async def build_signals(st: Dict[str, object]) -> str:
    if not in_session(st):
        return f"⏳ Поза торговою сесією (UTC {st['sess_from']:02d}:00–{st['sess_to']:02d}:00)."

    last_ts: Dict[str, float] = st["_last_sig_ts"]
    now_ts = datetime.utcnow().timestamp()

    async with aiohttp.ClientSession() as s:
        tickers = await get_tickers(s)

        # первинний відбір
        cands: List[Tuple[str, float, float]] = []
        for t in tickers:
            sym = str(t.get("symbol", ""))
            if st["whitelist"] and sym not in st["whitelist"]:
                continue
            if sym in st["blacklist"]:
                continue
            try:
                vol = float(t.get("turnover24h") or 0.0) / 1e6
                ch24 = float(t.get("price24hPcnt") or 0.0) * 100.0
                px = float(t.get("lastPrice") or 0.0)
            except Exception:
                continue
            if vol < float(st["min_turnover"]):
                continue
            if abs(ch24) > float(st["max_24h_change"]):
                continue
            if px <= 0:
                continue
            cands.append((sym, px, ch24))

        scored = []
        for sym, px, ch24 in cands:
            sp_bps = await get_orderbook_spread_bps(s, sym)
            if sp_bps > float(st["max_spread_bps"]):
                continue

            o15, h15, l15, c15, v15 = await get_klines(s, sym, "15", 300)
            await asyncio.sleep(0.12)
            o30, h30, l30, c30, v30 = await get_klines(s, sym, "30", 300)
            await asyncio.sleep(0.12)
            o60, h60, l60, c60, v60 = await get_klines(s, sym, "60", 300)
            if not (c15 and c30 and c60):
                continue

            # обсяг: остання 15m свіча > SMA20 * vol_mult
            vol_sma20 = sma_series(v15, 20)
            if vol_sma20 and vol_sma20[-1] is not None:
                if v15[-1] <= float(st["vol_mult"]) * float(vol_sma20[-1]):
                    continue

            v15m = votes_from_series(c15)
            v30m = votes_from_series(c30)
            v60m = votes_from_series(c60)
            direction = decide_direction(v15m["vote"], v30m["vote"], v60m["vote"], int(st["trend_weight"]))
            if not direction:
                continue

            atr_val = atr(h15, l15, c15, int(st["atr_len"]))
            if atr_val <= 0:
                continue

            adx30 = adx_last(h30, l30, c30, 14)
            adx60 = adx_last(h60, l60, c60, 14)
            if min(adx30, adx60) < float(st["min_adx"]):
                continue

            # мінімальний потенціал руху через ATR як % від ціни
            noise_pct = 100.0 * (atr_val / px)
            if noise_pct < float(st["noise"]):
                continue

            sl_k = float(st["sl_k"])
            rr_k = float(st["rr_k"])
            if direction == "LONG":
                sl = px - sl_k * atr_val
                risk = px - sl
                tp = px + rr_k * risk
            else:
                sl = px + sl_k * atr_val
                risk = sl - px
                tp = px - rr_k * risk

            # кулдаун
            if now_ts - last_ts.get(sym, 0.0) < float(st["cooldown_min"]) * 60.0:
                continue

            score = (v15m["vote"] + v30m["vote"] + v60m["vote"])
            if v60m["ema_trend"] == 1 and direction == "LONG":
                score += 1
            if v60m["ema_trend"] == -1 and direction == "SHORT":
                score += 1

            scored.append(
                (score, sym, direction, px, sl, tp, sp_bps, ch24, atr_val, noise_pct, adx30, adx60, (v15m, v30m, v60m))
            )

        if not scored:
            return "⚠️ Якісних сигналів не знайдено (фільтри відсіяли все)."

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max(1, int(st["top_n"]))]

        for _, sym, *_ in top:
            st["_last_sig_ts"][sym] = datetime.utcnow().timestamp()

        def mark(v: Dict[str, float]) -> str:
            r = v["rsi"]
            rtxt = f"{r:.0f}" if isinstance(r, (int, float)) else "-"
            m = v["macd"]
            s = v["sig"]
            mtxt = "↑" if (m is not None and s is not None and m > s) else ("↓" if (m is not None and s is not None and m < s) else "·")
            et = v["ema_trend"]
            etxt = "↑" if et == 1 else ("↓" if et == -1 else "·")
            return f"RSI:{rtxt} MACD:{mtxt} EMA:{etxt}"

        body: List[str] = []
        for sc, sym, direc, px, sl, tp, sp_bps, ch24, atr_v, noise_pct, adx30, adx60, (v15m, v30m, v60m) in top:
            rr = abs(tp - px) / max(1e-9, abs(px - sl))
            body.append(
                f"• *{sym}*: *{direc}* @ `{px:.6f}`\n"
                f"  SL:`{sl:.6f}` · TP:`{tp:.6f}` · ATR:`{atr_v:.6f}` · RR:`{rr:.2f}`\n"
                f"  spread:{sp_bps:.2f}bps · 24hΔ:{ch24:+.2f}% · ATR%≈{noise_pct:.2f}% · ADX30:{adx30:.0f} ADX1h:{adx60:.0f}\n"
                f"  15m {mark(v15m)} | 30m {mark(v30m)} | 1h {mark(v60m)}\n"
                f"  Менеджмент: +0.5R → SL=BE; далі трейл {st['trail_k']}×ATR."
            )
        return "📈 *Сильні сигнали:*\n\n" + "\n\n".join(body) + f"\n\nUTC: {utc_now_str()}"

# =============== AUTO HELPERS ===============
async def _start_autoposting(chat_id: int, app: Application, st: Dict[str, object], minutes: int) -> None:
    st["every"] = minutes
    st["auto_on"] = True
    name = f"auto_{chat_id}"
    for j in app.job_queue.get_jobs_by_name(name):
        j.schedule_removal()
    app.job_queue.run_repeating(auto_job, interval=minutes * 60, first=5, name=name, data={"chat_id": chat_id})

async def _scan_now_and_send(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = STATE.setdefault(chat_id, default_state())
    txt = await build_signals(st)
    for ch in split_long(txt):
        await context.bot.send_message(chat_id=chat_id, text=ch, parse_mode=ParseMode.MARKDOWN)

async def _apply_profile_and_scan(u: Update, c: ContextTypes.DEFAULT_TYPE, key: str) -> None:
    st = STATE.setdefault(u.effective_chat.id, default_state())
    p = PROFILES[key]
    st.update({
        "top_n": p["top_n"], "noise": p["noise"], "trend_weight": p["trend_weight"],
        "atr_len": p["atr_len"], "sl_k": p["sl_k"], "rr_k": p["rr_k"],
        "min_turnover": p["min_turnover"], "max_spread_bps": p["max_spread_bps"],
        "max_24h_change": p["max_24h_change"], "cooldown_min": p["cooldown_min"],
        "min_adx": p["min_adx"], "vol_mult": p["vol_mult"], "trail_k": p["trail_k"],
    })
    await _start_autoposting(u.effective_chat.id, c.application, st, p["every"])
    await u.message.reply_text(
        f"✅ Профіль *{key}* застосовано. Автоскан кожні {p['every']} хв.",
        parse_mode=ParseMode.MARKDOWN, reply_markup=_kb(st),
    )
    await _scan_now_and_send(u.effective_chat.id, c)

# =============== COMMANDS ===============
async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    STATE.setdefault(u.effective_chat.id, default_state())
    try:
        await c.bot.set_my_commands([
            BotCommand("help", "Довідка по командам"),
            BotCommand("scalp", "Агресивний режим (скальпінг)"),
            BotCommand("default", "Стандартний режим"),
            BotCommand("swing", "Середньостроковий режим"),
            BotCommand("signals", "Сканувати зараз"),
            BotCommand("status", "Поточні налаштування"),
        ])
    except Exception:
        pass
    await u.message.reply_text(
        "👋 Готово. Бот видає *сигнали без автотрейду*. Обери режим нижче.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=_kb({}),
    )

async def help_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "📘 <b>Довідка по командам</b>\n\n"
        "🔎 <b>Основні</b>\n"
        "/start — запуск і меню\n"
        "/status — налаштування\n"
        "/signals — сканувати ринок зараз\n\n"
        "⚙️ <b>Параметри</b>\n"
        "/set_top N — монет у сигналі\n"
        "/set_noise X — мін. ATR% (0.5..5)\n"
        "/set_trend 2|3|4 — суворість тренду\n"
        "/set_atr N — довжина ATR (7..50)\n"
        "/set_slk X — SL множник у ATR (0.5..3)\n"
        "/set_rrk X — цільовий R:R (1.2..5)\n"
        "/set_adx N — мін. ADX (5..50)\n"
        "/set_vol X — множник обсягу до SMA20 (0.5..3)\n"
        "/set_liq N — мін. обіг 24h, млн USDT\n"
        "/set_spread N — макс. спред, bps\n"
        "/set_24h N — макс. денний рух, %\n"
        "/set_cooldown N — пауза між сигналами, хв\n"
        "/set_session F T — торговий час UTC\n\n"
        "🤖 <b>Автопостинг</b>\n"
        "/auto_on N — кожні N хв\n"
        "/auto_off — вимкнути\n\n"
        "🎛 Профілі: /scalp /default /swing\n"
        "Менеджмент: +0.5R → SL=BE; далі трейл X×ATR."
    )
    for chunk in split_long(help_text):
        await u.message.reply_text(chunk, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

async def signals_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await _scan_now_and_send(u.effective_chat.id, c)

async def status_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    text = (
        f"Автопостинг: {'ON' if st['auto_on'] else 'OFF'} кожні {st['every']} хв\n"
        f"TOP_N={st['top_n']} · noise≈{st['noise']}% · trend_weight={st['trend_weight']}\n"
        f"ATR(len={st['atr_len']}) · SL_k={st['sl_k']} · RR_k={st['rr_k']} · "
        f"minADX={st['min_adx']} · volMult={st['vol_mult']}\n"
        f"turnover≥{st['min_turnover']}M · spread≤{st['max_spread_bps']}bps · 24hΔ≤{st['max_24h_change']}%\n"
        f"session UTC {st['sess_from']:02d}-{st['sess_to']:02d} · cooldown={st['cooldown_min']}м\n"
        f"whitelist: {', '.join(sorted(st['whitelist'])) or '—'}\n"
        f"blacklist: {', '.join(sorted(st['blacklist'])) or '—'}\n"
        f"UTC: {utc_now_str()}"
    )
    await u.message.reply_text(text, reply_markup=_kb(st))

# setters
async def _setter(u: Update, ok: bool, msg_ok: str, msg_err: str):
    await u.message.reply_text(msg_ok if ok else msg_err)

async def set_top_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v = int(c.args[0]); assert 1 <= v <= 10; st["top_n"] = v
        await _setter(u, True, f"OK. TOP_N = {v}.", "Формат: /set_top 3 (1..10)")
    except Exception:
        await _setter(u, False, "", "Формат: /set_top 3 (1..10)")

async def set_noise_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v = float(c.args[0]); assert 0.5 <= v <= 5; st["noise"] = v
        await _setter(u, True, f"OK. Мін. ATR%: {v:.2f}%.", "Формат: /set_noise 1.6")
    except Exception:
        await _setter(u, False, "", "Формат: /set_noise 1.6")

async def set_trend_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v = int(c.args[0]); assert v in (2, 3, 4); st["trend_weight"] = v
        await _setter(u, True, f"OK. Суворість тренду: {v}.", "Формат: /set_trend 2|3|4")
    except Exception:
        await _setter(u, False, "", "Формат: /set_trend 2|3|4")

async def set_atr_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v = int(c.args[0]); assert 7 <= v <= 50; st["atr_len"] = v
        await _setter(u, True, f"OK. ATR довжина = {v}.", "Формат: /set_atr 14")
    except Exception:
        await _setter(u, False, "", "Формат: /set_atr 14")

async def set_slk_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v = float(c.args[0]); assert 0.5 <= v <= 3.0; st["sl_k"] = v
        await _setter(u, True, f"OK. SL = {v}×ATR.", "Формат: /set_slk 1.5")
    except Exception:
        await _setter(u, False, "", "Формат: /set_slk 1.5")

async def set_rrk_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v = float(c.args[0]); assert 1.2 <= v <= 5.0; st["rr_k"] = v
        await _setter(u, True, f"OK. TP = {v:.2f}R.", "Формат: /set_rrk 2.6")
    except Exception:
        await _setter(u, False, "", "Формат: /set_rrk 2.6")

async def set_adx_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v = int(c.args[0]); assert 5 <= v <= 50; st["min_adx"] = v
        await _setter(u, True, f"OK. Мін. ADX = {v}.", "Формат: /set_adx 20")
    except Exception:
        await _setter(u, False, "", "Формат: /set_adx 20")

async def set_vol_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v = float(c.args[0]); assert 0.5 <= v <= 3.0; st["vol_mult"] = v
        await _setter(u, True, f"OK. Обсяг > {v:.2f}×SMA20.", "Формат: /set_vol 1.0")
    except Exception:
        await _setter(u, False, "", "Формат: /set_vol 1.0")

async def set_liq_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v = float(c.args[0]); assert 20 <= v <= 2000; st["min_turnover"] = v
        await _setter(u, True, f"OK. Мін. обіг 24h = {v:.0f}M.", "Формат: /set_liq 150")
    except Exception:
        await _setter(u, False, "", "Формат: /set_liq 150")

async def set_spread_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v = int(c.args[0]); assert 1 <= v <= 50; st["max_spread_bps"] = v
        await _setter(u, True, f"OK. Макс. спред = {v} bps.", "Формат: /set_spread 6")
    except Exception:
        await _setter(u, False, "", "Формат: /set_spread 6")

async def set_24h_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v = float(c.args[0]); assert 5 <= v <= 50; st["max_24h_change"] = v
        await _setter(u, True, f"OK. 24hΔ ≤ {v:.1f}%.", "Формат: /set_24h 18")
    except Exception:
        await _setter(u, False, "", "Формат: /set_24h 18")

async def set_session_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    try:
        f = int(c.args[0]); t = int(c.args[1]); assert 0 <= f <= 23 and 0 <= t <= 23
        st["sess_from"] = f; st["sess_to"] = t
        await _setter(u, True, f"OK. Сесія UTC {f:02d}-{t:02d}.", "Формат: /set_session 12 20")
    except Exception:
        await _setter(u, False, "", "Формат: /set_session 12 20")

async def set_cooldown_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    try:
        v = int(c.args[0]); assert 15 <= v <= 1440; st["cooldown_min"] = v
        await _setter(u, True, f"OK. Кулдаун: {v} хв.", "Формат: /set_cooldown 180")
    except Exception:
        await _setter(u, False, "", "Формат: /set_cooldown 180")

# whitelist / blacklist
async def wl_add_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    syms = [s.strip().upper() for s in c.args]
    for s in syms:
        if s:
            st["whitelist"].add(s)
    await u.message.reply_text(f"OK. whitelist += {', '.join(syms) or '—'}")

async def wl_clear_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    st["whitelist"].clear()
    await u.message.reply_text("OK. whitelist очищено.")

async def bl_add_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    syms = [s.strip().upper() for s in c.args]
    for s in syms:
        if s:
            st["blacklist"].add(s)
    await u.message.reply_text(f"OK. blacklist += {', '.join(syms) or '—'}")

async def bl_clear_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    st["blacklist"].clear()
    await u.message.reply_text("OK. blacklist очищено.")

# =============== AUTOPost ===============
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
    st = STATE.setdefault(u.effective_chat.id, default_state())
    minutes = st.get("every", DEFAULT_AUTO_MIN)
    if c.args:
        try:
            minutes = max(5, min(180, int(c.args[0])))
        except Exception:
            pass
    await _start_autoposting(u.effective_chat.id, c.application, st, minutes)
    await u.message.reply_text(f"✅ Автопостинг ON кожні {minutes} хв.", reply_markup=_kb(st))

async def auto_off_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    st["auto_on"] = False
    name = f"auto_{u.effective_chat.id}"
    for j in c.application.job_queue.get_jobs_by_name(name):
        j.schedule_removal()
    await u.message.reply_text("⏸ Автопостинг OFF.", reply_markup=_kb(st))

# профілі
async def scalp_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await _apply_profile_and_scan(u, c, "scalp")

async def default_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await _apply_profile_and_scan(u, c, "default")

async def swing_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await _apply_profile_and_scan(u, c, "swing")

# =============== MAIN ===============
def main():
    if not TG_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN")
        return

    app = Application.builder().token(TG_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    # profiles
    app.add_handler(CommandHandler("scalp", scalp_cmd))
    app.add_handler(CommandHandler("default", default_cmd))
    app.add_handler(CommandHandler("swing", swing_cmd))

    # setters
    app.add_handler(CommandHandler("set_top", set_top_cmd))
    app.add_handler(CommandHandler("set_noise", set_noise_cmd))
    app.add_handler(CommandHandler("set_trend", set_trend_cmd))
    app.add_handler(CommandHandler("set_atr", set_atr_cmd))
    app.add_handler(CommandHandler("set_slk", set_slk_cmd))
    app.add_handler(CommandHandler("set_rrk", set_rrk_cmd))
    app.add_handler(CommandHandler("set_adx", set_adx_cmd))
    app.add_handler(CommandHandler("set_vol", set_vol_cmd))
    app.add_handler(CommandHandler("set_liq", set_liq_cmd))
    app.add_handler(CommandHandler("set_spread", set_spread_cmd))
    app.add_handler(CommandHandler("set_24h", set_24h_cmd))
    app.add_handler(CommandHandler("set_session", set_session_cmd))
    app.add_handler(CommandHandler("set_cooldown", set_cooldown_cmd))

    # lists + auto
    app.add_handler(CommandHandler("wl_add", wl_add_cmd))
    app.add_handler(CommandHandler("wl_clear", wl_clear_cmd))
    app.add_handler(CommandHandler("bl_add", bl_add_cmd))
    app.add_handler(CommandHandler("bl_clear", bl_clear_cmd))
    app.add_handler(CommandHandler("auto_on", auto_on_cmd))
    app.add_handler(CommandHandler("auto_off", auto_off_cmd))

    app.run_polling()

if __name__ == "__main__":
    main()
