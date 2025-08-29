# bot_clean.py
# -*- coding: utf-8 -*-

import os, math, time, aiohttp
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta

from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ===== Налаштування =====
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# хто отримує heartbeat
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
# раз на скільки хвилин надсилати heartbeat
HEARTBEAT_MIN = int(os.getenv("HEARTBEAT_MIN", "60"))

# CoinGecko: топ-120 монет з ціною, sparkline та % за 24h
MARKET_URL = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=market_cap_desc&per_page=120&page=1"
    "&sparkline=true&price_change_percentage=24h"
)

# стейбли — виключаємо
STABLES = {"USDT","USDC","DAI","TUSD","FDUSD","USDD","PYUSD","EURS","EURT","BUSD"}

# простий in-memory стейт: chat_id -> {"auto_on": bool, "every": int}
STATE: Dict[int, Dict[str, int | bool]] = {}

# ===== Технічні індикатори =====
def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1:
        return series[:]
    k = 2 / (period + 1)
    out = [series[0]]
    for x in series[1:]:
        out.append(out[-1] + k * (x - out[-1]))
    return out

def rsi(series: List[float], period: int = 14) -> List[float]:
    if len(series) < period + 1:
        return []
    gains, losses = [], []
    for i in range(1, len(series)):
        ch = series[i] - series[i-1]
        gains.append(max(0.0, ch))
        losses.append(max(0.0, -ch))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [0.0] * (period)
    if avg_loss == 0:
        rsis.append(100.0)
    else:
        rsis.append(100.0 - (100.0 / (1.0 + (avg_gain/(avg_loss+1e-9)))))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rs = avg_gain/(avg_loss+1e-9)
            rsis.append(100.0 - (100.0/(1.0+rs)))
    return rsis

def macd(series: List[float], fast:int=12, slow:int=26, signal:int=9) -> Tuple[List[float], List[float]]:
    if len(series) < slow + signal:
        return [], []
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = [a-b for a,b in zip(ema_fast[-len(ema_slow):], ema_slow)]
    sig = ema(macd_line, signal)
    L = min(len(macd_line), len(sig))
    return macd_line[-L:], sig[-L:]

# ===== Логіка оцінки монети =====
def decide_signal(prices: List[float], p24: Optional[float]) -> Tuple[str, float, float, str]:
    """Повертає: (direction 'LONG/SHORT/NONE', sl_price, tp_price, пояснення)"""
    explain: List[str] = []
    if not prices or len(prices) < 40:
        return "NONE", 0.0, 0.0, "недостатньо даних"

    series = prices[-120:]
    px = series[-1]

    ema50 = ema(series, 50)
    ema200 = ema(series, min(200, len(series)//2 if len(series) >= 200 else 100))
    trend = 0
    if len(ema50) and len(ema200):
        if ema50[-1] > ema200[-1]: trend = 1
        elif ema50[-1] < ema200[-1]: trend = -1

    rsi15 = rsi(series, 7)
    rsi30 = rsi(series, 14)
    rsi60 = rsi(series, 28)
    macd_line, macd_sig = macd(series)

    votes = 0
    def rsi_vote(last: float) -> int:
        if last is None: return 0
        if last <= 30: return +1
        if last >= 70: return -1
        return 0

    if rsi15: votes += rsi_vote(rsi15[-1]); explain.append(f"RSI15={rsi15[-1]:.1f}{'→L' if rsi15[-1]<=30 else '→S' if rsi15[-1]>=70 else ''}")
    if rsi30: votes += rsi_vote(rsi30[-1]); explain.append(f"RSI30={rsi30[-1]:.1f}{'→L' if rsi30[-1]<=30 else '→S' if rsi30[-1]>=70 else ''}")
    if rsi60: votes += rsi_vote(rsi60[-1]); explain.append(f"RSI60={rsi60[-1]:.1f}{'→L' if rsi60[-1]<=30 else '→S' if rsi60[-1]>=70 else ''}")

    if macd_line and macd_sig:
        if macd_line[-1] > macd_sig[-1]:
            votes += 1; explain.append("MACD↑")
        elif macd_line[-1] < macd_sig[-1]:
            votes -= 1; explain.append("MACD↓")

    if trend > 0: votes += 1; explain.append("Trend=UP")
    elif trend < 0: votes -= 1; explain.append("Trend=DOWN")

    direction = "NONE"
    if votes >= 2: direction = "LONG"
    elif votes <= -2: direction = "SHORT"

    tail = series[-48:] if len(series) >= 48 else series
    if len(tail) >= 2:
        mean = sum(tail)/len(tail)
        var = sum((x-mean)**2 for x in tail)/len(tail)
        stdev = math.sqrt(var)
        vol_pct = (stdev/px)*100.0
    else:
        vol_pct = 1.0

    ctx = abs(p24 or 0.0)
    base_sl_pct = max(0.6, min(3.0, 0.7*vol_pct + ctx/2.5))
    base_tp_pct = max(0.8, min(5.0, 1.2*vol_pct + ctx/2.0))

    if direction == "LONG":
        sl_price = px * (1 - base_sl_pct/100.0)
        tp_price = px * (1 + base_tp_pct/100.0)
    elif direction == "SHORT":
        sl_price = px * (1 + base_sl_pct/100.0)
        tp_price = px * (1 - base_tp_pct/100.0)
    else:
        sl_price = 0.0; tp_price = 0.0

    return direction, round(sl_price, 6), round(tp_price, 6), " | ".join(explain)

# ===== Завантаження ринку =====
async def fetch_market(session: aiohttp.ClientSession) -> List[dict]:
    async with session.get(MARKET_URL, timeout=25) as r:
        r.raise_for_status()
        return await r.json()

def is_good_symbol(item: dict) -> bool:
    sym = (item.get("symbol") or "").upper()
    name = (item.get("name") or "").upper()
    if sym in STABLES or any(s in name for s in STABLES):
        return False
    return True

# ===== Формування тексту сигналів =====
async def build_signals_text(top_n: int = 3) -> str:
    async with aiohttp.ClientSession() as s:
        try:
            market = await fetch_market(s)
        except Exception as e:
            return f"⚠️ Помилка ринку: {e}"

    candidates = [m for m in market if is_good_symbol(m)]
    if not candidates:
        return "⚠️ Ринок недоступний або немає кандидатів."

    scored = []
    for it in candidates:
        prices = (((it.get("sparkline_in_7d") or {}).get("price")) or [])
        p24 = it.get("price_change_percentage_24h")
        direction, sl, tp, note = decide_signal(prices, p24)
        score = 0
        if direction in ("LONG", "SHORT"):
            score = 2
        if p24 is not None:
            score += min(2, abs(p24)/10.0)
        scored.append((score, direction, sl, tp, note, it))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [z for z in scored if z[1] != "NONE"][:top_n]
    if not top:
        return "⚠️ Зараз сильних підтверджених сигналів немає."

    lines: List[str] = []
    for _, direction, sl, tp, note, it in top:
        sym = (it.get("symbol") or "").upper()
        px = it.get("current_price")
        p24 = it.get("price_change_percentage_24h") or 0.0
        lines.append(
            f"• {sym}: *{direction}* @ {px}\n"
            f"  SL: `{sl}` · TP: `{tp}` · 24h: {p24:.2f}%\n"
            f"  {note}\n"
        )
    return "📈 *Сильні сигнали:*\n\n" + "\n".join(lines)

# ===== Команди =====
KB = ReplyKeyboardMarkup(
    [["/signals", "/auto_on 15"], ["/auto_off", "/status"]],
    resize_keyboard=True
)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    STATE.setdefault(chat_id, {"auto_on": False, "every": 15})
    await update.message.reply_text(
        "👋 Готовий!\n\nКоманди:\n"
        "• /signals — сканувати зараз\n"
        "• /auto_on 15 — автопуш кожні 15 хв (5–120)\n"
        "• /auto_off — вимкнути автопуш\n"
        "• /status — стан",
        reply_markup=KB
    )

async def signals_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = await build_signals_text()
    for chunk in split_long(txt):
        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

async def auto_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = STATE.setdefault(chat_id, {"auto_on": False, "every": 15})
    minutes = 15
    if context.args:
        try:
            minutes = max(5, min(120, int(context.args[0])))
        except:  # noqa
            pass
    st["auto_on"] = True
    st["every"] = minutes

    name = f"auto_{chat_id}"
    for j in context.application.job_queue.get_jobs_by_name(name):
        j.schedule_removal()
    context.application.job_queue.run_repeating(
        auto_push_job,
        interval=minutes*60,
        first=5,
        name=name,
        data={"chat_id": chat_id}
    )
    await update.message.reply_text(f"Автопуш увімкнено: кожні {minutes} хв.", reply_markup=KB)

async def auto_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = STATE.setdefault(chat_id, {"auto_on": False, "every": 15})
    st["auto_on"] = False
    name = f"auto_{chat_id}"
    for j in context.application.job_queue.get_jobs_by_name(name):
        j.schedule_removal()
    await update.message.reply_text("Автопуш вимкнено.", reply_markup=KB)

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = STATE.setdefault(chat_id, {"auto_on": False, "every": 15})
    await update.message.reply_text(f"Статус: {'ON' if st['auto_on'] else 'OFF'} · кожні {st['every']} хв.")

# ===== Автопуш =====
async def auto_push_job(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    st = STATE.get(chat_id, {})
    if not st or not st.get("auto_on"):
        return
    try:
        txt = await build_signals_text()
        for chunk in split_long(txt):
            await context.bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Автопуш помилка: {e}")

# ===== Утиліти =====
def split_long(text: str, chunk_len: int = 3500) -> List[str]:
    if not text:
        return [""]
    chunks = []
    while len(text) > chunk_len:
        chunks.append(text[:chunk_len])
        text = text[chunk_len:]
    chunks.append(text)
    return chunks

# ===== Heartbeat =====
START_TS = time.time()

def _fmt_uptime() -> str:
    secs = int(time.time() - START_TS)
    d, r = divmod(secs, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    parts = []
    if d: parts.append(f"{d}d")
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)

async def heartbeat(_: ContextTypes.DEFAULT_TYPE):
    if not ADMIN_ID:
        return
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    msg = (
        "✅ Bot is alive\n"
        f"⏱ Uptime: {_fmt_uptime()}\n"
        f"🕒 UTC: {now}"
    )
    try:
        await _.bot.send_message(chat_id=ADMIN_ID, text=msg)
    except Exception as e:
        print(f"[heartbeat] send failed: {e}")

async def setup_jobs(app: Application):
    app.job_queue.run_repeating(
        heartbeat,
        interval=timedelta(minutes=HEARTBEAT_MIN),
        first=10,
        name="heartbeat",
    )

# ===== Main =====
def build_app() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

    print("Bot running | BASE=CoinGecko")
    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(setup_jobs)
        .build()
    )

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))
    app.add_handler(CommandHandler("auto_on", auto_on_cmd))
    app.add_handler(CommandHandler("auto_off", auto_off_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    return app

def main():
    while True:
        try:
            app = build_app()
            app.run_polling(drop_pending_updates=True)
            print("[STOP] Polling finished. Restarting in 5s…")
        except Exception as e:
            print(f"[CRASH] {e}. Restarting in 5s…")
        time.sleep(5)

if __name__ == "__main__":
    main()
