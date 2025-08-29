# -*- coding: utf-8 -*-
# complete paid bot (CoinGecko signals + TRC20 subscription)

import os, math, sqlite3, asyncio, aiohttp
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta

from telegram import (
    Update, ReplyKeyboardMarkup,
    InlineKeyboardButton, InlineKeyboardMarkup
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes, CallbackQueryHandler
)

# ===================== ENV =====================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_ID           = int(os.getenv("ADMIN_ID", "0"))

WALLET_ADDRESS     = os.getenv("WALLET_ADDRESS", "").strip()     # TRC20
TRON_API_KEY       = os.getenv("TRON_API_KEY", "").strip()
SUB_PRICE          = float(os.getenv("SUBSCRIPTION_PRICE", "25"))
SUB_DAYS           = int(os.getenv("SUBSCRIPTION_DAYS", "30"))
MIN_AMOUNT_USDT    = float(os.getenv("MIN_AMOUNT_USDT", "25"))

HEARTBEAT_MIN      = int(os.getenv("HEARTBEAT_MIN", "60"))
USDT_TRON_CONTRACT = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"

# ================== MARKET API =================
MARKET_URL = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=market_cap_desc&per_page=120&page=1"
    "&sparkline=true&price_change_percentage=24h"
)
STABLES = {"USDT","USDC","DAI","TUSD","FDUSD","USDD","PYUSD","EURS","EURT","BUSD"}

STATE: Dict[int, Dict[str, int | bool]] = {}  # chat_id -> {"auto_on": bool, "every": int}

# ================= INDICATORS ==================
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

def decide_signal(prices: List[float], p24: Optional[float]) -> Tuple[str, float, float, str]:
    explain: List[str] = []
    if not prices or len(prices) < 40:
        return "NONE", 0.0, 0.0, "недостатньо даних"

    series = prices[-120:]
    px = series[-1]

    ema50 = ema(series, 50)
    ema200 = ema(series, min(200, len(series)//2 if len(series) >= 200 else 100))
    trend = 0
    if ema50 and ema200:
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

    if rsi15: votes += rsi_vote(rsi15[-1]); explain.append(f"RSI15={rsi15[-1]:.1f}")
    if rsi30: votes += rsi_vote(rsi30[-1]); explain.append(f"RSI30={rsi30[-1]:.1f}")
    if rsi60: votes += rsi_vote(rsi60[-1]); explain.append(f"RSI60={rsi60[-1]:.1f}")

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
        vol_pct = (stdev/px) * 100.0
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

async def build_signals_text(top_n: int = 3) -> str:
    text_lines: List[str] = []
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
        if direction in ("LONG", "SHORT"): score = 2
        if p24 is not None:
            score += min(2, abs(p24)/10.0)

        scored.append((score, direction, sl, tp, note, it))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [z for z in scored if z[1] != "NONE"][:top_n]
    if not top:
        return "⚠️ Зараз сильних підтверджених сигналів немає."

    for _, direction, sl, tp, note, it in top:
        sym = (it.get("symbol") or "").upper()
        px = it.get("current_price")
        p24 = it.get("price_change_percentage_24h") or 0.0
        text_lines.append(
            f"• {sym}: *{direction}* @ {px}\n"
            f"  SL: `{sl}` · TP: `{tp}` · 24h: {p24:.2f}%\n"
            f"  {note}\n"
        )

    return "📈 *Сильні сигнали:*\n\n" + "\n".join(text_lines)

# ================= SUBS STORAGE =================
DB_PATH = "subs.db"

def _db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def subs_init():
    con = _db(); cur = con.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS subs(
        user_id INTEGER PRIMARY KEY,
        expires_at TEXT
      )
    """)
    con.commit(); con.close()

def sub_set(user_id: int, days: int = SUB_DAYS):
    until = datetime.utcnow() + timedelta(days=days)
    con = _db(); cur = con.cursor()
    cur.execute("""INSERT INTO subs(user_id, expires_at) VALUES(?,?)
                   ON CONFLICT(user_id) DO UPDATE SET expires_at=?""",
                (user_id, until.isoformat(), until.isoformat()))
    con.commit(); con.close()
    return until

def sub_get(user_id: int):
    con = _db(); cur = con.cursor()
    cur.execute("SELECT expires_at FROM subs WHERE user_id=?", (user_id,))
    row = cur.fetchone(); con.close()
    if not row: return None
    try:
        return datetime.fromisoformat(row["expires_at"])
    except Exception:
        return None

def sub_active(user_id: int) -> bool:
    exp = sub_get(user_id)
    return bool(exp and exp > datetime.utcnow())

def sub_days_left(user_id: int) -> int:
    exp = sub_get(user_id)
    if not exp: return 0
    left = (exp - datetime.utcnow()).total_seconds()
    return max(0, int((left + 86399)//86400))

# ============== TRON VERIFY (TronGrid) =========
async def verify_tron_usdt_tx(tx_hash: str) -> tuple[bool, str]:
    if not TRON_API_KEY:
        return False, "TRON_API_KEY не задано."
    if not WALLET_ADDRESS:
        return False, "WALLET_ADDRESS не задано."

    headers = {"TRON-PRO-API-KEY": TRON_API_KEY}

    # 1) Events
    url_events = f"https://api.trongrid.io/v1/transactions/{tx_hash}/events"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url_events, headers=headers, timeout=20) as r:
                if r.status != 200:
                    return False, f"TronGrid events HTTP {r.status}"
                data = await r.json()
    except Exception as e:
        return False, f"Помилка TronGrid events: {e}"

    events = data.get("data") or []
    ok = False; found_amt = 0.0
    for ev in events:
        if ev.get("event_name") != "Transfer": continue
        contract_addr = ev.get("contract_address") or ev.get("contract")
        if contract_addr != USDT_TRON_CONTRACT: continue
        res = ev.get("result") or {}
        to_addr = res.get("to"); raw_val = res.get("value")
        try: amt = float(raw_val)/1_000_000.0
        except: amt = 0.0
        if to_addr == WALLET_ADDRESS and amt >= max(MIN_AMOUNT_USDT, SUB_PRICE):
            ok = True; found_amt = amt; break
    if not ok:
        return False, "Не знайдено USDT-TRC20 переказу на твою адресу з достатньою сумою."

    # 2) TX status
    url_tx = f"https://api.trongrid.io/v1/transactions/{tx_hash}"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url_tx, headers=headers, timeout=20) as r:
                if r.status != 200:
                    return False, f"TronGrid tx HTTP {r.status}"
                data_tx = await r.json()
    except Exception as e:
        return False, f"Помилка TronGrid tx: {e}"

    txs = data_tx.get("data") or []
    if not txs: return False, "TX не знайдено."
    ret = (txs[0].get("ret") or [{}])[0]
    if ret.get("contractRet") != "SUCCESS":
        return False, "TX ще не підтверджений."

    return True, f"Оплату підтверджено: {found_amt:.2f} USDT"

# ================ UI / COMMANDS =================
KB = ReplyKeyboardMarkup(
    [["/signals", "/auto_on 15"], ["/auto_off", "/status"], ["/pay", "/mysub"]],
    resize_keyboard=True
)

def split_long(text: str, chunk_len: int = 3500) -> List[str]:
    if not text: return [""]
    chunks = []
    while len(text) > chunk_len:
        chunks.append(text[:chunk_len])
        text = text[chunk_len:]
    chunks.append(text)
    return chunks

def is_admin(update: Update) -> bool:
    return bool(update.effective_user) and update.effective_user.id == ADMIN_ID

def require_sub(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if ADMIN_ID and uid == ADMIN_ID:
            return await func(update, context)
        if not sub_active(uid):
            return await update.message.reply_text("🔒 Немає активної підписки. Спершу оплатити: /pay")
        return await func(update, context)
    return wrapper

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    STATE.setdefault(chat_id, {"auto_on": False, "every": 15})
    await update.message.reply_text(
        "👋 Готовий!\n\nКоманди:\n"
        "• /signals — сканувати зараз (для підписників)\n"
        "• /auto_on 15 — автопуш кожні 15 хв (5–120)\n"
        "• /auto_off — вимкнути автопуш\n"
        "• /status — стан\n"
        "• /pay — оплатити підписку\n"
        "• /mysub — статус підписки",
        reply_markup=KB
    )

@require_sub
async def signals_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = await build_signals_text()
    for chunk in split_long(txt):
        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

@require_sub
async def auto_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = STATE.setdefault(chat_id, {"auto_on": False, "every": 15})
    minutes = 15
    if context.args:
        try:
            minutes = max(5, min(120, int(context.args[0])))
        except:
            pass
    st["auto_on"] = True
    st["every"] = minutes

    jq = context.application.job_queue
    name = f"auto_{chat_id}"
    if jq:
        for j in jq.get_jobs_by_name(name):
            j.schedule_removal()
        jq.run_repeating(
            auto_push_job, interval=minutes*60, first=5,
            name=name, data={"chat_id": chat_id}
        )
    await update.message.reply_text(f"Автопуш увімкнено: кожні {minutes} хв.", reply_markup=KB)

@require_sub
async def auto_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = STATE.setdefault(chat_id, {"auto_on": False, "every": 15})
    st["auto_on"] = False
    jq = context.application.job_queue
    name = f"auto_{chat_id}"
    if jq:
        for j in jq.get_jobs_by_name(name):
            j.schedule_removal()
    await update.message.reply_text("Автопуш вимкнено.", reply_markup=KB)

@require_sub
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = STATE.setdefault(chat_id, {"auto_on": False, "every": 15})
    await update.message.reply_text(f"Статус: {'ON' if st['auto_on'] else 'OFF'} · кожні {st['every']} хв.")

async def auto_push_job(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    st = STATE.get(chat_id, {})
    if not st or not st.get("auto_on"):
        return
    # субскрипція (адмін може без підписки)
    if int(chat_id) != ADMIN_ID and not sub_active(chat_id):
        try:
            await context.bot.send_message(chat_id, "🔒 Підписка неактивна. Автопуш призупинено. Оплатити: /pay")
        except:
            pass
        jq = context.application.job_queue
        if jq:
            for j in jq.get_jobs_by_name(f"auto_{chat_id}"):
                j.schedule_removal()
        return
    try:
        txt = await build_signals_text()
        for chunk in split_long(txt):
            await context.bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Автопуш помилка: {e}")

# ---------- Subscription commands ----------
async def pay_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not WALLET_ADDRESS:
        await update.message.reply_text("Адреса гаманця не налаштована (WALLET_ADDRESS)."); return
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Я оплатив — надішлю TX hash", callback_data="paid")]])
    txt = (
        f"💳 Підписка *{SUB_DAYS} днів* — *${SUB_PRICE:.2f}*\n\n"
        f"Надішліть *{max(MIN_AMOUNT_USDT, SUB_PRICE):.2f} USDT (TRC20)* на адресу:\n"
        f"`{WALLET_ADDRESS}`\n\n"
        f"Потім використайте `/claim <tx_hash>` або натисніть кнопку нижче."
    )
    await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN, reply_markup=kb)

async def on_cb_paid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    await q.edit_message_text("Надішліть TX hash відповіддю або командою: `/claim <tx_hash>`", parse_mode=ParseMode.MARKDOWN)

async def claim_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Використання: `/claim <tx_hash>`", parse_mode=ParseMode.MARKDOWN); return
    tx = context.args[0].strip()
    ok, msg = await verify_tron_usdt_tx(tx)
    if not ok:
        await update.message.reply_text(f"❌ {msg}"); return
    until = sub_set(update.effective_user.id, SUB_DAYS)
    await update.message.reply_text(
        f"✅ {msg}\nДоступ відкрито до *{until.strftime('%Y-%m-%d %H:%M UTC')}*.",
        parse_mode=ParseMode.MARKDOWN
    )
    if ADMIN_ID:
        u = update.effective_user
        try:
            await context.bot.send_message(ADMIN_ID, f"✅ Оплата: @{u.username or u.id} · TX {tx} · до {until.isoformat()}")
        except: pass

async def mysub_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if sub_active(update.effective_user.id):
        exp = sub_get(update.effective_user.id); left = sub_days_left(update.effective_user.id)
        await update.message.reply_text(f"🔐 Підписка активна: {left} дн. (до {exp.strftime('%Y-%m-%d %H:%M UTC')})")
    else:
        await update.message.reply_text("🔒 Підписки немає або закінчилась. Оплатити: /pay")

# ================ HEARTBEAT =====================
async def heartbeat(_: ContextTypes.DEFAULT_TYPE):
    if not ADMIN_ID:
        return
    try:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
        await _.bot.send_message(chat_id=ADMIN_ID, text=f"✅ Bot is alive | UTC {now}")
    except Exception as e:
        print(f"[heartbeat] send failed: {e}")

# ==================== APP ======================
def build_app() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN")

    subs_init()  # ensure DB table exists

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # commands
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))
    app.add_handler(CommandHandler("auto_on", auto_on_cmd))
    app.add_handler(CommandHandler("auto_off", auto_off_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    # subscription
    app.add_handler(CommandHandler("pay", pay_cmd))
    app.add_handler(CommandHandler("claim", claim_cmd))
    app.add_handler(CommandHandler("mysub", mysub_cmd))
    app.add_handler(CallbackQueryHandler(on_cb_paid, pattern="^paid$"))

    # heartbeat (працює лише якщо є job-queue extras)
    jq = app.job_queue
    if jq:
        jq.run_repeating(heartbeat, interval=timedelta(minutes=HEARTBEAT_MIN), first=10, name="heartbeat")
    else:
        print("[WARN] JobQueue недоступний. Встанови dependency: python-telegram-bot[job-queue]")

    return app

def main():
    app = build_app()
    # один процес = один run_polling. Якщо паралельно стартує інший інстанс — цей завершиться з Conflict і Railway перезапустить.
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
