# bot_paid.py
# -*- coding: utf-8 -*-

import os, json, math, asyncio, aiohttp, time, re
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta

from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ========= ENV =========
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # обов'язково
TRON_ADDRESS       = os.getenv("TRON_ADDRESS", "TEzU1DxQiYi5YLVdp6fxq1LmWmjtYYJnHb")  # твій гаманець (TRC20)
TRONSCAN_API_KEY   = os.getenv("TRONSCAN_API_KEY", "27444a19-f86b-48ce-841b-f1949f7df718")  # твій ключ
SUB_PRICE_USDT     = float(os.getenv("SUB_PRICE_USDT", "25"))
SUB_DAYS           = int(os.getenv("SUB_DAYS", "30"))

# Додаткові змінні для Railway (не обов'язково міняти)
TRONSCAN_BASES = [
    "https://apilist.tronscanapi.com",         # публічний
    "https://apilist.tronscan.org",            # альтернативний
]

USDT_CONTRACT = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"  # офіційний USDT-TRC20

# ========= СТЕЙТ / ПІДПИСКИ =========
SUBS_FILE = "subscribers.json"
STATE: Dict[int, Dict[str, int | bool]] = {}   # chat_id -> {"auto_on": bool, "every": int}

def load_subs() -> Dict[str, dict]:
    try:
        with open(SUBS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_subs(data: Dict[str, dict]) -> None:
    tmp = SUBS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, SUBS_FILE)

def now_ts() -> int:
    return int(time.time())

def is_active(chat_id: int) -> bool:
    subs = load_subs()
    rec = subs.get(str(chat_id))
    if not rec: return False
    return int(rec.get("paid_until", 0)) > now_ts()

def paid_until_str(ts: int) -> str:
    if ts <= 0: return "—"
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")

# ========= ТЕХНІЧНІ ІНДИКАТОРИ / СИГНАЛИ =========
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
        ch = series[i] - series[i-1]
        gains.append(max(0.0, ch))
        losses.append(max(0.0, -ch))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [0.0] * (period)
    if avg_loss == 0: rsis.append(100.0)
    else: rsis.append(100.0 - (100.0 / (1.0 + (avg_gain/(avg_loss+1e-9)))))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        if avg_loss == 0: rsis.append(100.0)
        else:
            rs = avg_gain/(avg_loss+1e-9)
            rsis.append(100.0 - (100.0/(1.0+rs)))
    return rsis

def macd(series: List[float], fast:int=12, slow:int=26, signal:int=9) -> Tuple[List[float], List[float]]:
    if len(series) < slow + signal: return [], []
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
    if len(ema50) and len(ema200):
        if ema50[-1] > ema200[-1]: trend = 1
        elif ema50[-1] < ema200[-1]: trend = -1

    rsi15 = rsi(series, 7)
    rsi30 = rsi(series, 14)
    rsi60 = rsi(series, 28)
    macd_line, macd_sig = macd(series)

    def rsi_vote(last: float) -> int:
        if last is None: return 0
        if last <= 30: return +1
        if last >= 70: return -1
        return 0

    votes = 0
    if rsi15: votes += rsi_vote(rsi15[-1]); explain.append(f"RSI15={rsi15[-1]:.1f}")
    if rsi30: votes += rsi_vote(rsi30[-1]); explain.append(f"RSI30={rsi30[-1]:.1f}")
    if rsi60: votes += rsi_vote(rsi60[-1]); explain.append(f"RSI60={rsi60[-1]:.1f}")

    if macd_line and macd_sig:
        if macd_line[-1] > macd_sig[-1]: votes += 1; explain.append("MACD↑")
        elif macd_line[-1] < macd_sig[-1]: votes -= 1; explain.append("MACD↓")
        else: explain.append("MACD·")

    if trend > 0: votes += 1; explain.append("Trend=UP")
    elif trend < 0: votes -= 1; explain.append("Trend=DOWN")
    else: explain.append("Trend=FLAT")

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

# ========= РИНОК =========
MARKET_URL = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=market_cap_desc&per_page=120&page=1"
    "&sparkline=true&price_change_percentage=24h"
)
STABLES = {"USDT","USDC","DAI","TUSD","FDUSD","USDD","PYUSD","EURS","EURT","BUSD"}

async def fetch_market(session: aiohttp.ClientSession) -> List[dict]:
    async with session.get(MARKET_URL, timeout=25) as r:
        r.raise_for_status()
        return await r.json()

def is_good_symbol(item: dict) -> bool:
    sym = (item.get("symbol") or "").upper()
    name = (item.get("name") or "").upper()
    if sym in STABLES or any(s in name for s in STABLES): return False
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
        if direction in ("LONG","SHORT"):
            score = 2
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

# ========= TronScan перевірка транзакції =========
async def fetch_tx_info(session: aiohttp.ClientSession, txid: str) -> Optional[dict]:
    headers = {"Accept": "application/json"}
    # деякі інстанси вимагають API-KEY у заголовку
    if TRONSCAN_API_KEY:
        headers["TRON-PRO-API-KEY"] = TRONSCAN_API_KEY

    for base in TRONSCAN_BASES:
        url = f"{base}/api/transaction-info?hash={txid}"
        try:
            async with session.get(url, headers=headers, timeout=25) as r:
                if r.status == 200:
                    return await r.json()
        except Exception:
            continue
    return None

def parse_usdt_incoming(tx: dict) -> float:
    """
    Повертає суму USDT (у десятковому вигляді), що ЗАЙШЛА на TRON_ADDRESS у цій транзакції.
    Якщо немає — 0.0
    """
    if not tx: return 0.0

    # TronScan часто повертає масиви trc20TransferInfo або tokenTransferInfo
    for key in ("trc20TransferInfo", "tokenTransferInfo"):
        arr = tx.get(key) or []
        for t in arr:
            to_addr = (t.get("to_address") or t.get("toAddress") or "").strip()
            contract = (t.get("contract_address") or t.get("contractAddress") or "").strip()
            if to_addr == TRON_ADDRESS and contract == USDT_CONTRACT:
                # value/decimals
                val = t.get("amount_str") or t.get("amount") or t.get("quant") or t.get("value")
                dec = t.get("decimals") or 6
                try:
                    # інколи amount у вже-десятковому вигляді
                    if isinstance(val, str) and val.isdigit():
                        amount = float(val) / (10 ** int(dec))
                    else:
                        amount = float(val)
                    return amount
                except Exception:
                    continue
    return 0.0

# ========= КНОПКИ / ДОСТУП =========
KB = ReplyKeyboardMarkup(
    [["/signals", "/auto_on 15"], ["/auto_off", "/status"], ["/subscribe", "/check <txid>"]],
    resize_keyboard=True
)

async def guard_active(update: Update) -> bool:
    chat_id = update.effective_chat.id
    if not is_active(chat_id):
        await update.message.reply_text(
            "🔒 У тебе **немає активної підписки**.\n"
            f"Ціна: *{SUB_PRICE_USDT} USDT (TRC20)* за *{SUB_DAYS} днів*.\n\n"
            "Натисни /subscribe щоб отримати адресу для оплати."
        )
        return False
    return True

# ========= КОМАНДИ =========
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    STATE.setdefault(chat_id, {"auto_on": False, "every": 15})
    subs = load_subs().get(str(chat_id), {})
    paid_until = int(subs.get("paid_until", 0))

    await update.message.reply_text(
        "👋 Привіт! Я Crypto Signals Bot (скальп/короткі сигнали).\n\n"
        "Основні команди:\n"
        "• /signals — показати сильні сигнали зараз\n"
        "• /auto_on 15 — автопуш кожні N хв (5–120)\n"
        "• /auto_off — вимкнути автопуш\n"
        "• /status — стан підписки/бота\n\n"
        "Оплата підписки:\n"
        f"• /subscribe — оплата {SUB_PRICE_USDT} USDT (TRC20) на {SUB_DAYS} днів\n"
        "• /check <txid> — підтвердити оплату, вказавши TXID\n\n"
        f"Твій статус: {'АКТИВНА' if is_active(chat_id) else 'НЕ активна'} до: {paid_until_str(paid_until)}",
        reply_markup=KB
    )

async def subscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    amount = SUB_PRICE_USDT
    text = (
        "💳 *Оплата підписки*\n\n"
        f"Сума: *{amount} USDT* (мережа *TRON / TRC20*)\n"
        f"Адреса для переказу:\n`{TRON_ADDRESS}`\n\n"
        "Після відправки — надішли TXID командою:\n"
        "`/check <txid>` (без кутових дужок)\n\n"
        "_Без memo/tag. Комісія мережі сплачується відправником._"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def check_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Будь ласка, надішли TXID у форматі: `/check <txid>`", parse_mode=ParseMode.MARKDOWN)
        return

    txid = context.args[0].strip()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", txid):
        await update.message.reply_text("Схоже, це не схожe на валідний TXID (очікується 64 hex-символи). Перевір і спробуй ще.")
        return

    await update.message.reply_text("🔎 Перевіряю транзакцію в TronScan… Будь ласка, зачекай.")

    async with aiohttp.ClientSession() as s:
        tx = await fetch_tx_info(s, txid)

    if not tx:
        await update.message.reply_text("Не вдалось отримати дані по транзакції. Спробуй ще раз за хвилину або перевір TXID.")
        return

    amount = parse_usdt_incoming(tx)
    if amount >= SUB_PRICE_USDT - 0.01:
        # зараховуємо підписку
        subs = load_subs()
        rec = subs.get(str(chat_id), {"paid_until": 0})
        base = max(int(rec.get("paid_until", 0)), now_ts())
        new_until = base + SUB_DAYS * 24 * 3600
        rec.update({
            "paid_until": new_until,
            "last_tx": txid,
            "last_amount": amount,
            "last_update": now_ts(),
        })
        subs[str(chat_id)] = rec
        save_subs(subs)

        await update.message.reply_text(
            f"✅ Оплату знайдено: {amount:.2f} USDT на `{TRON_ADDRESS}`.\n"
            f"Підписку продовжено до: *{paid_until_str(new_until)}*.\n"
            "Тепер доступні /signals та автопуш.",
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await update.message.reply_text(
            "❌ Не знайшов в цій транзакції достатнє поповнення USDT на наш адресу.\n"
            f"Знайдено: {amount:.6f} USDT. Потрібно: {SUB_PRICE_USDT:.2f} USDT.\n"
            "Перевір, будь ласка, що відправляв саме на TRC20 і на правильну адресу."
        )

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subs = load_subs().get(str(chat_id), {})
    paid_until = int(subs.get("paid_until", 0))
    st = STATE.setdefault(chat_id, {"auto_on": False, "every": 15})
    await update.message.reply_text(
        f"Підписка: {'АКТИВНА' if is_active(chat_id) else 'НЕ активна'} до {paid_until_str(paid_until)}\n"
        f"Автопуш: {'ON' if st['auto_on'] else 'OFF'} · кожні {st['every']} хв."
    )

async def signals_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard_active(update): return
    txt = await build_signals_text()
    for chunk in split_long(txt):
        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

async def auto_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard_active(update): return
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

# ========= АВТОПУШ =========
async def auto_push_job(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    # якщо підписка закінчилась — відключаємо автопуш
    if not is_active(chat_id):
        name = f"auto_{chat_id}"
        for j in context.application.job_queue.get_jobs_by_name(name):
            j.schedule_removal()
        STATE.get(chat_id, {}).update({"auto_on": False})
        try:
            await context.bot.send_message(chat_id=chat_id, text="🔒 Підписка закінчилась, автопуш зупинено. Онови /subscribe.")
        except Exception:
            pass
        return

    try:
        txt = await build_signals_text()
        for chunk in split_long(txt):
            await context.bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Автопуш помилка: {e}")

# ========= УТИЛІТИ =========
def split_long(text: str, chunk_len: int = 3500) -> List[str]:
    if not text: return [""]
    chunks = []
    while len(text) > chunk_len:
        chunks.append(text[:chunk_len])
        text = text[chunk_len:]
    chunks.append(text)
    return chunks

# ========= MAIN =========
def main():
    if not TELEGRAM_BOT_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN env var"); return

    # гарантуємо наявність файлу підписок
    if not os.path.exists(SUBS_FILE):
        with open(SUBS_FILE, "w", encoding="utf-8") as f:
            f.write("{}")

    print("Bot running | Paid + Signals | CoinGecko + TronScan")
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("subscribe", subscribe_cmd))
    app.add_handler(CommandHandler("check", check_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))
    app.add_handler(CommandHandler("auto_on", auto_on_cmd))
    app.add_handler(CommandHandler("auto_off", auto_off_cmd))

    app.run_polling()

if __name__ == "__main__":
    main()
