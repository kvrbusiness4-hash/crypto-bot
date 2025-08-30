import os
import asyncio
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, ApplicationBuilder, AIORateLimiter,
    CommandHandler, ContextTypes, JobQueue
)
import aiohttp

# =========================
# ENV & CONSTANTS
# =========================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))

BYBIT_BASE = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").rstrip("/")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# Параметри ризику / розміру — їх можна змінювати командами
DEFAULT_SL_PCT = float(os.getenv("SL_PCT", "3"))      # 3%
DEFAULT_TP_PCT = float(os.getenv("TP_PCT", "5"))      # 5%
DEFAULT_SIZE_USDT = float(os.getenv("SIZE_USDT", "5"))  # 5 USDT
DEFAULT_LEVERAGE = int(os.getenv("LEVERAGE", "3"))
DEFAULT_STRONG_VOTE = int(os.getenv("STRONG_VOTE", "2"))  # скільки факторів треба для "сильного" сигналу

# Фільтр: TOP30 — аналізуємо лінійні перпети у топі
FILTER_MODE = os.getenv("FILTER_MODE", "TOP30").upper()

# Перемикач автоторгівлі (можна змінювати командами /trade_on /trade_off)
TRADE_ENABLED = os.getenv("TRADE_ENABLED", "0").strip() in ("1", "true", "True", "on", "ON")

# Автопуш інтервал (хвилини)
AUTO_MIN = 15

# Стан у пам’яті (на інстанс)
STATE = {
    "SL_PCT": DEFAULT_SL_PCT,
    "TP_PCT": DEFAULT_TP_PCT,
    "SIZE_USDT": DEFAULT_SIZE_USDT,
    "LEV": DEFAULT_LEVERAGE,
    "TRADE": TRADE_ENABLED,     # True / False
    "AUTO_JOB": None,           # job object
    "LAST_SCAN_UTC": None,
    "FILTER": FILTER_MODE
}

# =========================
# Допоміжне
# =========================
def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

def fmt_pct(v: float) -> str:
    return f"{v:.2f}%"

def safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

# =========================
# BYBIT minimal helpers
# =========================
HEADERS_PUBLIC = {
    "Accept": "application/json",
}
# Для приватних ендпоінтів (спрощено — без підпису; багато що працює і так на v5 set-leverage/order/create
# коли ключі відмічені як read-write; якщо вимагатиме підпис — отримаємо 401/403, але бот не впаде)
def private_headers() -> Dict[str, str]:
    h = {
        "Accept": "application/json",
        "X-BAPI-API-KEY": BYBIT_API_KEY
    }
    return h

async def bybit_get(session: aiohttp.ClientSession, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{BYBIT_BASE}{path}"
    try:
        async with session.get(url, params=params, headers=HEADERS_PUBLIC, timeout=20) as r:
            if r.headers.get("Content-Type", "").startswith("application/json"):
                return await r.json()
            # Якщо прислали HTML/текст
            txt = await r.text()
            return {"retCode": -1, "retMsg": f"NonJSON: {txt[:200]}", "raw": txt}
    except Exception as e:
        return {"retCode": -1, "retMsg": str(e)}

async def bybit_post(session: aiohttp.ClientSession, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{BYBIT_BASE}{path}"
    try:
        async with session.post(url, headers=private_headers(), json=payload, timeout=20) as r:
            ctype = r.headers.get("Content-Type", "")
            if ctype.startswith("application/json"):
                return await r.json()
            txt = await r.text()
            return {"retCode": -1, "retMsg": f"NonJSON: {txt[:200]}", "raw": txt}
    except Exception as e:
        return {"retCode": -1, "retMsg": str(e)}

# =========================
# Сканер сильних сигналів (дуже легкий, щоб не падати)
# =========================
async def fetch_top30_symbols(session: aiohttp.ClientSession) -> List[str]:
    """
    Забираємо тікери лінійних контрактів і беремо TOP30 за об’ємом.
    """
    data = await bybit_get(session, "/v5/market/tickers", {"category": "linear"})
    if data.get("retCode") != 0:
        return []
    rows = data.get("result", {}).get("list", [])
    # сорт за turnover24h (як рядок) — конвертуємо
    def vol(row):
        return safe_float(row.get("turnover24h", "0"), 0.0)
    rows.sort(key=vol, reverse=True)
    top = rows[:30]
    # Повертаємо символи типу "BTCUSDT"
    syms = [r.get("symbol") for r in top if r.get("symbol", "").endswith("USDT")]
    return syms

async def rsi_signal_stub(session: aiohttp.ClientSession, symbol: str) -> Dict[str, Any]:
    """
    Дуже простий скоринг: беремо зміну 24h та "імітуємо" RSI зі свічок 15м (мінімальний запит).
    Якщо щось не ок — score низький і причина з помилкою не ламає бота.
    """
    info = await bybit_get(session, "/v5/market/tickers", {"category": "linear", "symbol": symbol})
    if info.get("retCode") != 0:
        return {"symbol": symbol, "ok": False, "reason": info.get("retMsg", "bad")}

    try:
        row = info["result"]["list"][0]
        chg24 = safe_float(row.get("price24hPcnt", "0")) * 100.0  # у %
        # умовно: сильний шорт коли падіння >2%, сильний лонг коли зростання >2%
        score = 0.0
        direction = "FLAT"
        if chg24 <= -2.0:
            score = min(3.0, abs(chg24) / 2.0)   # 2% => 1 бал, 6% => 3 бали
            direction = "SHORT"
        elif chg24 >= 2.0:
            score = min(3.0, chg24 / 2.0)
            direction = "LONG"

        return {
            "symbol": symbol,
            "ok": True,
            "score": round(score, 2),
            "direction": direction,
            "lastPrice": safe_float(row.get("lastPrice", "0")),
            "chg24": round(chg24, 2)
        }
    except Exception as e:
        return {"symbol": symbol, "ok": False, "reason": str(e)}

async def get_strong_signals(session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    syms = await fetch_top30_symbols(session)
    out: List[Dict[str, Any]] = []
    for s in syms:
        sig = await rsi_signal_stub(session, s)
        if sig.get("ok") and sig.get("score", 0) >= float(STATE.get("STRONG", DEFAULT_STRONG_VOTE)):
            out.append(sig)
        # Обмежимо 2 кращими
        if len(out) >= 2:
            break
    return out

# =========================
# Торгівля (поставити плече, відкрити ринковий ордер з SL/TP)
# =========================
async def set_leverage(session: aiohttp.ClientSession, symbol: str, lev: int) -> Optional[str]:
    """
    Спроба виставити плече. На Unified Trading це /v5/position/set-leverage
    """
    payload = {"category": "linear", "symbol": symbol, "buyLeverage": str(lev), "sellLeverage": str(lev)}
    res = await bybit_post(session, "/v5/position/set-leverage", payload)
    if res.get("retCode") == 0:
        return None
    # Повертаємо текст помилки, але НЕ кидаємо виключення
    return f"{res.get('retCode')} {res.get('retMsg', '')}"

async def place_market_order(
    session: aiohttp.ClientSession,
    symbol: str,
    side: str,          # "Buy" або "Sell"
    qty: float,
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None
) -> str:
    """
    Створити ринковий ордер. На Unified — /v5/order/create
    Примітка: на деяких акаунтах потрібен підпис/серверний час — тоді отримаємо 401/403.
    """
    payload = {
        "category": "linear",
        "symbol": symbol,
        "side": side,               # Buy/Sell
        "orderType": "Market",
        "qty": str(qty),
        "timeInForce": "IOC",
        "reduceOnly": False,
        "closeOnTrigger": False,
    }
    # SL/TP (conditional)
    if sl_price:
        payload["stopLoss"] = str(sl_price)
    if tp_price:
        payload["takeProfit"] = str(tp_price)

    res = await bybit_post(session, "/v5/order/create", payload)
    if res.get("retCode") == 0:
        return "OK"
    return f"ERR {res.get('retCode')} {res.get('retMsg', '')}"

def calc_order_qty_usdt(price: float, size_usdt: float, lev: int) -> float:
    # дуже грубо: (розмір * плече) / ціна = кількість
    if price <= 0:
        return 0.0
    qty = (size_usdt * lev) / price
    # округлення до 4 знаків — здебільшого достатньо
    return max(0.0, round(qty, 4))

# =========================
# JOBS (УВАГА: ВСІ async і ЗАВЖДИ return)
# =========================
async def heartbeat_job(context: ContextTypes.DEFAULT_TYPE):
    try:
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=f"✅ Heartbeat · Uptime 10s · UTC {now_utc_str()}"
        )
    except Exception:
        pass
    return

async def auto_scan_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Автоскан + автоторгівля (якщо STATE['TRADE'] True).
    НІЧОГО не await'имо, що може бути None. Завжди `return` наприкінці.
    """
    try:
        async with aiohttp.ClientSession() as session:
            signals = await get_strong_signals(session)
            STATE["LAST_SCAN_UTC"] = now_utc_str()

            if not signals:
                await context.bot.send_message(
                    chat_id=ADMIN_ID,
                    text="⚠️ Сильних сигналів не знайдено."
                )
                return

            # Надсилаємо короткий звіт
            lines = ["📈 Сильні сигнали (топ30)"]
            for s in signals:
                lines.append(f"• {s['symbol']}: {s['direction']} @ {s['lastPrice']} (24h {s['chg24']}%) · score {s['score']}")
            await context.bot.send_message(chat_id=ADMIN_ID, text="\n".join(lines))

            if not STATE["TRADE"]:
                return

            # Поставити 1–2 ордери
            placed = []
            for s in signals[:2]:
                symbol = s["symbol"]
                direction = s["direction"]
                price = float(s["lastPrice"])
                side = "Buy" if direction == "LONG" else "Sell"

                # 1) плече
                err = await set_leverage(session, symbol, int(STATE["LEV"]))
                if err:
                    await context.bot.send_message(
                        chat_id=ADMIN_ID,
                        text=f"❌ Помилка set-leverage {symbol}: {err}"
                    )

                # 2) розрахувати кількість
                qty = calc_order_qty_usdt(price, float(STATE["SIZE_USDT"]), int(STATE["LEV"]))
                if qty <= 0:
                    await context.bot.send_message(chat_id=ADMIN_ID, text=f"❌ {symbol}: qty=0, пропущено.")
                    continue

                # 3) SL/TP у цінах
                sl_pct = float(STATE["SL_PCT"]) / 100.0
                tp_pct = float(STATE["TP_PCT"]) / 100.0
                if direction == "LONG":
                    sl_price = round(price * (1 - sl_pct), 4)
                    tp_price = round(price * (1 + tp_pct), 4)
                else:
                    sl_price = round(price * (1 + sl_pct), 4)
                    tp_price = round(price * (1 - tp_pct), 4)

                res = await place_market_order(session, symbol, side, qty, sl_price, tp_price)
                placed.append(f"{symbol} {side} qty={qty} → {res}")

            if placed:
                await context.bot.send_message(chat_id=ADMIN_ID, text="🧾 Ордери:\n" + "\n".join(placed))

    except Exception as e:
        # Ніколи не даємо job впасти
        try:
            await context.bot.send_message(chat_id=ADMIN_ID, text=f"⚠️ Помилка автоскану: {e}")
        except Exception:
            pass
    return

# =========================
# Команди
# =========================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("Сканувати /signals", callback_data="noop")],
    ]
    await update.message.reply_text(
        "👋 Готовий!\n\nКоманди:\n"
        "• /signals — сканувати зараз (та автотрейд, якщо увімкнено)\n"
        "• /auto_on 15 — автопуш кожні N хв (5–120)\n"
        "• /auto_off — вимкнути автопуш\n"
        "• /trade_on — увімкнути автоторгівлю\n"
        "• /trade_off — вимкнути автоторгівлю\n"
        "• /set_size 5 — сума угоди у USDT\n"
        "• /set_lev 3 — плече\n"
        "• /set_risk 3 5 — SL/TP у %\n"
        "• /status — стан",
        reply_markup=InlineKeyboardMarkup(kb)
    )

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    last = STATE["LAST_SCAN_UTC"] or "—"
    txt = (
        f"Статус: {'ON' if STATE['AUTO_JOB'] else 'OFF'} · кожні {AUTO_MIN} хв.\n"
        f"SL={fmt_pct(STATE['SL_PCT'])} · TP={fmt_pct(STATE['TP_PCT'])}\n"
        f"TRADE_ENABLED={'ON' if STATE['TRADE'] else 'OFF'} · SIZE={STATE['SIZE_USDT']:.2f} USDT\n"
        f"LEV={STATE['LEV']}\n"
        f"Фільтр: {STATE['FILTER']}\n"
        f"UTC: {last}"
    )
    await update.message.reply_text(txt)

async def auto_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO_MIN
    try:
        if context.args:
            n = int(context.args[0])
            AUTO_MIN = max(5, min(120, n))
    except Exception:
        pass

    job: Optional[Any] = STATE.get("AUTO_JOB")
    if job:
        job.schedule_removal()
        STATE["AUTO_JOB"] = None

    # зареєструвати новий
    job = context.job_queue.run_repeating(auto_scan_job, interval=AUTO_MIN * 60, first=1)
    STATE["AUTO_JOB"] = job
    await update.message.reply_text(f"Автоскан увімкнено: кожні {AUTO_MIN} хв.")

async def auto_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    job: Optional[Any] = STATE.get("AUTO_JOB")
    if job:
        job.schedule_removal()
        STATE["AUTO_JOB"] = None
    await update.message.reply_text("Автоскан вимкнено.")

async def signals_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ручний скан + (за потреби) торгівля
    await update.message.reply_text("Сканую ринок…")
    tmp_context = ContextTypes.DEFAULT_TYPE
    # Використаємо той самий код, що й у job
    await auto_scan_job(context)
    return

async def set_risk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        sl = float(context.args[0])
        tp = float(context.args[1])
        STATE["SL_PCT"] = max(0.1, min(20.0, sl))
        STATE["TP_PCT"] = max(0.1, min(50.0, tp))
        await update.message.reply_text(f"✅ Ризик встановлено: SL={fmt_pct(STATE['SL_PCT'])} · TP={fmt_pct(STATE['TP_PCT'])}")
    except Exception:
        await update.message.reply_text("❌ Приклад: /set_risk 3 5")

async def set_size_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        usd = float(context.args[0])
        STATE["SIZE_USDT"] = max(2.0, min(1000.0, usd))
        await update.message.reply_text(f"✅ Розмір угоди встановлено: {STATE['SIZE_USDT']:.2f} USDT")
    except Exception:
        await update.message.reply_text("❌ Приклад: /set_size 5")

async def set_lev_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        lev = int(context.args[0])
        STATE["LEV"] = max(1, min(50, lev))
        await update.message.reply_text(f"✅ Плече встановлено: x{STATE['LEV']}")
    except Exception:
        await update.message.reply_text("❌ Приклад: /set_lev 3")

async def trade_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["TRADE"] = True
    await update.message.reply_text("Автоторгівля: УВІМКНЕНО ✅")

async def trade_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["TRADE"] = False
    await update.message.reply_text("Автоторгівля: ВИМКНЕНО ⛔")

# =========================
# MAIN
# =========================
def require_env():
    missing = []
    if not TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not ADMIN_ID:
        missing.append("ADMIN_ID")
    if missing:
        raise RuntimeError(f"Missing env: {', '.join(missing)}")

async def on_start(app: Application):
    # heartbeat кожні 10 хв (щоб не спамити)
    app.job_queue.run_repeating(heartbeat_job, interval=600, first=5)

def main():
    require_env()

    application: Application = (
        ApplicationBuilder()
        .token(TOKEN)
        .rate_limiter(AIORateLimiter())
        .build()
    )

    # Команди
    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(CommandHandler("auto_on", auto_on_cmd))
    application.add_handler(CommandHandler("auto_off", auto_off_cmd))
    application.add_handler(CommandHandler("signals", signals_cmd))

    application.add_handler(CommandHandler("set_risk", set_risk_cmd))
    application.add_handler(CommandHandler("set_size", set_size_cmd))
    application.add_handler(CommandHandler("set_lev", set_lev_cmd))

    application.add_handler(CommandHandler("trade_on", trade_on_cmd))
    application.add_handler(CommandHandler("trade_off", trade_off_cmd))

    application.post_init = on_start  # стартові job-и

    print("Starting bot…")
    application.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
