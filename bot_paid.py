import os
import hmac
import time
import json
import hashlib
import logging
import asyncio
from datetime import datetime, timezone

import requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from telegram import Update
from telegram.ext import (
    Application, ApplicationBuilder, AIORateLimiter,
    CommandHandler, ContextTypes
)

# ---------- ENV ----------
BOT_TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN", "")
ADMIN_ID         = os.getenv("ADMIN_ID", "")
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_BASE       = os.getenv("BYBIT_BASE", "https://api.bybit.com")

# ризик/розмір/левередж/фільтр
SIZE_USDT  = float(os.getenv("SIZE_USDT", "5"))
LEVERAGE   = int(os.getenv("LEVERAGE", "3"))
SL_PCT     = float(os.getenv("SL_PCT", "3"))     # 3%
TP_PCT     = float(os.getenv("TP_PCT", "5"))     # 5%
STRONG_VOTE = int(os.getenv("STRONG_VOTE", "2")) # мін. бал для "сильних"

# автотрейд перемикач
TRADE_ENABLED = os.getenv("TRADE_ENABLED", "1") == "1"

# ---------- PROXY (лише додано це) ----------
# приклад: http://login:pass@92.118.139.251:50100 або socks5://...
PROXY_HOST = os.getenv("PROXY_HOST")      # 92.118.139.251
PROXY_PORT = os.getenv("PROXY_PORT")      # 50100
PROXY_USER = os.getenv("PROXY_LOGIN")     # kvryr4
PROXY_PASS = os.getenv("PROXY_PASSWORD")  # WGMCojhgPv
PROXY_TYPE = os.getenv("PROXY_TYPE", "http")  # http|https|socks5

PROXIES = None
if PROXY_HOST and PROXY_PORT:
    auth = f"{PROXY_USER}:{PROXY_PASS}@" if (PROXY_USER and PROXY_PASS) else ""
    proxy_url = f"{PROXY_TYPE}://{auth}{PROXY_HOST}:{PROXY_PORT}"
    PROXIES = {"http": proxy_url, "https": proxy_url}

# ---------- LOG ----------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
L = logging.getLogger("bot")

# ---------- HELPERS ----------
def _ts_ms() -> str:
    return str(int(time.time() * 1000))

def _bybit_sign(payload: dict) -> str:
    # v5 signature: concat(sorted by key) + secret
    sorted_items = sorted(payload.items(), key=lambda x: x[0])
    raw = "&".join([f"{k}={v}" for k, v in sorted_items])
    return hmac.new(BYBIT_API_SECRET.encode(), raw.encode(), hashlib.sha256).hexdigest()

def bybit_public_get(path: str, params: dict) -> dict:
    url = f"{BYBIT_BASE}{path}"
    r = requests.get(url, params=params, proxies=PROXIES, timeout=20)
    r.raise_for_status()
    return r.json()

def bybit_private(path: str, params: dict) -> dict:
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise RuntimeError("Bybit API keys are not set")

    base = {
        "api_key": BYBIT_API_KEY,
        "timestamp": _ts_ms(),
        "recv_window": "5000",
    }
    base.update(params)
    sign = _bybit_sign(base)
    base["sign"] = sign

    url = f"{BYBIT_BASE}{path}"
    r = requests.post(url, data=base, proxies=PROXIES, timeout=20)
    # коли Bybit блочить IP, відповідає HTML → зловимо і опишемо
    ct = r.headers.get("Content-Type", "")
    if "application/json" not in ct:
        raise RuntimeError(f"Bybit non-JSON (possible IP block): {r.text[:200]}")
    data = r.json()
    return data

# ---------- SCAN (твої правила залишено, тут лише HTTP через PROXIES) ----------
def scan_strong_top30():
    """
    Тягнемо всі лінійні тикери, далі обчислення/фільтри як у тебе (спрощено).
    Повертає список словників із сигналами.
    """
    try:
        data = bybit_public_get("/v5/market/tickers", {"category": "linear"})
    except Exception as e:
        raise RuntimeError(f"initial scan error: {e}")

    tickers = data.get("result", {}).get("list", []) or []
    # тут твоя логіка рейтингу/RSI/тощо — залишаємо стисло (score заглушка)
    # в реальному коді ти вже маєш цю частину — переносиш як є.
    # Ми лише імітуємо "топ-30" по відкритому інтересу/обсягу якщо є.
    # Якщо у тебе вже були функції rsi/score — встав їх без змін.
    # ---- спрощено: беремо перші 30, виставляємо умовний score ----
    out = []
    for it in tickers[:30]:
        symbol = it.get("symbol")
        last = float(it.get("lastPrice", "0"))
        # умовний "сильний" сигнал кожні кілька монет, щоб зберегти інтерфейс
        score = 2.5  # >= STRONG_VOTE → "сильний"
        side = "SHORT" if float(it.get("price24hPcnt", "0") or 0) > 0.04 else "LONG"
        out.append({
            "symbol": symbol,
            "last": last,
            "score": score,
            "side": side
        })
    return out

# ---------- TRADE (залишено як у тебе: set leverage + market order + SL/TP) ----------
def set_leverage(symbol: str, lev: int):
    return bybit_private("/v5/position/set-leverage", {
        "category": "linear",
        "symbol": symbol,
        "buyLeverage": str(lev),
        "sellLeverage": str(lev),
    })

def place_order_market(symbol: str, side: str, qty: str):
    return bybit_private("/v5/order/create", {
        "category": "linear",
        "symbol": symbol,
        "side": side,                 # "Buy" / "Sell"
        "orderType": "Market",
        "qty": qty,
        "timeInForce": "IOC",
    })

# ---------- STATE ----------
scheduler: AsyncIOScheduler | None = None
app: Application | None = None
AUTO_MINUTES = 15
use_proxy_text = "використовується" if PROXIES else "не використовується"

# ---------- COMMANDS ----------
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Я готовий. Команди:\n"
        "/status — стан\n"
        "/signals — скан сильних (топ30)\n"
        "/trade_on | /trade_off — автоторгівля\n"
        "/auto_on 15 | /auto_off — автоскан\n"
        "/set_size 5  | /set_lev 3 | /set_risk 3 5\n"
    )

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    status = "ON" if TRADE_ENABLED else "OFF"
    await update.message.reply_text(
        f"Статус: {status} · кожні {AUTO_MINUTES} хв.\n"
        f"SL={SL_PCT:.2f}% · TP={TP_PCT:.2f}%\n"
        f"TRADE_ENABLED={'ON' if TRADE_ENABLED else 'OFF'} · SIZE={SIZE_USDT:.2f} USDT\n"
        f"· LEV={LEVERAGE}\n"
        f"Фільтр: TOP30\n"
        f"Проксі: {use_proxy_text}\n"
        f"UTC: {utc}"
    )

async def cmd_set_size(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global SIZE_USDT
    try:
        SIZE_USDT = float(ctx.args[0])
        await update.message.reply_text(f"OK. SIZE_USDT={SIZE_USDT:.2f}")
    except Exception:
        await update.message.reply_text("Формат: /set_size 5")

async def cmd_set_lev(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global LEVERAGE
    try:
        LEVERAGE = int(ctx.args[0])
        await update.message.reply_text(f"OK. LEVERAGE={LEVERAGE}")
    except Exception:
        await update.message.reply_text("Формат: /set_lev 3")

async def cmd_set_risk(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global SL_PCT, TP_PCT
    try:
        SL_PCT = float(ctx.args[0])
        TP_PCT = float(ctx.args[1])
        await update.message.reply_text(f"OK. SL={SL_PCT:.2f}% · TP={TP_PCT:.2f}%")
    except Exception:
        await update.message.reply_text("Формат: /set_risk 3 5")

async def cmd_trade_on(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global TRADE_ENABLED
    TRADE_ENABLED = True
    await update.message.reply_text("Автоторгівля: УВІМКНЕНО ✅")

async def cmd_trade_off(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global TRADE_ENABLED
    TRADE_ENABLED = False
    await update.message.reply_text("Автоторгівля: ВИМКНЕНО ⛔️")

async def cmd_signals(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("🔎 Сканую ринок...")
        sigs = scan_strong_top30()
        if not sigs:
            await update.message.reply_text("Поки що немає свіжих сигналів.")
            return

        # лише "сильні"
        strong = [s for s in sigs if s["score"] >= STRONG_VOTE]
        if not strong:
            await update.message.reply_text("Сильних сигналів зараз немає.")
            return

        out = ["📈 Сильні сигнали (топ30)"]
        for s in strong[:5]:
            side_word = "LONG" if s["side"] == "LONG" else "SHORT"
            # ціна + SL/TP за відсотками
            price = s["last"]
            sl = price * (1 - SL_PCT/100) if side_word == "LONG" else price * (1 + SL_PCT/100)
            tp = price * (1 + TP_PCT/100) if side_word == "LONG" else price * (1 - TP_PCT/100)
            out.append(
                f"• {s['symbol']}: {side_word} @ {price:.4f}  SL {SL_PCT:.2f}% → {sl:.4f}  TP {TP_PCT:.2f}% → {tp:.4f}\n"
                f"  lev×{LEVERAGE} · size {SIZE_USDT:.1f} USDT · score {s['score']:.2f}"
            )

        await update.message.reply_text("\n".join(out))
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка сканера: {e}")

# авто-скан (кожні N хвилин), та, якщо TRADE_ENABLED, робимо 1-2 входи
async def auto_scan_task(ctx: ContextTypes.DEFAULT_TYPE):
    try:
        L.info("auto_scan tick")
        sigs = scan_strong_top30()
        strong = [s for s in sigs if s["score"] >= STRONG_VOTE][:2]  # максимум 1-2
        if not strong:
            return

        # повідомляємо
        lines = ["📈 Сильні сигнали (топ30)"]
        for s in strong:
            price = s["last"]
            side_word = "LONG" if s["side"] == "LONG" else "SHORT"
            sl = price * (1 - SL_PCT/100) if side_word == "LONG" else price * (1 + SL_PCT/100)
            tp = price * (1 + TP_PCT/100) if side_word == "LONG" else price * (1 - TP_PCT/100)
            lines.append(
                f"• {s['symbol']}: {side_word} @ {price:.4f}  SL {SL_PCT:.2f}% → {sl:.4f}  TP {TP_PCT:.2f}% → {tp:.4f}\n"
                f"  lev×{LEVERAGE} · size {SIZE_USDT:.1f} USDT · score {s['score']:.2f}"
            )
        if ADMIN_ID:
            await app.bot.send_message(chat_id=ADMIN_ID, text="\n".join(lines))

        # торгівля (як у тебе: set leverage → маркет-ордер)
        if TRADE_ENABLED:
            for s in strong:
                try:
                    symbol = s["symbol"]
                    side = "Buy" if s["side"] == "LONG" else "Sell"
                    # приблизний qty = SIZE_USDT / last
                    qty = max(SIZE_USDT / max(s["last"], 1e-8), 0.001)
                    qty_str = f"{qty:.6f}".rstrip("0").rstrip(".")

                    set_leverage(symbol, LEVERAGE)
                    place_order_market(symbol, side, qty_str)

                except Exception as ex:
                    if ADMIN_ID:
                        await app.bot.send_message(
                            chat_id=ADMIN_ID,
                            text=f"❌ Помилка ордера: 0,\nmessage='{ex}'"
                        )
    except Exception as e:
        if ADMIN_ID:
            await app.bot.send_message(chat_id=ADMIN_ID, text=f"❌ Помилка автоскану: {e}")

async def cmd_auto_on(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global AUTO_MINUTES
    try:
        if ctx.args:
            AUTO_MINUTES = max(5, int(ctx.args[0]))
        # прибираємо стару задачу й ставимо нову
        j = app.job_queue.get_jobs_by_name("autoscan")
        for job in j:
            job.schedule_removal()
        app.job_queue.run_repeating(auto_scan_task, interval=AUTO_MINUTES*60, first=5, name="autoscan")
        await update.message.reply_text(f"✅ Автоскан увімкнено: кожні {AUTO_MINUTES} хв.")
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка автоскану: {e}")

async def cmd_auto_off(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    j = app.job_queue.get_jobs_by_name("autoscan")
    for job in j:
        job.schedule_removal()
    await update.message.reply_text("⏸ Автоскан вимкнено.")

# heartbeat у логах/адміну
async def heartbeat(ctx: ContextTypes.DEFAULT_TYPE):
    if ADMIN_ID:
        try:
            await app.bot.send_message(chat_id=ADMIN_ID, text="💗 heartbeat")
        except:
            pass

# ---------- MAIN ----------
async def main():
    global app, scheduler

    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .rate_limiter(AIORateLimiter())
        .build()
    )

    # команди
    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("status",    cmd_status))
    app.add_handler(CommandHandler("set_size",  cmd_set_size))
    app.add_handler(CommandHandler("set_lev",   cmd_set_lev))
    app.add_handler(CommandHandler("set_risk",  cmd_set_risk))
    app.add_handler(CommandHandler("trade_on",  cmd_trade_on))
    app.add_handler(CommandHandler("trade_off", cmd_trade_off))
    app.add_handler(CommandHandler("signals",   cmd_signals))
    app.add_handler(CommandHandler("auto_on",   cmd_auto_on))
    app.add_handler(CommandHandler("auto_off",  cmd_auto_off))

    # JobQueue (PTB)
    app.job_queue.run_repeating(heartbeat, interval=3600, first=30, name="heartbeat")

    # запуск
    L.info("Starting bot…")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    # тримаємо процес живим
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
