# bot_paid.py
# -*- coding: utf-8 -*-

import os
import logging
from typing import Any, Dict, Optional

import requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    AIORateLimiter,   # ← ось тут тепер правильно
)
from telegram.ext._rate_limiter import AIORateLimiter  # PTB 20.x

# ------------ Логи ------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
L = logging.getLogger("bot")

# ------------ Конфіг / ENV ------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_ID = os.getenv("ADMIN_ID", "").strip()

BYBIT_BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").rstrip("/")
BYBIT_PROXY = os.getenv("BYBIT_PROXY", "").strip()  # напр.: http://user:pass@ip:port

DEFAULT_SCAN_MIN = int(os.getenv("DEFAULT_SCAN_MIN", os.getenv("HEARTBEAT_MIN", "15")))
SIZE_USDT = float(os.getenv("SIZE_USDT", "5"))
LEVERAGE = int(os.getenv("LEVERAGE", "3"))
SL_PCT = float(os.getenv("SL_PCT", "3"))
TP_PCT = float(os.getenv("TP_PCT", "5"))
TRADE_ENABLED = os.getenv("TRADE_ENABLED", "ON").upper() == "ON"

UTC_FMT = "%Y-%m-%d %H:%M:%SZ"

# ------------ Глобальні ------------
app: Optional[Application] = None
scheduler: Optional[AsyncIOScheduler] = None
auto_scan_job = None  # APScheduler job-об'єкт


# ------------ Утиліти ------------
def utc_now_str() -> str:
    import datetime as dt
    return dt.datetime.utcnow().strftime(UTC_FMT)


def get_interval_min(job) -> int:
    """
    Безпечно дістає інтервал автоскану в хвилинах для будь-якої версії APScheduler/PTB.
    Повертає DEFAULT_SCAN_MIN, якщо job немає або структура інша.
    """
    try:
        if not job:
            return int(DEFAULT_SCAN_MIN)

        # APScheduler: job.trigger.interval -> timedelta
        trig = getattr(job, "trigger", None)
        if trig is not None:
            iv = getattr(trig, "interval", None)
            if iv is not None:
                ts = getattr(iv, "total_seconds", None)
                if callable(ts):
                    sec = int(ts())
                else:
                    sec = int(iv)
                return max(1, sec // 60) if sec >= 60 else max(1, sec)

        # На деяких збірках PTB JobQueue може бути .interval (td або секунди)
        iv = getattr(job, "interval", None)
        if iv is not None:
            ts = getattr(iv, "total_seconds", None)
            if callable(ts):
                sec = int(ts())
            else:
                sec = int(iv)
            return max(1, sec // 60) if sec >= 60 else max(1, sec)
    except Exception:
        pass
    return int(DEFAULT_SCAN_MIN)


def _requests_proxies() -> Optional[Dict[str, str]]:
    """
    Готує dict для requests з урахуванням BYBIT_PROXY.
    Підтримка http/https/socks5 (через requests[socks] якщо встановлено).
    """
    if not BYBIT_PROXY:
        return None
    return {
        "http": BYBIT_PROXY,
        "https": BYBIT_PROXY,
    }


async def api_get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Асинхронний wrapper над requests.get (через to_thread), з проксі та обробкою 403/HTML.
    """
    url = f"{BYBIT_BASE_URL}{path}"
    proxies = _requests_proxies()
    headers = {"Accept": "application/json"}

    def _do() -> Dict[str, Any]:
        r = requests.get(url, params=params, headers=headers, timeout=20, proxies=proxies)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        # якщо раптом прийшов HTML (IP block/403 сторінка)
        if "application/json" not in ct.lower():
            raise RuntimeError(f"Bybit non-JSON (possible IP block): {r.text[:200]}")
        return r.json()

    return await asyncio.to_thread(_do)


async def initial_scan() -> Dict[str, Any]:
    """
    Отримує список тікерів (linear).
    """
    return await api_get_json("/v5/market/tickers", {"category": "linear"})


# ------------ Команди ------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Привет! Я готов. Команды:\n"
        "/status — статус\n"
        "/signals — скан сильных (top30)\n"
        "/trade_on | /trade_off — автоторговля\n"
        f"/auto_on {DEFAULT_SCAN_MIN} | /auto_off — автоскан\n"
        f"/set_size {int(SIZE_USDT)} — размер сделки в USDT\n"
        f"/set_lev {LEVERAGE} — плечо\n"
        f"/set_risk {int(SL_PCT)} {int(TP_PCT)} — SL/TP в %"
    )
    await update.message.reply_text(text)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    interval_min = get_interval_min(auto_scan_job)
    proxy_state = "используется" if BYBIT_PROXY else "не используется"

    text = (
        f"Статус: {'ON' if auto_scan_job else 'OFF'} · каждые {interval_min} мин.\n"
        f"SL={SL_PCT:.2f}% · TP={TP_PCT:.2f}%\n"
        f"TRADE_ENABLED={'ON' if TRADE_ENABLED else 'OFF'} · SIZE={SIZE_USDT:.2f} USDT\n"
        f"· LEV={LEVERAGE}\n"
        f"Фильтр: TOP30\n"
        f"Прокси: {proxy_state}\n"
        f"UTC: {utc_now_str()}"
    )
    await update.message.reply_text(text)


async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔎 Сканирую рынок…")
    try:
        data = await initial_scan()
        rows = data.get("result", {}).get("list", []) or []
        await msg.edit_text(f"🔍 Скан OK · получено {len(rows)} тикеров ·\nUTC {utc_now_str()}")
    except Exception as e:
        # кидаємо файл з “тілом” відповіді при 403/HTML
        try:
            j = {"error": str(e)}
            s = json.dumps(j, ensure_ascii=False, indent=2)
        except Exception:
            s = str(e)
        await msg.edit_text(
            f"❌ Ошибка сканера: {s[:300]}..."
        )


async def cmd_trade_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TRADE_ENABLED
    TRADE_ENABLED = True
    await update.message.reply_text("Автоторговля: ВКЛЮЧЕНА ✅")


async def cmd_trade_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TRADE_ENABLED
    TRADE_ENABLED = False
    await update.message.reply_text("Автоторговля: ВЫКЛЮЧЕНА ⛔️")


async def cmd_set_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SIZE_USDT
    try:
        v = float(context.args[0])
        if v <= 0:
            raise ValueError
        SIZE_USDT = v
        await update.message.reply_text(f"OK. SIZE_USDT={SIZE_USDT:.2f}")
    except Exception:
        await update.message.reply_text("Формат: /set_size 5")


async def cmd_set_lev(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LEVERAGE
    try:
        v = int(context.args[0])
        if v < 1:
            raise ValueError
        LEVERAGE = v
        await update.message.reply_text(f"OK. LEVERAGE={LEVERAGE}")
    except Exception:
        await update.message.reply_text("Формат: /set_lev 3")


async def cmd_set_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SL_PCT, TP_PCT
    try:
        sl = float(context.args[0])
        tp = float(context.args[1])
        if sl <= 0 or tp <= 0:
            raise ValueError
        SL_PCT, TP_PCT = sl, tp
        await update.message.reply_text(f"OK. SL={SL_PCT:.2f}%  TP={TP_PCT:.2f}%")
    except Exception:
        await update.message.reply_text("Формат: /set_risk 3 5")


# ------------ Автоскан/Хартбит ------------
async def heartbeat(ctx: ContextTypes.DEFAULT_TYPE):
    if ADMIN_ID:
        try:
            await app.bot.send_message(ADMIN_ID, "💗 heartbeat")
        except Exception:
            pass


async def auto_scan_tick():
    """
    Те саме, що /signals, але без чату—пише адмінам ( необов'язково ), головне—логіка скану.
    """
    try:
        data = await initial_scan()
        rows = data.get("result", {}).get("list", []) or []
        L.info("Auto-scan OK: %s tickers", len(rows))
    except Exception as e:
        L.error("Auto-scan error: %s", e)


async def cmd_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_scan_job
    # інтервал у хвилинах з аргумента або дефолт
    try:
        minutes = int(context.args[0]) if context.args else DEFAULT_SCAN_MIN
        minutes = max(1, minutes)
    except Exception:
        minutes = DEFAULT_SCAN_MIN

    # знімаємо старий job, якщо був
    if auto_scan_job:
        try:
            scheduler.remove_job(auto_scan_job.id)
        except Exception:
            pass
        auto_scan_job = None

    auto_scan_job = scheduler.add_job(auto_scan_tick, "interval", minutes=minutes, next_run_time=None)
    await update.message.reply_text(f"✅ Автоскан включён: каждые {minutes} мин.")


async def cmd_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_scan_job
    if auto_scan_job:
        try:
            scheduler.remove_job(auto_scan_job.id)
        except Exception:
            pass
        auto_scan_job = None
    await update.message.reply_text("⛔️ Автоскан выключен.")


# ------------ Main ------------
async def main():
    global app, scheduler, auto_scan_job

    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .rate_limiter(AIORateLimiter())
        .build()
    )

    # Команди
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("signals", cmd_signals))
    app.add_handler(CommandHandler("trade_on", cmd_trade_on))
    app.add_handler(CommandHandler("trade_off", cmd_trade_off))
    app.add_handler(CommandHandler("set_size", cmd_set_size))
    app.add_handler(CommandHandler("set_lev", cmd_set_lev))
    app.add_handler(CommandHandler("set_risk", cmd_set_risk))
    app.add_handler(CommandHandler("auto_on", cmd_auto_on))
    app.add_handler(CommandHandler("auto_off", cmd_auto_off))

    # Планувальник
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.start()

    # Пульс (для логів/живості)
    scheduler.add_job(lambda: asyncio.create_task(heartbeat(None)), "interval", minutes=60)

    L.info("Starting bot…")
    await app.initialize()
    await app.start()
    # PTB v20: start polling
    await app.updater.start_polling()

    # Не виходимо
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
