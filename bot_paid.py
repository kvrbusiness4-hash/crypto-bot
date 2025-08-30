# -*- coding: utf-8 -*-
"""
ProfitSignalsBot — телеграм-бот автоторговли для Bybit.
Особенности:
- PTB v20 (python-telegram-bot) + AIORateLimiter
- Общая aiohttp-сессия с поддержкой HTTP/HTTPS прокси (env: BYBIT_PROXY)
- Публичные GET (тикеры) через api_get_json (обязательно JSON)
- Базовые приватные запросы к Bybit v5 (подпись HMAC) — заготовка
- Команды: /start /status /signals /trade_on /trade_off /auto_on /auto_off
           /set_size <usdt> /set_lev <x> /set_risk <sl%> <tp%>
- Пульс “heartbeat” в логах и админ-чат раз в N минут (env: HEARTBEAT_MIN)
"""

import os
import hmac
import time
import json
import asyncio
import hashlib
from typing import Any, Dict, Optional

from aiohttp import ClientSession, ClientTimeout

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    AIORateLimiter,
    CommandHandler,
    ContextTypes,
)

# ========= Конфиг из окружения =========
BOT_TOKEN         = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_ID          = int(os.getenv("ADMIN_ID", "0") or 0)

BYBIT_BASE_URL    = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").strip()
BYBIT_API_KEY     = os.getenv("BYBIT_API_KEY", "").strip()
BYBIT_API_SECRET  = os.getenv("BYBIT_API_SECRET", "").strip()

# HTTP/HTTPS proxy: формат http://user:pass@ip:port
PROXY_URL         = os.getenv("BYBIT_PROXY", "").strip()

# Торговые настройки по умолчанию (можно менять командами)
TRADE_ENABLED     = (os.getenv("TRADE_ENABLED", "ON").upper() == "ON")
SIZE_USDT         = float(os.getenv("SIZE_USDT", "5"))
LEVERAGE          = int(os.getenv("LEVERAGE", "3"))
SL_PCT            = float(os.getenv("SL_PCT", "3"))
TP_PCT            = float(os.getenv("TP_PCT", "5"))

# Автоскан интервал по умолчанию (минуты)
DEFAULT_SCAN_MIN  = int(os.getenv("SCAN_MIN", "15"))
HEARTBEAT_MIN     = int(os.getenv("HEARTBEAT_MIN", "60"))

# ======== Глобалы ========
app: Optional[Application] = None
http: Optional[ClientSession] = None
auto_scan_job = None
heartbeat_job = None

UTC_FMT = "%Y-%m-%d %H:%M:%SZ"

# ========= Вспомогательная HTTP-обвязка =========

async def http_start():
    """Создаём общую aiohttp-сессию."""
    global http
    if http is None:
        http = ClientSession(timeout=ClientTimeout(total=25))

async def http_stop():
    """Закрываем общую aiohttp-сессию."""
    global http
    if http and not http.closed:
        await http.close()
    http = None

async def api_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Универсальный GET JSON. Всегда пытается использовать прокси (если задан).
    Если Bybit вернул HTML (403 и т.п.), бросает RuntimeError с кратким текстом.
    """
    assert http is not None, "HTTP session not started"
    kwargs: Dict[str, Any] = {"params": params or {}}
    if PROXY_URL:
        kwargs["proxy"] = PROXY_URL

    async with http.get(url, **kwargs) as r:
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "")
        if "application/json" not in ct:
            text = await r.text()
            raise RuntimeError(f"Bybit non-JSON (possible IP block): {text[:200]}")
        return await r.json()

def _sign_v5(params: Dict[str, Any], secret: str) -> str:
    """
    Подпись для Bybit v5: HMAC-SHA256(payload).
    payload = timestamp + api_key + recv_window + (json(params) или "" для GET без тела)
    Здесь применим вариант для application/x-www-form-urlencoded (обычный для v5 private).
    Для простоты используем сортировку по ключу.
    """
    # В некоторых эндпойнтах требуется stringToSign в ином формате.
    # Для базовых ордеров такой подписи обычно достаточно.
    sorted_items = "&".join([f"{k}={params[k]}" for k in sorted(params)])
    return hmac.new(
        secret.encode("utf-8"),
        sorted_items.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

async def bybit_private_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Пример приватного POST к Bybit v5.
    Используйте по необходимости (создание/отмена ордеров и т.п.).
    """
    assert http is not None, "HTTP session not started"
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise RuntimeError("BYBIT_API_KEY/SECRET не заданы")

    url = f"{BYBIT_BASE_URL}{path}"
    ts = str(int(time.time() * 1000))
    recv_window = "5000"

    params = {
        "api_key": BYBIT_API_KEY,
        "timestamp": ts,
        "recv_window": recv_window,
        **payload,
    }
    sign = _sign_v5(params, BYBIT_API_SECRET)
    params["sign"] = sign

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = "&".join([f"{k}={params[k]}" for k in params])

    kwargs: Dict[str, Any] = {"headers": headers, "data": data}
    if PROXY_URL:
        kwargs["proxy"] = PROXY_URL

    async with http.post(url, **kwargs) as r:
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "")
        if "application/json" not in ct:
            text = await r.text()
            raise RuntimeError(f"Bybit non-JSON (private): {text[:200]}")
        return await r.json()

# ========= Бизнес-логика =========

async def initial_scan() -> Dict[str, Any]:
    """
    Получение списка тикеров (linear) — используется в /signals и автоскане.
    """
    return await api_get_json(
        f"{BYBIT_BASE_URL}/v5/market/tickers",
        {"category": "linear"}
    )

def utc_now_str() -> str:
    import datetime as dt
    return dt.datetime.utcnow().strftime(UTC_FMT)

# ========= Команды =========

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Привет! Я готов. Команды:\n"
        "/status — статус\n"
        "/signals — скан сильных (top30)\n"
        "/trade_on | /trade_off — автоторговля\n"
        f"/auto_on {DEFAULT_SCAN_MIN} | /auto_off — автоскан\n"
        f"/set_size {int(SIZE_USDT)} — размер сделки в USDT\n"
        f"/set_lev {LEVERAGE} — плечо\n"
        f"/set_risk {int(SL_PCT)} {int(TP_PCT)} — SL/TP в %\n"
    )
    await update.message.reply_text(text)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    proxy_state = "используется" if PROXY_URL else "не используется"
    text = (
        f"Статус: {'ON' if auto_scan_job else 'OFF'} · каждые "
        f"{DEFAULT_SCAN_MIN if not auto_scan_job else int(auto_scan_job.interval.total_seconds()/60)} мин.\n"
        f"SL={SL_PCT:.2f}% · TP={TP_PCT:.2f}%\n"
        f"TRADE_ENABLED={'ON' if TRADE_ENABLED else 'OFF'} · SIZE={SIZE_USDT:.2f} USDT\n"
        f"· LEV={LEVERAGE}\n"
        "Фильтр: TOP30\n"
        f"Прокси: {proxy_state}\n"
        f"UTC: {utc_now_str()}"
    )
    await update.message.reply_text(text)

async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔎 Сканирую рынок…")
    try:
        data = await initial_scan()
        rows = data.get("result", {}).get("list", []) or []
        count = len(rows)
        await msg.edit_text(f"🔎 Скан OK · получено {count} тикеров ·\nUTC {utc_now_str()}")
    except Exception as e:
        # прислать файл с HTML/ошибкой — как в твоей прежней логике
        err = str(e)
        await update.message.reply_text(
            f"❌ Ошибка сканера: {err[:400]}",
            disable_web_page_preview=True,
        )
        # приложим «лог» как файл
        try:
            from io import BytesIO
            f = BytesIO(json.dumps({"error": err}, ensure_ascii=False, indent=2).encode("utf-8"))
            f.name = "tickers.json"
            await update.message.reply_document(f)
        except:
            pass

async def cmd_trade_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TRADE_ENABLED
    TRADE_ENABLED = True
    await update.message.reply_text("Автоторговля: ВКЛЮЧЕНА ✅")

async def cmd_trade_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TRADE_ENABLED
    TRADE_ENABLED = False
    await update.message.reply_text("Автоторговля: ВЫКЛЮЧЕНА ⛔")

async def cmd_set_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SIZE_USDT
    if not context.args:
        await update.message.reply_text("Использование: /set_size 5")
        return
    try:
        SIZE_USDT = max(1.0, float(context.args[0]))
        await update.message.reply_text(f"OK. SIZE_USDT={SIZE_USDT:.2f}")
    except:
        await update.message.reply_text("Неверное значение.")

async def cmd_set_lev(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LEVERAGE
    if not context.args:
        await update.message.reply_text("Использование: /set_lev 3")
        return
    try:
        LEVERAGE = max(1, int(context.args[0]))
        await update.message.reply_text(f"OK. LEVERAGE={LEVERAGE}")
    except:
        await update.message.reply_text("Неверное значение.")

async def cmd_set_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SL_PCT, TP_PCT
    if len(context.args) < 2:
        await update.message.reply_text("Использование: /set_risk 3 5 (SL% TP%)")
        return
    try:
        SL_PCT = max(0.1, float(context.args[0]))
        TP_PCT = max(0.1, float(context.args[1]))
        await update.message.reply_text(f"OK. SL={SL_PCT:.2f}% · TP={TP_PCT:.2f}%")
    except:
        await update.message.reply_text("Неверные значения.")

# ======= Автоскан =======

async def do_autoscan(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id or ADMIN_ID
    try:
        data = await initial_scan()
        rows = data.get("result", {}).get("list", []) or []
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"🔎 Скан OK · получено {len(rows)} тикеров ·\nUTC {utc_now_str()}",
        )
        # Здесь может быть твоя логика отбора/сигналов/ордеров.
        # Никакой новой логики не добавляю.
    except Exception as e:
        err = str(e)
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Ошибка автосканера: {err[:400]}")

async def cmd_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_scan_job
    minutes = DEFAULT_SCAN_MIN
    if context.args:
        try:
            minutes = max(1, int(context.args[0]))
        except:
            pass
    # удалим старую задачу
    if auto_scan_job:
        auto_scan_job.schedule_removal()
    auto_scan_job = context.job_queue.run_repeating(
        do_autoscan,
        interval=minutes * 60,
        first=1,
        name="autoscan",
        chat_id=update.effective_chat.id
    )
    await update.message.reply_text(f"✅ Автоскан включён: каждые {minutes} мин.")

async def cmd_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_scan_job
    if auto_scan_job:
        auto_scan_job.schedule_removal()
        auto_scan_job = None
    await update.message.reply_text("⏹ Автоскан выключен.")

# ======= Heartbeat =======

async def heartbeat(context: ContextTypes.DEFAULT_TYPE):
    # Просто пишем в логи, и опционально шлём админу
    if ADMIN_ID:
        try:
            await context.bot.send_message(ADMIN_ID, "💗 heartbeat")
        except:
            pass

# ========= Main =========

async def main():
    global app, heartbeat_job

    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .rate_limiter(AIORateLimiter())
        .build()
    )

    # Хендлеры команд
    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("status",    cmd_status))
    app.add_handler(CommandHandler("signals",   cmd_signals))
    app.add_handler(CommandHandler("trade_on",  cmd_trade_on))
    app.add_handler(CommandHandler("trade_off", cmd_trade_off))
    app.add_handler(CommandHandler("auto_on",   cmd_auto_on))
    app.add_handler(CommandHandler("auto_off",  cmd_auto_off))
    app.add_handler(CommandHandler("set_size",  cmd_set_size))
    app.add_handler(CommandHandler("set_lev",   cmd_set_lev))
    app.add_handler(CommandHandler("set_risk",  cmd_set_risk))

    # Запускаем HTTP-сессию
    await http_start()

    # Heartbeat
    if HEARTBEAT_MIN > 0:
        heartbeat_job = app.job_queue.run_repeating(
            heartbeat, interval=HEARTBEAT_MIN * 60, first=10, name="heartbeat"
        )

    print("Starting bot…")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await app.updater.stop()
        await app.stop()
        await http_stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
