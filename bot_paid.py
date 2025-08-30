# -*- coding: utf-8 -*-
"""
ProfitSignalsBot — мінімальний варіант з проксі для Bybit.
Правка: усі HTTP-запити до Bybit йдуть через проксі (env: BYBIT_PROXY).
"""

import os
import asyncio
import json
from datetime import datetime

import aiohttp
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# ========= Конфіг =========
BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_ID    = os.getenv("ADMIN_ID")  # не обовʼязково
BYBIT_PROXY = os.getenv("BYBIT_PROXY")  # напр.: socks5://user:pass@ip:port
BYBIT_BASE  = os.getenv("BYBIT_BASE", "https://api.bybit.com")

SCAN_MINUTES_DEFAULT = 15  # інтервал автоскану за замовчуванням

# стан автосканера в памʼяті процеса
state = {
    "auto_on": False,
    "auto_task": None,
    "minutes": SCAN_MINUTES_DEFAULT,
}

# ========= Утиліти =========
def _utcnow_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")


async def bybit_request(path: str, params: dict | None = None, method: str = "GET") -> dict:
    """HTTP-запит до Bybit через проксі (якщо BYBIT_PROXY заданий)."""
    url = f"{BYBIT_BASE}{path}"
    params = params or {}

    # важливо: proxy передаємо ПРЯМО в запит
    kwargs = {"proxy": BYBIT_PROXY, "timeout": aiohttp.ClientTimeout(total=15)}

    async with aiohttp.ClientSession() as session:
        try:
            if method.upper() == "GET":
                async with session.get(url, params=params, **kwargs) as resp:
                    # інколи замість JSON приходить html (коли IP блочать) — підхопимо це
                    ct = resp.headers.get("Content-Type", "")
                    text = await resp.text()
                    if "application/json" not in ct:
                        return {"error": "non_json", "content_type": ct, "text": text[:400]}
                    return json.loads(text)
            else:
                async with session.post(url, json=params, **kwargs) as resp:
                    ct = resp.headers.get("Content-Type", "")
                    text = await resp.text()
                    if "application/json" not in ct:
                        return {"error": "non_json", "content_type": ct, "text": text[:400]}
                    return json.loads(text)
        except Exception as e:
            return {"error": str(e)}


async def scan_top30_once() -> str:
    """
    Простий сканер: тягне тікери 'linear' і перевіряє, що API доступний через проксі.
    Повертає короткий текст для повідомлення.
    """
    data = await bybit_request("/v5/market/tickers", {"category": "linear"})
    if "error" in data:
        if data["error"] == "non_json":
            return (f"❌ Помилка сканера: Bybit non-JSON (можливий IP block)\n"
                    f"{data.get('text','')}")
        return f"❌ Помилка сканера: {data['error']}"

    # базова перевірка валідності
    result = data.get("result", {})
    list_ = result.get("list", [])
    count = len(list_)
    return f"🔍 Скан OK · отримано {count} тікерів · UTC { _utcnow_str() }"


# ========= Команди =========
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привіт! Бот запущено.\n"
        "Команди:\n"
        "/status — стан\n"
        "/signals — разовий скан\n"
        "/auto_on 15 — автоскан кожні N хв\n"
        "/auto_off — вимкнути автоскан"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = [
        f"Статус: {'ON' if state['auto_on'] else 'OFF'} · кожні {state['minutes']} хв.",
        f"Проксі: {'використовується' if BYBIT_PROXY else 'не задано'}",
        f"UTC: { _utcnow_str() }",
    ]
    await update.message.reply_text("\n".join(text))


async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔎 Сканую ринок…")
    info = await scan_top30_once()
    await update.message.reply_text(info)


async def _auto_loop(context: ContextTypes.DEFAULT_TYPE):
    # ця функція викликається у фоні кожні N хвилин
    info = await scan_top30_once()
    chat_id = context.job.chat_id if context.job else (ADMIN_ID if ADMIN_ID else None)
    if chat_id:
        try:
            await context.bot.send_message(chat_id=chat_id, text=info)
        except Exception:
            pass


async def cmd_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /auto_on 15
    minutes = SCAN_MINUTES_DEFAULT
    if context.args:
        try:
            minutes = max(1, int(context.args[0]))
        except Exception:
            pass

    # прибираємо попередню джобу якщо була
    if state["auto_task"]:
        state["auto_task"].cancel()

    # створюємо нову періодичну задачу через JobQueue (PTB v20)
    job_queue = context.application.job_queue
    job = job_queue.run_repeating(_auto_loop, interval=minutes * 60, first=5, chat_id=update.effective_chat.id)

    state["auto_task"] = job
    state["auto_on"] = True
    state["minutes"] = minutes

    await update.message.reply_text(f"✅ Автоскан увімкнено: кожні {minutes} хв.")


async def cmd_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if state["auto_task"]:
        state["auto_task"].cancel()
    state["auto_task"] = None
    state["auto_on"] = False
    await update.message.reply_text("🛑 Автоскан вимкнено.")


# ========= Main =========
async def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("signals", cmd_signals))
    app.add_handler(CommandHandler("auto_on", cmd_auto_on))
    app.add_handler(CommandHandler("auto_off", cmd_auto_off))

    print("🚀 Бот запускається…")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    # не завершуємо процес
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
