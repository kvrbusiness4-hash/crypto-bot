# -*- coding: utf-8 -*-

"""
bot_alpaca.py
Повний каркас телеграм-бота з Alpaca, із виправленим керуванням event loop.
- Без конфліктів "This event loop is already running"
- Підтримка режимів: scalp / aggressive / safe
- Стан по чатах у STATE
- Місця для твоєї логіки сканера/автотрейду позначені TODO
"""

import os
import math
import json
import asyncio
import time
from typing import Dict, Any, Optional, Tuple, List

import aiohttp
from aiohttp import ClientSession, ClientTimeout

from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    JobQueue,
)

# ========= ENV =========
TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_TOKEN") or "").strip()

ALPACA_API_KEY   = (os.getenv("ALPACA_API_KEY") or "").strip()
ALPACA_API_SECRET= (os.getenv("ALPACA_API_SECRET") or "").strip()

ALPACA_BASE_URL  = (os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_URL  = (os.getenv("ALPACA_DATA_URL") or "https://data-api.alpaca.markets").rstrip("/")

# Номінали/ліміти за замовчуванням — можеш підкрутити/замість цього брати з ENV
ALPACA_NOTIONAL = float(os.getenv("ALPACA_NOTIONAL") or 200)
ALPACA_TOP_N    = int(os.getenv("ALPACA_TOP_N") or 25)

# інтервали автоскану
SCAN_INTERVAL_SEC    = int(os.getenv("SCAN_INTERVAL_SEC") or 300)
DEDUP_COOLDOWN_MIN   = int(os.getenv("DEDUP_COOLDOWN_MIN") or 240)

# ====== ГЛОБАЛ STATE (per chat) ======
STATE: Dict[int, Dict[str, Any]] = {}

# ===== РЕЖИМИ (прикладні параметри – коригуй під себе) =====
MODE_PARAMS = {
    "aggressive": {
        "bars": ("15Min", "30Min", "1Hour"),
        "rsi_buy": 55.0, "rsi_sell": 45.0,
        "ema_fast": 15, "ema_slow": 30,
        "top_n": ALPACA_TOP_N,
        "tp_pct": 0.015,   # 1.5%
        "sl_pct": 0.008,   # 0.8%
    },
    "scalp": {
        "bars": ("5Min", "15Min", "1Hour"),
        "rsi_buy": 58.0, "rsi_sell": 42.0,
        "ema_fast": 9, "ema_slow": 21,
        "top_n": ALPACA_TOP_N,
        "tp_pct": 0.006,   # 0.6%
        "sl_pct": 0.0035,  # 0.35%
    },
    "safe": {
        "bars": ("30Min", "1Hour", "1Day"),
        "rsi_buy": 52.0, "rsi_sell": 48.0,
        "ema_fast": 20, "ema_slow": 50,
        "top_n": ALPACA_TOP_N,
        "tp_pct": 0.01,    # 1%
        "sl_pct": 0.006,   # 0.6%
    },
}
DEFAULT_MODE = "scalp"

# ====== HTTP session (створюється в main) ======
HTTP: Optional[ClientSession] = None

# ====== Хелпери ======
def _chat_state(chat_id: int) -> Dict[str, Any]:
    """Ініт стан чату за потреби."""
    if chat_id not in STATE:
        STATE[chat_id] = {
            "mode": DEFAULT_MODE,
            "autotrade": False,
            "autoscan": False,
            "side": "long",
            "interval": SCAN_INTERVAL_SEC,
            "last_signals_at": 0.0,
        }
    return STATE[chat_id]

def alpaca_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

async def alpaca_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    assert HTTP is not None, "HTTP session not initialized"
    url = f"{ALPACA_BASE_URL}{path}"
    async with HTTP.get(url, headers=alpaca_headers(), params=params) as r:
        if r.status >= 400:
            txt = await r.text()
            raise RuntimeError(f"GET {path} {r.status}: {txt}")
        return await r.json()

async def alpaca_post(path: str, payload: Dict[str, Any]) -> Any:
    assert HTTP is not None, "HTTP session not initialized"
    url = f"{ALPACA_BASE_URL}{path}"
    async with HTTP.post(url, headers=alpaca_headers(), data=json.dumps(payload)) as r:
        txt = await r.text()
        if r.status >= 400:
            raise RuntimeError(f"POST {path} {r.status}: {txt}")
        return json.loads(txt) if txt else {}

# ====== Ваші ДАНІ / СИГНАЛИ (TODO встав свої реалізації) ======
async def scan_crypto_symbols(state: Dict[str, Any]) -> List[str]:
    """
    TODO: тут твоя логіка відбору топ-криптопар (за обсягом/сигналами/фільтрами).
    Поверни список символів типу ["AAVE/USD","AVAX/USD","BAT/USD"] або пустий.
    """
    # приклад-заглушка: повернемо 3 символи для демо
    return ["AAVE/USD", "AVAX/USD", "BAT/USD"]

async def scan_stock_symbols(state: Dict[str, Any]) -> List[str]:
    """
    TODO: тут твоя логіка відбору акцій (топ-N, фільтри тощо).
    Поверни список тикерів, напр. ["AAPL","AMAT","ADBE"]
    """
    return ["AAPL", "AMAT", "ADBE"]

# ====== Телеграм хелпери ======
def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            ["/aggressive", "/scalp", "/safe"],
            ["/signals_crypto", "/signals_stocks"],
            ["/alp_on", "/alp_off", "/alp_status"],
            ["/auto_on", "/auto_off", "/auto_status"],
        ],
        resize_keyboard=True
    )

async def send_info(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str) -> None:
    try:
        await ctx.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML)
    except Exception:
        # тихо, щоб не роняти job
        pass

# ====== HANDLERS ======
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = _chat_state(update.effective_chat.id)
    await update.message.reply_text(
        "Крипта торгується 24/7; акції — коли ринок відкритий.\n"
        "Сканер/автотрейд може працювати у фоні.\n"
        "Увімкнути автотрейд: /alp_on  ·  Зупинити: /alp_off  ·  Стан: /alp_status\n"
        "Фоновий автоскан: /auto_on  ·  /auto_off  ·  /auto_status",
        reply_markup=main_keyboard(),
    )

async def mode_set(update: Update, context: ContextTypes.DEFAULT_TYPE, name: str) -> None:
    st = _chat_state(update.effective_chat.id)
    st["mode"] = name
    await update.message.reply_text(f"Режим встановлено: {name.upper()}", reply_markup=main_keyboard())

async def aggressive_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await mode_set(update, context, "aggressive")

async def scalp_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await mode_set(update, context, "scalp")

async def safe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await mode_set(update, context, "safe")

async def alp_status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        acc = await alpaca_get("/v2/account")
        st = _chat_state(update.effective_chat.id)
        txt = (
            "📦 Alpaca:\n"
            f"• status=<b>{acc.get('status','?').upper()}</b>\n"
            f"• cash=${acc.get('cash','?')}\n"
            f"• buying_power=${acc.get('buying_power','?')}\n"
            f"• equity=${acc.get('equity','?')}\n"
            f"Mode=<b>{st['mode']}</b> · Autotrade=<b>{'ON' if st['autotrade'] else 'OFF'}</b> · "
            f"AutoScan=<b>{'ON' if st['autoscan'] else 'OFF'}</b> · Side=<b>{st['side']}</b>"
        )
        await update.message.reply_text(txt, parse_mode=ParseMode.HTML, reply_markup=main_keyboard())
    except Exception as e:
        await update.message.reply_text(f"❌ alp_status error: {e}")

async def alp_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = _chat_state(update.effective_chat.id)
    st["autotrade"] = True
    await update.message.reply_text("✅ Alpaca AUTOTRADE: ON")

async def alp_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = _chat_state(update.effective_chat.id)
    st["autotrade"] = False
    await update.message.reply_text("⛔ Alpaca AUTOTRADE: OFF")

async def auto_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = _chat_state(update.effective_chat.id)
    st["autoscan"] = True
    await update.message.reply_text(f"✅ AUTO-SCAN: ON (кожні {st['interval']}s)")

async def auto_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = _chat_state(update.effective_chat.id)
    st["autoscan"] = False
    await update.message.reply_text("⛔ AUTO-SCAN: OFF")

async def auto_status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = _chat_state(update.effective_chat.id)
    await update.message.reply_text(
        f"AutoScan={'ON' if st['autoscan'] else 'OFF'}; Autotrade={'ON' if st['autotrade'] else 'OFF'}; "
        f"Mode={st['mode']} · Side={st['side']} · Interval={st['interval']}s"
    )

# ====== SIGNALS (викликають твій сканер) ======
async def signals_crypto_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    st = _chat_state(chat_id)

    try:
        symbols = await scan_crypto_symbols(st)
        top = symbols[: min(len(symbols), 25)]
        use_n = min( len(top),  max(1,  st.get("use_for_trade",  st.get("limit", 3))) )
        head = f"🦅 Сканер (крипта):\n• Активних USD-пар: {len(symbols)}\n• Використаємо для торгівлі (лімітом): {use_n}\n• Перші 25: {', '.join(top)}"
        await update.message.reply_text(head)

        # TODO: тут твій код входу в позицію/виставлення SL/TP
        # Можеш викликати свою функцію trade_crypto(symbol, st)
        # і в ній робити все як у твоєму попередньому файлі.
    except Exception as e:
        await update.message.reply_text(f"🔴 signals_crypto error: {e}")

async def signals_stocks_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    st = _chat_state(chat_id)

    try:
        symbols = await scan_stock_symbols(st)
        top = symbols[: min(len(symbols), 25)]
        use_n = min( len(top),  max(1,  st.get("use_for_trade",  st.get("limit", 3))) )
        head = f"📡 Сканер (акції):\n• Символів у списку: {len(symbols)}\n• Використаємо для торгівлі (лімітом): {use_n}\n• Перші 25: {', '.join(top)}"
        await update.message.reply_text(head)

        # TODO: тут твій код входу/брекети/SL/TP для акцій
    except Exception as e:
        await update.message.reply_text(f"🔴 signals_stocks error: {e}")

# ====== Фоновий цикл сканера (JobQueue) ======
async def scanner_loop(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Запускається джобою раз у N секунд для КОЖНОГО чату, де autoscan=True."""
    now = time.time()
    for chat_id, st in list(STATE.items()):
        if not st.get("autoscan"):
            continue
        # холодний даунліміт, щоб не спамити
        if now - st.get("last_signals_at", 0) < st.get("interval", SCAN_INTERVAL_SEC) - 2:
            continue
        st["last_signals_at"] = now

        try:
            # TODO: тут можеш викликати свій комбінований автоскан із входом у позиції
            # Напр., спочатку crypto, потім stocks — згідно з твоїми правилами:
            # symbols = await scan_crypto_symbols(st)
            # ... твій автотрейд-код ...
            await send_info(ctx, chat_id, "🟢 AUTO-SCAN tick...")
        except Exception as e:
            await send_info(ctx, chat_id, f"🔴 AUTO-SCAN error: {e}")

# ====== MAIN ======
async def main() -> None:
    global HTTP
    if not TG_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

    HTTP = aiohttp.ClientSession(timeout=ClientTimeout(total=30))

    app = Application.builder().token(TG_TOKEN).build()

    # Команди
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("alp_status", alp_status_cmd))
    app.add_handler(CommandHandler("alp_on", alp_on_cmd))
    app.add_handler(CommandHandler("alp_off", alp_off_cmd))

    app.add_handler(CommandHandler("auto_on", auto_on_cmd))
    app.add_handler(CommandHandler("auto_off", auto_off_cmd))
    app.add_handler(CommandHandler("auto_status", auto_status_cmd))

    app.add_handler(CommandHandler("aggressive", aggressive_cmd))
    app.add_handler(CommandHandler("scalp", scalp_cmd))
    app.add_handler(CommandHandler("safe", safe_cmd))

    app.add_handler(CommandHandler("signals_crypto", signals_crypto_cmd))
    app.add_handler(CommandHandler("signals_stocks", signals_stocks_cmd))

    # Фонова джоба (одна на застосунок). Всередині вона проходиться по STATE.
    app.job_queue.run_repeating(scanner_loop, interval=5, first=5)  # 5s tick, ти всередині контролиш частоту на чат

    print("Bot started.")
    await app.initialize()
    await app.start()
    # PTB v20+: для polling
    await app.updater.start_polling()
    await app.updater.idle()

    # Graceful shutdown
    await app.stop()
    await app.shutdown()
    if HTTP:
        await HTTP.close()

if __name__ == "__main__":
    asyncio.run(main())
