# bot_alpaca.py
# -*- coding: utf-8 -*-

import os
import asyncio
import json
from typing import Dict, Any, Tuple, List

import aiohttp
from aiohttp import ClientSession, ClientTimeout

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# =========================
# ENV
# =========================
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or os.getenv("TELEGRAM_TOKEN", "").strip()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "").strip()
# БАЗОВИЙ endpoint БЕЗ зайвих /v2 у кінці
ALPACA_BASE_URL = (os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/"))
ALPACA_DATA_URL = (os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/"))

ALPACA_NOTIONAL = float(os.getenv("ALPACA_NOTIONAL", "50") or 50)  # сума на ордер
SCAN_EVERY_SEC = int(os.getenv("SCAN_EVERY_SEC", "120") or 120)

# обмеження на кількість інструментів у сигналі (для безпеки)
ALPACA_MAX_STOCKS = int(os.getenv("ALPACA_MAX_STOCKS", "300") or 300)
ALPACA_MAX_CRYPTO = int(os.getenv("ALPACA_MAX_CRYPTO", "50") or 50)

SIGLOG_PATH = os.getenv("SIGLOG_PATH", "signals_log.csv")

# =========================
# СТАН НА ЧАТ
# =========================
def default_state() -> Dict[str, Any]:
    return {
        "autotrade": False,
        "mode": "default",         # профіль ризику (поки що інформативно)
        "last_scan_txt": "",
    }

STATE: Dict[int, Dict[str, Any]] = {}

def stedef(chat_id: int) -> Dict[str, Any]:
    return STATE.setdefault(chat_id, default_state())

# =========================
# КОРИСТУВАЦЬКІ КНОПКИ
# =========================
def main_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        ["/aggressive", "/scalp", "/default"],
        ["/swing", "/safe", "/help"],
        ["/signals_alpaca"],
        ["/alp_on", "/alp_status", "/alp_off"],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# =========================
# HTTP / Alpaca helpers
# =========================
def alp_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }

def alp_url(path: str) -> str:
    # усі трейд-ендпоїнти — під /v2
    path = path.lstrip("/")
    return f"{ALPACA_BASE_URL}/v2/{path}"

async def alp_get(session: ClientSession, path: str) -> Any:
    async with session.get(alp_url(path), headers=alp_headers(), timeout=ClientTimeout(total=30)) as r:
        if r.status >= 400:
            txt = await r.text()
            raise RuntimeError(f"Alpaca GET {r.url} {r.status}: {txt}")
        return await r.json()

async def alp_post(session: ClientSession, path: str, payload: Dict[str, Any]) -> Any:
    async with session.post(alp_url(path), headers=alp_headers(),
                            data=json.dumps(payload),
                            timeout=ClientTimeout(total=30)) as r:
        if r.status >= 400:
            txt = await r.text()
            raise RuntimeError(f"Alpaca POST {r.url} {r.status}: {txt}")
        return await r.json()

async def alp_account() -> Dict[str, Any]:
    timeout = ClientTimeout(total=30)
    async with ClientSession(timeout=timeout) as s:
        return await alp_get(s, "account")

async def place_notional_order(symbol: str, side: str, notional: float) -> Dict[str, Any]:
    """
    Універсальний ордер у доларовій сумі (USD). Працює і для акцій, і для crypto (24/7).
    Для крипти використовуйте символи виду 'BTC/USD', 'ETH/USD'.
    Для акцій — звичайні тікери 'AAPL', 'TSLA' і т.д.
    """
    payload = {
        "symbol": symbol,
        "side": side,                   # "buy" | "sell"
        "type": "market",
        "time_in_force": "gtc",
        "notional": str(float(notional)),
    }
    timeout = ClientTimeout(total=30)
    async with ClientSession(timeout=timeout) as s:
        return await alp_post(s, "orders", payload)

# =========================
# СКАНЕР (демо)
# =========================
async def scan_all(st: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    """
    Повертає:
      - текстовий звіт,
      - списки picks_s (stocks) і picks_c (crypto)
    Тут демо-сканер: щоб нічого не ламати — повертає порожні сигнали.
    Можеш вставити свою логіку: відбір за об'ємом, трендом, й т.д.
    """
    # TODO: додай реальну логіку сканування
    rep_lines = ["🛰 Сканер: наразі немає нових сигналів."]
    report = "\n".join(rep_lines)
    picks_s: List[str] = []  # приклад: ["AAPL", "TSLA"]
    picks_c: List[str] = []  # приклад: ["BTC/USD", "ETH/USD"]
    return report, picks_s, picks_c

# =========================
# КОМАНДИ БОТА
# =========================
async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(u.effective_chat.id)
    txt = (
        "👋 Готово. Бот видає сигнали та (за бажанням) ставить ордери в **Alpaca**.\n"
        "• /alp_on — увімкнути автотрейд\n"
        "• /alp_off — вимкнути автотрейд\n"
        "• /alp_status — стан акаунту\n"
        "• /signals_alpaca — ручний скан зараз\n\n"
        "Крипта торгується 24/7. Без перевірки торгової сесії."
    )
    await u.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN, reply_markup=main_keyboard())

async def aggressive_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(u.effective_chat.id)
    st["mode"] = "aggressive"
    await u.message.reply_text("✅ Mode: AGGRESSIVE", reply_markup=main_keyboard())

async def scalp_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(u.effective_chat.id)
    st["mode"] = "scalp"
    await u.message.reply_text("✅ Mode: SCALP", reply_markup=main_keyboard())

async def default_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(u.effective_chat.id)
    st["mode"] = "default"
    await u.message.reply_text("✅ Mode: DEFAULT", reply_markup=main_keyboard())

async def swing_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(u.effective_chat.id)
    st["mode"] = "swing"
    await u.message.reply_text("✅ Mode: SWING", reply_markup=main_keyboard())

async def safe_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(u.effective_chat.id)
    st["mode"] = "safe"
    await u.message.reply_text("✅ Mode: SAFE", reply_markup=main_keyboard())

async def alp_on_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(u.effective_chat.id)
    st["autotrade"] = True
    await u.message.reply_text("✅ Alpaca AUTOTRADE: ON", reply_markup=main_keyboard())

async def alp_off_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(u.effective_chat.id)
    st["autotrade"] = False
    await u.message.reply_text("⏹ Alpaca AUTOTRADE: OFF", reply_markup=main_keyboard())

async def alp_status_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        acc = await alp_account()
        txt = (
            "💼 Alpaca:\n"
            f"• status={acc.get('status','?')}\n"
            f"• cash=${float(acc.get('cash',0)):,.2f}\n"
            f"• buying_power=${float(acc.get('buying_power',0)):,.2f}\n"
            f"• equity=${float(acc.get('equity',0)):,.2f}"
        )
    except Exception as e:
        txt = f"❌ Alpaca error: {e}"
    await u.message.reply_text(txt, reply_markup=main_keyboard())

async def signals_alpaca_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(u.effective_chat.id)
    rep, picks_s, picks_c = await scan_all(st)
    st["last_scan_txt"] = rep

    # надсилаємо звіт (розбиваємо, якщо довгий)
    chunks = [rep[i:i+3500] for i in range(0, len(rep), 3500)] or [rep]
    for ch in chunks:
        await u.message.reply_text(ch, parse_mode=ParseMode.MARKDOWN)

    # автотрейд
    if st.get("autotrade"):
        try:
            # приклад: купити кожен сигнал на ALPACA_NOTIONAL
            for sym in (picks_s[:ALPACA_MAX_STOCKS] + picks_c[:ALPACA_MAX_CRYPTO]):
                try:
                    r = await place_notional_order(sym, "buy", ALPACA_NOTIONAL)
                    await u.message.reply_text(f"🟢 ORDER OK: {sym} ${ALPACA_NOTIONAL:.2f}")
                except Exception as e:
                    await u.message.reply_text(f"🔴 ORDER FAIL {sym}: {e}")
        except Exception as e:
            await u.message.reply_text(f"🔴 Autotrade error: {e}")

# =========================
# ФОНОВИЙ СКАНЕР (24/7)
# =========================
async def periodic_scan_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Працює через JobQueue — без проблем з event loop.
    Біжить завжди, крипта не залежить від сесії.
    """
    # проходимось по всіх чатах, де нас уже стартували
    for chat_id, st in list(STATE.items()):
        try:
            rep, picks_s, picks_c = await scan_all(st)
            st["last_scan_txt"] = rep
            if st.get("autotrade"):
                # спроба виставити ордери
                for sym in (picks_s[:ALPACA_MAX_STOCKS] + picks_c[:ALPACA_MAX_CRYPTO]):
                    try:
                        await place_notional_order(sym, "buy", ALPACA_NOTIONAL)
                        await ctx.bot.send_message(chat_id, f"🟢 ORDER OK: {sym} ${ALPACA_NOTIONAL:.2f}")
                    except Exception as e:
                        await ctx.bot.send_message(chat_id, f"🔴 ORDER FAIL {sym}: {e}")
        except Exception as e:
            try:
                await ctx.bot.send_message(chat_id, f"🔴 periodic_scan error: {e}")
            except Exception:
                pass

# =========================
# HELP
# =========================
async def help_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    await u.message.reply_text(
        "Команди:\n"
        "• /alp_on, /alp_off, /alp_status\n"
        "• /signals_alpaca — ручний скан\n"
        "• /aggressive /scalp /default /swing /safe — режим профілю\n"
        "Крипта 24/7, без перевірки торгової сесії.",
        reply_markup=main_keyboard()
    )
# ----- /alp_status -----
async def alp_status_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    try:
        acc = await alp_account()
        clk = await alp_clock()
        txt = (
            "🧳 Alpaca: "
            f"status={acc.get('status','?')}\n"
            f"• cash=${float(acc.get('cash',0)):.2f}\n"
            f"• buying_power=${float(acc.get('buying_power',0)):.2f}\n"
            f"• equity=${float(acc.get('equity',0)):.2f}\n"
            f"• market_open={'YES' if bool(clk.get('is_open')) else 'NO'}\n"
            f"Mode={st.get('mode')} · Autotrade={'ON' if st.get('autotrade') else 'OFF'}"
        )
    except Exception as e:
        txt = f"❌ Alpaca error: {e}"
    await u.message.reply_text(txt)
# ===== MAIN =====

def main():
    if not TG_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задано")

    app = Application.builder().token(TG_TOKEN).build()

    # handlers
    app.add_handler(CommandHandler("alp_status", alp_status_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("aggressive", aggressive_cmd))
    app.add_handler(CommandHandler("scalp", scalp_cmd))
    app.add_handler(CommandHandler("default", default_cmd))
    app.add_handler(CommandHandler("swing", swing_cmd))
    app.add_handler(CommandHandler("safe", safe_cmd))

    app.add_handler(CommandHandler("alp_on", alp_on_cmd))
    app.add_handler(CommandHandler("alp_off", alp_off_cmd))
    app.add_handler(CommandHandler("signals_alpaca", signals_cmd))

    # Фоновий сканер запускаємо лише через JobQueue
    app.job_queue.run_repeating(periodic_scan_job, interval=120, first=5)

    # Блокуючий запуск БЕЗ await і БЕЗ asyncio.run
    app.run_polling()

if __name__ == "__main__":
    main()
