# bot_alpaca.py
# -*- coding: utf-8 -*-

import os
import json
from typing import Dict, Any, Tuple, List, Optional

from aiohttp import ClientSession, ClientTimeout

from telegram import (
    Update,
    ReplyKeyboardMarkup,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# =========================
# ENV
# =========================
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or os.getenv("TELEGRAM_TOKEN", "").strip()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "").strip()

# базові URL (без зайвих слешів у кінці)
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")

# налаштування
ALPACA_NOTIONAL = float(os.getenv("ALPACA_NOTIONAL", "50") or 50.0)     # $ на ордер
ALPACA_MAX_STOCKS = int(os.getenv("ALPACA_MAX_STOCKS", "0") or 0)       # скільки акцій купувати (0 = не купуємо)
ALPACA_MAX_CRYPTO = int(os.getenv("ALPACA_MAX_CRYPTO", "50") or 50)     # скільки crypto купувати
SCAN_EVERY_SEC = int(os.getenv("SCAN_EVERY_SEC", "120") or 120)

# =========================
# СТАН НА ЧАТ
# =========================
def default_state() -> Dict[str, Any]:
    return {
        "autotrade": False,
        "mode": "default",
        "last_scan_txt": "",
    }

STATE: Dict[int, Dict[str, Any]] = {}

def stedef(chat_id: int) -> Dict[str, Any]:
    return STATE.setdefault(chat_id, default_state())

# =========================
# КНОПКИ
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
# HTTP helpers
# =========================
def alp_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }

def trade_url(path: str) -> str:
    path = path.lstrip("/")
    return f"{ALPACA_BASE_URL}/v2/{path}"

def data_url(path: str) -> str:
    path = path.lstrip("/")
    return f"{ALPACA_DATA_URL}/{path}"

async def http_get_json(session: ClientSession, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    async with session.get(url, headers=alp_headers(), params=params, timeout=ClientTimeout(total=30)) as r:
        if r.status >= 400:
            txt = await r.text()
            raise RuntimeError(f"GET {r.url} {r.status}: {txt}")
        return await r.json()

async def http_post_json(session: ClientSession, url: str, payload: Dict[str, Any]) -> Any:
    async with session.post(url, headers=alp_headers(), data=json.dumps(payload), timeout=ClientTimeout(total=30)) as r:
        if r.status >= 400:
            txt = await r.text()
            raise RuntimeError(f"POST {r.url} {r.status}: {txt}")
        return await r.json()

# =========================
# Alpaca trading API
# =========================
async def alp_account() -> Dict[str, Any]:
    async with ClientSession() as s:
        return await http_get_json(s, trade_url("account"))

async def place_notional_order(symbol: str, side: str, notional: float) -> Dict[str, Any]:
    payload = {
        "symbol": symbol,
        "side": side,                 # "buy" | "sell"
        "type": "market",
        "time_in_force": "gtc",
        "notional": str(float(notional)),
    }
    async with ClientSession() as s:
        return await http_post_json(s, trade_url("orders"), payload)

# =========================
# DATA: список усіх активних crypto
# =========================
async def fetch_active_crypto_symbols(limit: int = 2000) -> List[str]:
    """
    Тягнемо всі активні крипто-символи через ОФІЦІЙНИЙ endpoint:
      GET https://data.alpaca.markets/v2/assets?asset_class=crypto&status=active
    """
    out: List[str] = []
    async with ClientSession() as s:
        resp = await http_get_json(s, data_url("v2/assets"), params={"asset_class": "crypto", "status": "active"})
        # віддає масив активів; беремо поле symbol
        for a in resp:
            sym = a.get("symbol")
            if sym:
                out.append(sym)
                if len(out) >= limit:
                    break
    return out

# =========================
# СКАНЕР (демо-логіка: просто тягнемо весь список crypto)
# =========================
async def scan_all(st: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    crypto = await fetch_active_crypto_symbols()
    crypto_sorted = sorted(set(crypto))
    # picks: просто перші N (щоб не палити депозит без твоїх фільтрів)
    picks_c = crypto_sorted[:ALPACA_MAX_CRYPTO] if ALPACA_MAX_CRYPTO > 0 else []
    # акції наразі не скануємо
    picks_s: List[str] = []

    rep_lines = [
        "🛰 Сканер Alpaca:",
        f"• Усього активних crypto: {len(crypto_sorted)}",
        f"• Буде використано для торгівлі (за лімітом): {len(picks_c)}",
    ]
    if picks_c:
        sample = ", ".join(picks_c[:25])
        rep_lines.append(f"• Перші {min(25, len(picks_c))}: {sample}")
    report = "\n".join(rep_lines)
    return report, picks_s, picks_c

# =========================
# КОМАНДИ
# =========================
async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    stedef(u.effective_chat.id)
    txt = (
        "👋 Готово. Бот видає сигнали та (за бажанням) ставить ордери в **Alpaca**.\n"
        "• /alp_on — увімкнути автотрейд\n"
        "• /alp_off — вимкнути автотрейд\n"
        "• /alp_status — стан акаунту\n"
        "• /signals_alpaca — ручний скан зараз\n\n"
        "Крипта торгується 24/7."
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
    try:
        rep, picks_s, picks_c = await scan_all(st)
        st["last_scan_txt"] = rep

        # надсилаємо звіт (порціями, щоб не впертися в ліміт)
        chunks = [rep[i:i+3500] for i in range(0, len(rep), 3500)] or [rep]
        for ch in chunks:
            await u.message.reply_text(ch, parse_mode=ParseMode.MARKDOWN)

        # автотрейд
        if st.get("autotrade"):
            bought = 0
            for sym in picks_c:
                try:
                    await place_notional_order(sym, "buy", ALPACA_NOTIONAL)
                    bought += 1
                    await u.message.reply_text(f"🟢 ORDER OK: {sym} ${ALPACA_NOTIONAL:.2f}")
                except Exception as e:
                    await u.message.reply_text(f"🔴 ORDER FAIL {sym}: {e}")
            if bought == 0:
                await u.message.reply_text("ℹ️ Немає обраних символів для купівлі (перевір ліміти).")
    except Exception as e:
        await u.message.reply_text(f"🔴 scan error: {e}")

# =========================
# ФОНОВИЙ СКАНЕР (через JobQueue)
# =========================
async def periodic_scan_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    for chat_id, st in list(STATE.items()):
        try:
            rep, picks_s, picks_c = await scan_all(st)
            st["last_scan_txt"] = rep
            if st.get("autotrade") and picks_c:
                for sym in picks_c:
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
        "Сканер крипти працює 24/7.",
        reply_markup=main_keyboard()
    )

# =========================
# MAIN
# =========================
def main():
    if not TG_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задано")

    app = Application.builder().token(TG_TOKEN).build()

    # handlers
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))

    app.add_handler(CommandHandler("aggressive", aggressive_cmd))
    app.add_handler(CommandHandler("scalp", scalp_cmd))
    app.add_handler(CommandHandler("default", default_cmd))
    app.add_handler(CommandHandler("swing", swing_cmd))
    app.add_handler(CommandHandler("safe", safe_cmd))

    app.add_handler(CommandHandler("alp_on", alp_on_cmd))
    app.add_handler(CommandHandler("alp_off", alp_off_cmd))
    app.add_handler(CommandHandler("alp_status", alp_status_cmd))
    app.add_handler(CommandHandler("signals_alpaca", signals_alpaca_cmd))

    # фонова задача
    app.job_queue.run_repeating(periodic_scan_job, interval=SCAN_EVERY_SEC, first=5)

    # запуск
    app.run_polling()

if __name__ == "__main__":
    main()
