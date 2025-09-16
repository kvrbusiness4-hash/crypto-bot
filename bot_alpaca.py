# bot_alpaca.py
# -*- coding: utf-8 -*-

import os
import asyncio
import json
from typing import Dict, Any, Tuple, List, Optional

import aiohttp
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

# БЕЗ /v2 в кінці
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")

ALPACA_NOTIONAL = float(os.getenv("ALPACA_NOTIONAL", "50") or 50)
SCAN_EVERY_SEC = int(os.getenv("SCAN_EVERY_SEC", "120") or 120)

# ліміти/пороги
ALPACA_MAX_STOCKS = int(os.getenv("ALPACA_MAX_STOCKS", "300") or 300)
ALPACA_MAX_CRYPTO = int(os.getenv("ALPACA_MAX_CRYPTO", "200") or 200)
CRYPTO_MIN_D1_PCT = float(os.getenv("CRYPTO_MIN_D1_PCT", "2") or 2.0)  # поріг відбору, %

SIGLOG_PATH = os.getenv("SIGLOG_PATH", "signals_log.csv")

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
            raise RuntimeError(f"GET {r.url} {r.status}: {await r.text()}")
        return await r.json()

async def http_post_json(session: ClientSession, url: str, payload: Dict[str, Any]) -> Any:
    async with session.post(url, headers=alp_headers(), data=json.dumps(payload), timeout=ClientTimeout(total=30)) as r:
        if r.status >= 400:
            raise RuntimeError(f"POST {r.url} {r.status}: {await r.text()}")
        return await r.json()

# =========================
# Alpaca TRADE
# =========================
async def alp_account() -> Dict[str, Any]:
    async with ClientSession() as s:
        return await http_get_json(s, trade_url("account"))

async def place_notional_order(symbol: str, side: str, notional: float) -> Dict[str, Any]:
    payload = {
        "symbol": symbol,
        "side": side,                   # "buy" | "sell"
        "type": "market",
        "time_in_force": "gtc",
        "notional": str(float(notional)),
    }
    async with ClientSession() as s:
        return await http_post_json(s, trade_url("orders"), payload)

# =========================
# Alpaca DATA (crypto)
# =========================
async def fetch_active_crypto_symbols(limit: int = 1000) -> List[str]:
    """
    Тягнемо усі активні crypto-символи (US venue).
    API: GET /v1beta3/crypto/us/assets?status=active
    Повертає symbol у форматі 'BTC/USD'.
    """
    out: List[str] = []
    page_token: Optional[str] = None
    async with ClientSession() as s:
        while True:
            params = {"status": "active"}
            if page_token:
                params["page_token"] = page_token
            resp = await http_get_json(s, data_url("v1beta3/crypto/us/assets"), params=params)
            for a in resp.get("assets", []):
                sym = a.get("symbol") or ""
                if not sym:
                    continue
                out.append(sym)
                if len(out) >= limit:
                    return out
            page_token = resp.get("next_page_token")
            if not page_token:
                break
    return out

async def fetch_daily_change_pct(symbols: List[str]) -> Dict[str, float]:
    """
    Для кожного символу рахуємо зміну за останній день:
    pct = (last_close - prev_close) / prev_close * 100
    API: GET /v1beta3/crypto/us/bars?timeframe=1Day&symbols=..&limit=2
    """
    result: Dict[str, float] = {}
    if not symbols:
        return result

    # Альпака дозволяє запитувати кілька символів за раз — згрупуємо батчами
    BATCH = 50
    async with ClientSession() as s:
        for i in range(0, len(symbols), BATCH):
            chunk = symbols[i:i+BATCH]
            params = {
                "timeframe": "1Day",
                "symbols": ",".join(chunk),
                "limit": 2,
                "adjustment": "raw",
            }
            data = await http_get_json(s, data_url("v1beta3/crypto/us/bars"), params=params)
            bars_all = data.get("bars", {})
            for sym, bars in bars_all.items():
                if not bars or len(bars) < 2:
                    continue
                prev_c = float(bars[-2].get("c", 0) or 0)
                last_c = float(bars[-1].get("c", 0) or 0)
                if prev_c > 0:
                    pct = (last_c - prev_c) / prev_c * 100.0
                    result[sym] = pct
    return result

# =========================
# СКАНЕР — без заглушок
# =========================
async def scan_all(st: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    """
    Тягнемо ВСІ активні crypto-символи з Alpaca, рахуємо добову зміну і
    відбираємо ті, що зросли більше ніж на CRYPTO_MIN_D1_PCT (%).
    Повертаємо:
      report_text, picks_s (stocks поки порожній), picks_c (crypto-пари)
    """
    # 1) активні crypto
    symbols = await fetch_active_crypto_symbols(limit=ALPACA_MAX_CRYPTO)
    # 2) денної зміни
    pct_map = await fetch_daily_change_pct(symbols)

    picks_c: List[str] = []
    lines: List[str] = ["🛰 Сканер (crypto, D1 %+):"]

    # Відбір за порогом
    for sym, pct in sorted(pct_map.items(), key=lambda kv: kv[1], reverse=True):
        if pct >= CRYPTO_MIN_D1_PCT:
            picks_c.append(sym)
        # Для звіту покажемо топ-30, щоб не засмічувати чат
    for sym, pct in list(sorted(pct_map.items(), key=lambda kv: kv[1], reverse=True))[:30]:
        lines.append(f" • {sym}: {pct:+.2f}%")

    if not picks_c:
        lines.append("Немає монет, що перевищили поріг.")

    report = "\n".join(lines)
    picks_s: List[str] = []  # акціями займемось окремо, щоб не ускладнювати
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
        "Крипта тягнеться автоматично по всіх доступних парах."
    )
    await u.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN, reply_markup=main_keyboard())

async def aggressive_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    stedef(u.effective_chat.id)["mode"] = "aggressive"
    await u.message.reply_text("✅ Mode: AGGRESSIVE", reply_markup=main_keyboard())

async def scalp_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    stedef(u.effective_chat.id)["mode"] = "scalp"
    await u.message.reply_text("✅ Mode: SCALP", reply_markup=main_keyboard())

async def default_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    stedef(u.effective_chat.id)["mode"] = "default"
    await u.message.reply_text("✅ Mode: DEFAULT", reply_markup=main_keyboard())

async def swing_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    stedef(u.effective_chat.id)["mode"] = "swing"
    await u.message.reply_text("✅ Mode: SWING", reply_markup=main_keyboard())

async def safe_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    stedef(u.effective_chat.id)["mode"] = "safe"
    await u.message.reply_text("✅ Mode: SAFE", reply_markup=main_keyboard())

async def alp_on_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    stedef(u.effective_chat.id)["autotrade"] = True
    await u.message.reply_text("✅ Alpaca AUTOTRADE: ON", reply_markup=main_keyboard())

async def alp_off_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    stedef(u.effective_chat.id)["autotrade"] = False
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
        # обмежимося тільки крипто, щоб не дублити
        targets = picks_c[:ALPACA_MAX_CRYPTO]
        for sym in targets:
            try:
                await place_notional_order(sym, "buy", ALPACA_NOTIONAL)
                await u.message.reply_text(f"🟢 ORDER OK: {sym} ${ALPACA_NOTIONAL:.2f}")
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym}: {e}")

# =========================
# ФОНОВИЙ СКАНЕР (24/7)
# =========================
async def periodic_scan_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    for chat_id, st in list(STATE.items()):
        try:
            rep, picks_s, picks_c = await scan_all(st)
            st["last_scan_txt"] = rep
            if st.get("autotrade"):
                for sym in picks_c[:ALPACA_MAX_CRYPTO]:
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
        f"Поріг відбору crypto за добу: {CRYPTO_MIN_D1_PCT}%.\n"
        "Крипта 24/7, без перевірки торгової сесії.",
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

    # фоновий сканер
    app.job_queue.run_repeating(periodic_scan_job, interval=SCAN_EVERY_SEC, first=5)

    app.run_polling()

if __name__ == "__main__":
    main()
