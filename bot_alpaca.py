# bot_alpaca.py
# -*- coding: utf-8 -*-

import os
import time
import json
from typing import Dict, Any, Tuple, List

from aiohttp import ClientSession, ClientTimeout

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# =========================
# ENV
# =========================
TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN", "") or os.getenv("TELEGRAM_TOKEN", "")).strip()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "").strip()

# Базові url (БЕЗ /v2 в кінці)
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
# v1beta3 crypto також іде через paper-api (на paper)
CRYPTO_EXCHANGE = "us"  # роут для v1beta3 крипти

ALPACA_NOTIONAL = float(os.getenv("ALPACA_NOTIONAL", "25") or 25.0)

# Налаштування сканера/відбору
MAX_PAIRS = int(os.getenv("MAX_PAIRS", "12") or 12)               # скільки показати в дайджесті
SYMBOL_BLACKLIST = {s.strip().upper() for s in (os.getenv("SYMBOL_BLACKLIST", "USDT,USDC").split(","))}
TIMEOUT = ClientTimeout(total=30)

# =========================
# СТАН НА ЧАТ
# =========================
def default_state() -> Dict[str, Any]:
    return {
        "autotrade": False,
        "mode": "default",
        "last_scan_txt": "",
        "last_posted_at": {},
        "pending_side": "buy",       # напрямок для підтвердження
        "pending_syms": [],          # підготовлені символи (ТОП-5)
        "pending_kind": "crypto",    # "crypto" або "stocks"
    }

STATE: Dict[int, Dict[str, Any]] = {}
def stedef(chat_id: int) -> Dict[str, Any]:
    return STATE.setdefault(chat_id, default_state())

# =========================
# КНОПКИ
# =========================
def main_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        ["/signals_crypto", "/signals_stocks"],   # окремо крипта / акції
        ["/aggressive", "/scalp", "/default"],
        ["/swing", "/safe", "/help"],
        ["/alp_on", "/alp_status", "/alp_off"],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def confirm_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Торгувати 3", callback_data="trade:3"),
         InlineKeyboardButton("✅ Торгувати 5", callback_data="trade:5")],
        [InlineKeyboardButton("❌ Скасувати", callback_data="trade:0")]
    ])

# =========================
# Alpaca helpers
# =========================
def alp_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }

def alp_url_v2(path: str) -> str:
    path = path.lstrip("/")
    return f"{ALPACA_BASE_URL}/v2/{path}"

async def alp_get_v2(session: ClientSession, path: str):
    async with session.get(alp_url_v2(path), headers=alp_headers(), timeout=TIMEOUT) as r:
        if r.status >= 400:
            raise RuntimeError(f"Alpaca GET {r.url} {r.status}: {await r.text()}")
        return await r.json()

async def alp_post_v2(session: ClientSession, path: str, payload: Dict[str, Any]):
    async with session.post(alp_url_v2(path), headers=alp_headers(),
                            data=json.dumps(payload), timeout=TIMEOUT) as r:
        if r.status >= 400:
            raise RuntimeError(f"Alpaca POST {r.url} {r.status}: {await r.text()}")
        return await r.json()

async def alp_account() -> Dict[str, Any]:
    async with ClientSession(timeout=TIMEOUT) as s:
        return await alp_get_v2(s, "account")

async def place_notional_order(symbol: str, side: str, notional: float) -> Dict[str, Any]:
    payload = {
        "symbol": symbol,
        "side": side,                   # "buy" | "sell"
        "type": "market",
        "time_in_force": "gtc",
        "notional": str(float(notional)),
    }
    async with ClientSession(timeout=TIMEOUT) as s:
        return await alp_post_v2(s, "orders", payload)

async def fetch_json_full(url: str) -> Any:
    async with ClientSession(timeout=TIMEOUT) as s:
        async with s.get(url, headers=alp_headers(), timeout=TIMEOUT) as r:
            if r.status >= 400:
                raise RuntimeError(f"GET {url} {r.status}: {await r.text()}")
            return await r.json()

# =========================
# СКАНЕРИ (ранжування)
# =========================
async def scan_crypto_top() -> List[Tuple[str, float]]:
    """
    Повертає список (symbol, strength), де strength — |% зміна за добу| з dailyBar snapshot’у.
    Беремо тільки USD-коти, ігноруємо стейбли з чорного списку.
    """
    out: List[Tuple[str, float]] = []

    # 1) активні крипто-асети
    assets_url = f"{ALPACA_BASE_URL}/v1beta3/crypto/{CRYPTO_EXCHANGE}/assets?status=active"
    assets = await fetch_json_full(assets_url)
    symbols: List[str] = []
    for a in assets.get("assets", []):
        sym = a.get("symbol", "").upper()
        if not sym or "/USD" not in sym:
            continue
        if any(b in sym for b in SYMBOL_BLACKLIST):
            continue
        symbols.append(sym)

    symbols = sorted(set(symbols))[:300]  # safety cap

    # 2) поштучні snapshots (простий і сумісний варіант)
    for sym in symbols:
        try:
            snaps_url = f"{ALPACA_BASE_URL}/v1beta3/crypto/{CRYPTO_EXCHANGE}/snapshots?symbols={sym}"
            js = await fetch_json_full(snaps_url)
            # ключ у відповіді — URL-encoded символ, але у більшості розгортань доступний і plain
            snap = (js.get("snapshots") or {}).get(sym) or next(iter((js.get("snapshots") or {}).values()), None)
            chg = 0.0
            if snap and snap.get("dailyBar"):
                o = float(snap["dailyBar"].get("o", 0) or 0)
                c = float(snap["dailyBar"].get("c", 0) or 0)
                if o > 0:
                    chg = abs((c - o) / o * 100.0)
            out.append((sym, chg))
        except Exception:
            out.append((sym, 0.0))

    out.sort(key=lambda x: x[1], reverse=True)
    return out

async def scan_stocks_top() -> List[Tuple[str, float]]:
    """
    Базове ранжування акцій: активні US-equity, tradable.
    strength = 2 якщо marginable & shortable, інакше 1. Сортуємо за strength.
    """
    out: List[Tuple[str, float]] = []
    async with ClientSession(timeout=TIMEOUT) as s:
        data = await alp_get_v2(s, "assets?status=active&asset_class=us_equity")
        for a in data:
            if not a.get("tradable"):
                continue
            sym = a.get("symbol", "").upper()
            strength = 2.0 if (a.get("marginable") and a.get("shortable")) else 1.0
            out.append((sym, strength))
    out.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return out

# =========================
# КОМАНДИ: режими/статус/он-оф
# =========================
async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    stedef(u.effective_chat.id)
    await u.message.reply_text(
        "👋 Готово. Вибери скан:\n"
        "• /signals_crypto — крипто (USD-коти)\n"
        "• /signals_stocks — акції US\n\n"
        "Після скану підтвердь кнопку «Торгувати 3/5».",
        reply_markup=main_keyboard(),
    )

async def help_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    await u.message.reply_text(
        "Команди:\n"
        "• /signals_crypto — скан крипти\n"
        "• /signals_stocks — скан акцій\n"
        "• /alp_on /alp_off — автотрейд прапорець (для майбутнього)\n"
        "• /alp_status — стан акаунту\n"
        "• /aggressive /scalp /default /swing /safe — режим профілю (інформ.)",
        reply_markup=main_keyboard(),
    )

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

# =========================
# СКАН + ПІДТВЕРДЖЕННЯ
# =========================
async def send_digest(u: Update, title: str, ranked: List[Tuple[str, float]]):
    preview = "\n".join([f"{i+1:>2}. {sym}  (★ {strength:.2f})"
                         for i, (sym, strength) in enumerate(ranked[:MAX_PAIRS])]) or "—"
    await u.message.reply_text(
        f"🛰 {title}\nТОП {min(MAX_PAIRS, len(ranked))}:\n{preview}\n\n"
        f"Натисни, щоб підтвердити угоди.",
        reply_markup=confirm_kb()
    )

async def signals_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stedef(u.effective_chat.id)
    ranked = await scan_crypto_top()
    st["pending_kind"] = "crypto"
    st["pending_syms"] = [sym for sym, _ in ranked[:5]]  # запас до 5
    st["pending_side"] = "buy"
    await send_digest(u, "Крипто (USD-коти): сила за добовою волатильністю", ranked)

async def signals_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stedef(u.effective_chat.id)
    ranked = await scan_stocks_top()
    st["pending_kind"] = "stocks"
    st["pending_syms"] = [sym for sym, _ in ranked[:5]]
    st["pending_side"] = "buy"
    await send_digest(u, "Акції US-equity: базовий рейтинг", ranked)

async def on_trade_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    st = stedef(q.message.chat_id)

    try:
        _, n_str = q.data.split(":")
        take = int(n_str)
    except Exception:
        take = 0

    if take <= 0:
        await q.edit_message_text("Скасовано.")
        st["pending_syms"] = []
        return

    syms = st.get("pending_syms", [])[:take]
    if not syms:
        await q.edit_message_text("Немає підготовлених символів.")
        return

    side = st.get("pending_side", "buy")
    ok = 0
    fail = 0
    for sym in syms:
        try:
            await place_notional_order(sym, side, ALPACA_NOTIONAL)
            ok += 1
            st["last_posted_at"][sym] = int(time.time())
            await context.bot.send_message(q.message.chat_id, f"🟢 ORDER OK: {sym} ${ALPACA_NOTIONAL:.2f}")
        except Exception as e:
            fail += 1
            await context.bot.send_message(q.message.chat_id, f"🔴 ORDER FAIL {sym}: {e}")

    await q.edit_message_text(f"Готово. Успішно: {ok}, помилок: {fail}.")
    st["pending_syms"] = []

# =========================
# MAIN
# =========================
def main():
    if not TG_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задано")

    app = Application.builder().token(TG_TOKEN).build()

    # базові команди
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

    # скани
    app.add_handler(CommandHandler("signals_crypto", signals_crypto))
    app.add_handler(CommandHandler("signals_stocks", signals_stocks))

    # підтвердження
    app.add_handler(CallbackQueryHandler(on_trade_confirm, pattern=r"^trade:\d+$"))

    # запуск (без asyncio.run/await)
    app.run_polling()

if __name__ == "__main__":
    main()
