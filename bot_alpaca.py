# bot_alpaca.py
# -*- coding: utf-8 -*-

import os
import json
from typing import Any, Dict, List, Tuple

from aiohttp import ClientSession, ClientTimeout
from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes
)

# =========================
# ENV
# =========================
TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or "").strip()

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "").strip()

# базові URL БЕЗ зайвих “/” в кінці
ALPACA_BASE_URL = (os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")).rstrip("/")
ALPACA_DATA_URL = (os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")).rstrip("/")

# налаштування торгівлі / скану
ALPACA_NOTIONAL        = float(os.getenv("ALPACA_NOTIONAL", "25") or 25)  # $ на один ордер
MAX_STRONG_ORDERS      = int(os.getenv("MAX_STRONG_ORDERS", "5") or 5)    # скільки кращих тикерів купувати
CRYPTO_MIN_MOVE_PCT    = float(os.getenv("CRYPTO_MIN_MOVE_PCT", "1.5") or 1.5)   # мін. добова зміна для крипти
EQUITY_MIN_MOVE_PCT    = float(os.getenv("EQUITY_MIN_MOVE_PCT", "2.0") or 2.0)   # мін. добова зміна для акцій
ENABLE_EQUITIES        = (os.getenv("ENABLE_EQUITIES", "1") != "0")
SCAN_EVERY_SEC         = int(os.getenv("SCAN_EVERY_SEC", "120") or 120)

TIMEOUT = ClientTimeout(total=30)

# =========================
# СТАН чату
# =========================
def default_state() -> Dict[str, Any]:
    return {"autotrade": False, "mode": "default", "last_scan_txt": ""}

STATE: Dict[int, Dict[str, Any]] = {}

def stedef(chat_id: int) -> Dict[str, Any]:
    return STATE.setdefault(chat_id, default_state())

# =========================
# КНОПКИ
# =========================
def main_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        ["/signals_crypto", "/signals_stocks" if ENABLE_EQUITIES else "/help"],
        ["/alp_on", "/alp_status", "/alp_off"],
        ["/aggressive", "/scalp", "/default", "/swing", "/safe"],
        ["/help"],
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

def t_url(base: str, path: str) -> str:
    return f"{base}/{path.lstrip('/')}"

async def fetch_json_full(url: str) -> Any:
    async with ClientSession(timeout=TIMEOUT) as s:
        async with s.get(url, headers=alp_headers()) as r:
            if r.status >= 400:
                txt = await r.text()
                raise RuntimeError(f"GET {url} {r.status}: {txt}")
            return await r.json()

async def alp_get_v2(session: ClientSession, path: str) -> Any:
    url = t_url(ALPACA_BASE_URL, f"v2/{path}")
    async with session.get(url, headers=alp_headers(), timeout=TIMEOUT) as r:
        if r.status >= 400:
            txt = await r.text()
            raise RuntimeError(f"Alpaca GET {url} {r.status}: {txt}")
        return await r.json()

async def alp_post_v2(path: str, payload: Dict[str, Any]) -> Any:
    url = t_url(ALPACA_BASE_URL, f"v2/{path}")
    async with ClientSession(timeout=TIMEOUT) as s:
        async with s.post(url, headers=alp_headers(), data=json.dumps(payload)) as r:
            if r.status >= 400:
                txt = await r.text()
                raise RuntimeError(f"Alpaca POST {url} {r.status}: {txt}")
            return await r.json()

# =========================
# Alpaca account / order
# =========================
async def alp_account() -> Dict[str, Any]:
    async with ClientSession(timeout=TIMEOUT) as s:
        return await alp_get_v2(s, "account")

async def place_notional_order(symbol: str, side: str, notional: float) -> Dict[str, Any]:
    payload = {
        "symbol": symbol,
        "side": side,            # "buy" / "sell"
        "type": "market",
        "time_in_force": "gtc",
        "notional": f"{float(notional)}",
    }
    return await alp_post_v2("orders", payload)

# =========================
# Сканери
# =========================
STABLES = ("USDT", "USDC")

async def scan_crypto_top() -> List[Tuple[str, float]]:
    """
    Беремо список крипто-асетів через v2/assets, а дані — через DATA API snapshots.
    Рахуємо абсолютну добову зміну у % (із dailyBar o->c).
    """
    out: List[Tuple[str, float]] = []
    async with ClientSession(timeout=TIMEOUT) as s:
        assets = await alp_get_v2(s, "assets?status=active&asset_class=crypto")
        symbols: List[str] = []
        for a in assets:
            sym = (a.get("symbol") or "").upper()
            # тримаємо тільки пари до USD (зручні для notional)
            if "/USD" not in sym:
                continue
            # ігноруємо чисті стейбли, типу USDT/USD, USDC/USD
            if any(st in sym.split("/")[0] for st in STABLES):
                continue
            symbols.append(sym)

    # підрізаємо щоб не тягти сотні
    symbols = sorted(set(symbols))[:150]
    if not symbols:
        return out

    # snapshots: DATA API
    snap_url = t_url(ALPACA_DATA_URL, f"v1beta3/crypto/us/snapshots?symbols={','.join(symbols)}")
    js = await fetch_json_full(snap_url)
    snaps = js.get("snapshots", {})

    for sym in symbols:
        snap = snaps.get(sym)
        chg = 0.0
        if snap and snap.get("dailyBar"):
            o = float(snap["dailyBar"].get("o") or 0)
            c = float(snap["dailyBar"].get("c") or 0)
            if o > 0:
                chg = abs((c - o) / o * 100.0)
        out.append((sym, chg))

    out.sort(key=lambda t: t[1], reverse=True)
    return out

async def scan_stocks_top() -> List[Tuple[str, float]]:
    """
    Акції: список через v2/assets?asset_class=us_equity, дані — DATA API v2/stocks/snapshots.
    """
    if not ENABLE_EQUITIES:
        return []

    out: List[Tuple[str, float]] = []
    async with ClientSession(timeout=TIMEOUT) as s:
        assets = await alp_get_v2(s, "assets?status=active&asset_class=us_equity&tradable=true")
        # відберемо “ліквідніші” (без OTC, з короткими тікерами)
        syms = [a.get("symbol", "").upper() for a in assets if a.get("symbol") and len(a["symbol"]) <= 5]
    syms = sorted(set(syms))[:150]
    if not syms:
        return out

    snap_url = t_url(ALPACA_DATA_URL, f"v2/stocks/snapshots?symbols={','.join(syms)}")
    js = await fetch_json_full(snap_url)
    snaps = js.get("snapshots", {})

    for sym, snap in snaps.items():
        chg = 0.0
        if snap and snap.get("dailyBar"):
            o = float(snap["dailyBar"].get("o") or 0)
            c = float(snap["dailyBar"].get("c") or 0)
            if o > 0:
                chg = abs((c - o) / o * 100.0)
        out.append((sym, chg))

    out.sort(key=lambda t: t[1], reverse=True)
    return out

# =========================
# Команди бота
# =========================
async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    stedef(u.effective_chat.id)
    await u.message.reply_text(
        "👋 Готово. /signals_crypto — крипта, /signals_stocks — акції.\n"
        "Autotrade: /alp_on /alp_off · Статус: /alp_status",
        reply_markup=main_keyboard(),
    )

async def help_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    await u.message.reply_text(
        "Команди:\n"
        "• /signals_crypto — скан крипти\n"
        "• /signals_stocks — скан акцій\n"
        "• /alp_on /alp_off /alp_status — автотрейд\n"
        "• /aggressive /scalp /default /swing /safe — режим (інформативно)\n"
        f"Фільтри: crypto ≥ {CRYPTO_MIN_MOVE_PCT}% | stocks ≥ {EQUITY_MIN_MOVE_PCT}%\n"
        f"Ордерів: до {MAX_STRONG_ORDERS} шт., по {ALPACA_NOTIONAL:.2f}$",
        reply_markup=main_keyboard(),
    )

async def set_mode(u: Update, _: ContextTypes.DEFAULT_TYPE, mode: str) -> None:
    st = stedef(u.effective_chat.id)
    st["mode"] = mode
    await u.message.reply_text(f"✅ Mode: {mode.upper()}", reply_markup=main_keyboard())

async def aggressive_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE): await set_mode(u, c, "aggressive")
async def scalp_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):      await set_mode(u, c, "scalp")
async def default_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):    await set_mode(u, c, "default")
async def swing_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):      await set_mode(u, c, "swing")
async def safe_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):       await set_mode(u, c, "safe")

async def alp_on_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(u.effective_chat.id); st["autotrade"] = True
    await u.message.reply_text("✅ Alpaca AUTOTRADE: ON", reply_markup=main_keyboard())

async def alp_off_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    st = stedef(u.effective_chat.id); st["autotrade"] = False
    await u.message.reply_text("⏹ Alpaca AUTOTRADE: OFF", reply_markup=main_keyboard())

async def alp_status_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        acc = await alp_account()
        txt = (
            "💼 Alpaca:\n"
            f"• status={acc.get('status','?')}\n"
            f"• cash=${float(acc.get('cash',0)):,.2f}\n"
            f"• buying_power=${float(acc.get('buying_power',0)):,.2f}\n"
            f"• equity=${float(acc.get('equity',0)):,.2f}\n"
            f"Mode={stedef(u.effective_chat.id).get('mode')} · "
            f"Autotrade={'ON' if stedef(u.effective_chat.id).get('autotrade') else 'OFF'}"
        )
    except Exception as e:
        txt = f"❌ Alpaca error: {e}"
    await u.message.reply_text(txt, reply_markup=main_keyboard())

async def _rank_and_trade(u: Update, c: ContextTypes.DEFAULT_TYPE, what: str) -> None:
    """
    what: 'crypto' або 'stocks'
    """
    st = stedef(u.effective_chat.id)

    try:
        if what == "crypto":
            ranked = await scan_crypto_top()
            min_move = CRYPTO_MIN_MOVE_PCT
        else:
            ranked = await scan_stocks_top()
            min_move = EQUITY_MIN_MOVE_PCT

        # фільтр “сильних”: мінімальна добова зміна
        strong = [(sym, chg) for sym, chg in ranked if chg >= min_move][:MAX_STRONG_ORDERS]

        if not strong:
            await u.message.reply_text(f"🛰 {what}: немає кандидатів (фільтр ≥ {min_move}%).")
            return

        # звіт
        report_lines = [
            f"🛰 Сканер {what.capitalize()}:",
            f"• Топ {len(strong)} за критерієм (≥ {min_move}%):",
            "• " + ", ".join([f"{s} ({chg:.2f}%)" for s, chg in strong]),
            f"• Ордер по ${ALPACA_NOTIONAL:.2f} кожен" if st.get("autotrade") else "• Режим перегляду (autotrade OFF)"
        ]
        await u.message.reply_text("\n".join(report_lines), parse_mode=ParseMode.MARKDOWN)

        # автотрейд
        if st.get("autotrade"):
            for sym, _ in strong:
                try:
                    await place_notional_order(sym, "buy", ALPACA_NOTIONAL)
                    await u.message.reply_text(f"🟢 ORDER OK: {sym} ${ALPACA_NOTIONAL:.2f}")
                except Exception as e:
                    await u.message.reply_text(f"🔴 ORDER FAIL {sym}: {e}")

    except Exception as e:
        await u.message.reply_text(f"🔴 {what} scan error: {e}")

async def signals_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    await _rank_and_trade(u, c, "crypto")

async def signals_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    if not ENABLE_EQUITIES:
        await u.message.reply_text("ℹ️ Акції вимкнено (ENABLE_EQUITIES=0).")
        return
    await _rank_and_trade(u, c, "stocks")

# =========================
# Periodic job (опційно)
# =========================
async def periodic_scan_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    for chat_id, st in list(STATE.items()):
        if not st.get("autotrade"):
            continue
        # автотрейд тільки по крипті (24/7); акції можна додати за потреби
        try:
            # “фейковий” Update для однакової логіки
            class U: effective_chat = type("C", (), {"id": chat_id})
            await _rank_and_trade(U(), ctx, "crypto")
        except Exception as e:
            try:
                await ctx.bot.send_message(chat_id, f"🔴 periodic_scan error: {e}")
            except Exception:
                pass

# =========================
# MAIN
# =========================
def main() -> None:
    if not TG_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задано")

    app = Application.builder().token(TG_TOKEN).build()

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

    app.add_handler(CommandHandler("signals_crypto", signals_crypto))
    app.add_handler(CommandHandler("signals_stocks", signals_stocks))

    # фоновий автоскладний скан (крипта) — кожні SCAN_EVERY_SEC
    app.job_queue.run_repeating(periodic_scan_job, interval=SCAN_EVERY_SEC, first=5)

    app.run_polling()

if __name__ == "__main__":
    main()
