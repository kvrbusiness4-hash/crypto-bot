# -*- coding: utf-8 -*-

import os
import math
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple

import aiohttp
from aiohttp import ClientSession, ClientTimeout

from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# ================== ENV ==================
TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_TOKEN") or "").strip()

ALPACA_API_KEY = (os.getenv("ALPACA_API_KEY") or "").strip()
ALPACA_API_SECRET = (os.getenv("ALPACA_API_SECRET") or "").strip()

# API base urls
ALPACA_BASE_URL = (os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_URL = (os.getenv("ALPACA_DATA_URL") or "https://data.alpaca.markets").rstrip("/")

# торг. суми за замовчуванням (можна змінювати ENV)
NOTIONAL_CRYPTO = float(os.getenv("ALPACA_NOTIONAL_CRYPTO") or 200)   # $/угоду
NOTIONAL_STOCKS = float(os.getenv("ALPACA_NOTIONAL_STOCKS") or 300)   # $/угоду

# максимальна к-сть одночасних позицій, що ми відкриємо сканером
ALPACA_MAX_CRYPTO = int(os.getenv("ALPACA_MAX_CRYPTO") or 3)
ALPACA_MAX_STOCKS = int(os.getenv("ALPACA_MAX_STOCKS") or 3)

# інтервал фонових сканів
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC") or 300)
# анти-дубль: після сигналу на символ — пауза (хв)
DEDUP_COOLDOWN_MIN = int(os.getenv("DEDUP_COOLDOWN_MIN") or 240)

# список що сканувати (за бажанням — через ENV)
SCAN_LIST_CRYPTO = [s.strip() for s in (os.getenv("SCAN_LIST_CRYPTO") or "AAVE/USD,AVAX/USD,BAT/USD").split(",") if s.strip()]
SCAN_LIST_STOCKS = [s.strip() for s in (os.getenv("SCAN_LIST_STOCKS") or "AAPL,AMAT,ADBE,AMD").split(",") if s.strip()]

# HTTP
TIMEOUT = ClientTimeout(total=30)
HTTP: Optional[ClientSession] = None

# ============ ПАРАМЕТРИ РЕЖИМІВ ============
MODE_PARAMS = {
    "scalp": {
        # малі швидкі тейки/стопи
        "tp_pct": 0.004,   # +0.4%
        "sl_pct": 0.005,   # -0.5%
        "bars": ("5Min", "15Min"),
    },
    "aggressive": {
        "tp_pct": 0.01,    # +1.0%
        "sl_pct": 0.01,    # -1.0%
        "bars": ("15Min", "30Min", "1Hour"),
    },
    "safe": {
        "tp_pct": 0.006,   # +0.6%
        "sl_pct": 0.006,   # -0.6%
        "bars": ("15Min", "1Hour"),
    },
}
DEFAULT_MODE = "scalp"

# ============ ГЛОБАЛЬНИЙ СТАН (по чату) ============
STATE: Dict[int, Dict[str, Any]] = {}  # chat_id -> {mode, autotrade, autoscan, side, last_signal_at, open_positions_seen}

# ================== УТИЛІТИ ==================


def _headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


async def alp_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    assert HTTP is not None
    url = f"{ALPACA_BASE_URL}{path}" if path.startswith("/v2") else f"{ALPACA_DATA_URL}{path}"
    async with HTTP.get(url, headers=_headers(), params=params) as r:
        if r.status >= 400:
            return {"_error": r.status, "_text": await r.text()}
        return await r.json()


async def alp_post(path: str, payload: Dict[str, Any]) -> Any:
    assert HTTP is not None
    url = f"{ALPACA_BASE_URL}{path}"
    async with HTTP.post(url, headers=_headers(), json=payload) as r:
        if r.status >= 400:
            return {"_error": r.status, "_text": await r.text()}
        return await r.json()


def pct(a: float) -> str:
    return f"{a * 100:.3f}%"


def fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def now_ts() -> float:
    return time.time()


def _state(chat_id: int) -> Dict[str, Any]:
    if chat_id not in STATE:
        STATE[chat_id] = {
            "mode": DEFAULT_MODE,
            "autotrade": False,
            "autoscan": False,
            "side": "long",
            "interval": SCAN_INTERVAL_SEC,
            "last_signal_at": {},  # symbol -> ts
            "open_positions_seen": set(),  # track closures
        }
    return STATE[chat_id]


# ================== АЛПАКА ХЕЛПЕРИ ==================

async def get_account() -> Dict[str, Any]:
    return await alp_get("/v2/account")


async def list_positions() -> List[Dict[str, Any]]:
    data = await alp_get("/v2/positions")
    if isinstance(data, dict) and data.get("_error"):
        return []
    return data


async def get_last_trade_price(symbol: str, is_crypto: bool) -> Optional[float]:
    """
    Для оцінки кількості по notional.
    Crypto: /v1beta3/crypto/us/trades/latest?symbol=BTC/USD
    Stocks: /v2/stocks/trades/latest?symbol=AAPL
    """
    if is_crypto:
        res = await alp_get("/v1beta3/crypto/us/trades/latest", params={"symbol": symbol})
        try:
            return float(res["trade"]["p"])
        except Exception:
            return None
    else:
        res = await alp_get("/v2/stocks/trades/latest", params={"symbol": symbol})
        try:
            return float(res["trade"]["p"])
        except Exception:
            return None


def is_crypto_symbol(symbol: str) -> bool:
    return "/" in symbol  # простий маркер: AAVE/USD, BTC/USD тощо


def round_qty(symbol: str, qty: float) -> float:
    # для crypto залишимо 6 знаків після крапки, для stocks — 4
    return float(f"{qty:.6f}") if is_crypto_symbol(symbol) else float(f"{qty:.4f}")


# ============== ВИСТАВЛЕННЯ ОРДЕРІВ =================

async def place_simple_buy_with_tp_sl(
    symbol: str,
    notional_usd: float,
    mode: str,
) -> Tuple[bool, str]:
    """
    1) Купуємо на 'notional' простим ринковим (fractional ок).
    2) Рахуємо TP/SL на основі MODE_PARAMS[mode].
    3) Ставимо ОКРЕМІ sell-ордера: limit (TP) і stop (SL). Без order_class — працює і для crypto, і для stocks.
    Повертає (ok, message)
    """
    is_crypto = is_crypto_symbol(symbol)
    params = MODE_PARAMS.get(mode, MODE_PARAMS[DEFAULT_MODE])
    tp_pct = float(params["tp_pct"])
    sl_pct = float(params["sl_pct"])

    # 1) купівля
    last = await get_last_trade_price(symbol, is_crypto)
    if not last or last <= 0:
        return False, f"{symbol}: не вдалось отримати ціну."

    qty = round_qty(symbol, notional_usd / last)

    buy_payload = {
        "symbol": symbol.replace("/", ""),  # для ALPACA orders crypto допускає формат AAVEUSD
        "side": "buy",
        "type": "market",
        "time_in_force": "gtc",
        "qty": f"{qty}",
    }

    # Для stocks fractionals можна через notional; для crypto qty — стабільніше.
    if not is_crypto:
        # краще віддати notional — Alpaca сама порахує фракцію
        buy_payload = {
            "symbol": symbol,
            "side": "buy",
            "type": "market",
            "time_in_force": "gtc",
            "notional": f"{notional_usd}",
        }

    res_buy = await alp_post("/v2/orders", buy_payload)
    if res_buy.get("_error"):
        return False, f"BUY FAIL {symbol}: {res_buy.get('_error')}: {res_buy.get('_text')}"

    entry_price = last  # приблизно, маркет може відрізнятись, але для повідомлення достатньо

    # 2) TP / SL ціни
    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 - sl_pct)

    # 3) окремі sell-ордера
    # qty для sell:
    sell_qty = qty if is_crypto else None  # для stocks створимо notional приблизно на всю суму

    # TP limit
    tp_payload = {
        "symbol": symbol.replace("/", "") if is_crypto else symbol,
        "side": "sell",
        "type": "limit",
        "time_in_force": "gtc",
        "limit_price": f"{tp_price:.6f}",
    }
    if is_crypto:
        tp_payload["qty"] = f"{sell_qty}"
    else:
        tp_payload["notional"] = f"{notional_usd * (1 + tp_pct)}"

    res_tp = await alp_post("/v2/orders", tp_payload)

    # SL stop
    sl_payload = {
        "symbol": symbol.replace("/", "") if is_crypto else symbol,
        "side": "sell",
        "type": "stop",
        "time_in_force": "gtc",
        "stop_price": f"{sl_price:.6f}",
    }
    if is_crypto:
        sl_payload["qty"] = f"{sell_qty}"
    else:
        sl_payload["notional"] = f"{notional_usd * (1 - sl_pct)}"

    res_sl = await alp_post("/v2/orders", sl_payload)

    msg = []
    if res_tp.get("_error"):
        msg.append(f"TP FAIL {res_tp['_error']}: {res_tp['_text']}")
    if res_sl.get("_error"):
        msg.append(f"SL FAIL {res_sl['_error']}: {res_sl['_text']}")

    info = (
        f"ORDER OK: {symbol} BUY {fmt_money(notional_usd)}\n"
        f"Вхід ≈ {entry_price:.6f}\n"
        f"TP: {tp_price:.6f} ({pct(tp_pct)}) • SL: {sl_price:.6f} ({pct(sl_pct)})"
    )
    if msg:
        info += "\n⚠️ " + " | ".join(msg)

    return True, info


# ============== СКАНЕРИ (дуже легкі-заглушки) =================

async def pick_crypto_symbols() -> List[str]:
    # Можеш замінити на свій реальний скан. Поки — список зі змінної/дефолту
    return SCAN_LIST_CRYPTO[:]


async def pick_stock_symbols() -> List[str]:
    return SCAN_LIST_STOCKS[:]


async def scan_and_maybe_trade(chat_id: int, app: Application) -> None:
    s = _state(chat_id)
    if not s["autotrade"]:
        return

    # актуальні відкриті позиції
    pos = await list_positions()
    open_symbols = set(p["symbol"] for p in pos)

    # обмежимо кількість для кожного класу
    open_crypto = [p for p in pos if p.get("asset_class") == "crypto"]
    open_stocks = [p for p in pos if p.get("asset_class") == "us_equity"]

    # виняток: не відкривати той самий символ, якщо недавно сигналили
    def _cool(symbol: str) -> bool:
        last_at = s["last_signal_at"].get(symbol, 0)
        return (now_ts() - last_at) >= DEDUP_COOLDOWN_MIN * 60

    # CRYPTO
    if len(open_crypto) < ALPACA_MAX_CRYPTO:
        for sym in await pick_crypto_symbols():
            sym_order = sym.replace("/", "")
            if sym_order in open_symbols:
                continue
            if not _cool(sym):
                continue
            ok, info = await place_simple_buy_with_tp_sl(sym, NOTIONAL_CRYPTO, s["mode"])
            s["last_signal_at"][sym] = now_ts()
            await app.bot.send_message(chat_id, info)
            if ok:
                break

    # STOCKS
    if len(open_stocks) < ALPACA_MAX_STOCKS:
        for sym in await pick_stock_symbols():
            if sym in open_symbols:
                continue
            if not _cool(sym):
                continue
            ok, info = await place_simple_buy_with_tp_sl(sym, NOTIONAL_STOCKS, s["mode"])
            s["last_signal_at"][sym] = now_ts()
            await app.bot.send_message(chat_id, info)
            if ok:
                break


# ======= Монітор закриття позицій (сповіщення) =======

async def notify_closures(chat_id: int, app: Application) -> None:
    s = _state(chat_id)
    prev = s["open_positions_seen"]
    pos = await list_positions()
    cur = set(p["symbol"] for p in pos)

    closed = prev - cur
    if closed:
        for sym in closed:
            await app.bot.send_message(chat_id, f"✅ Позицію *{sym}* закрито.", parse_mode=ParseMode.MARKDOWN)

    s["open_positions_seen"] = cur


# ================== КОМАНДИ TG ==================

def _keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            ["/scalp", "/aggressive", "/safe"],
            ["/alp_on", "/alp_off", "/alp_status"],
            ["/auto_on", "/auto_off", "/auto_status"],
            ["/signals_crypto", "/signals_stocks"],
        ],
        resize_keyboard=True,
    )


async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    _state(chat_id)  # ініціалізуємо
    await update.message.reply_text(
        "Крипта торгується 24/7; акції — коли ринок відкритий.\n"
        "Сканер/автотрейд може працювати у фоні.\n"
        "Увімкнути автотрейд: /alp_on · Зупинити: /alp_off · Стан: /alp_status\n"
        "Фоновий автоскан: /auto_on · /auto_off · /auto_status",
        reply_markup=_keyboard(),
    )


async def scalp_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    s = _state(chat_id)
    s["mode"] = "scalp"
    await update.message.reply_text("Режим встановлено: SCALP", reply_markup=_keyboard())


async def aggressive_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    s = _state(chat_id)
    s["mode"] = "aggressive"
    await update.message.reply_text("Режим встановлено: AGGRESSIVE", reply_markup=_keyboard())


async def safe_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    s = _state(chat_id)
    s["mode"] = "safe"
    await update.message.reply_text("Режим встановлено: SAFE", reply_markup=_keyboard())


async def alp_on_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    s = _state(chat_id)
    s["autotrade"] = True
    await update.message.reply_text("✅ Alpaca AUTOTRADE: ON", reply_markup=_keyboard())


async def alp_off_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    s = _state(chat_id)
    s["autotrade"] = False
    await update.message.reply_text("⛔ Alpaca AUTOTRADE: OFF", reply_markup=_keyboard())


async def auto_on_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    s = _state(chat_id)
    s["autoscan"] = True
    await update.message.reply_text(f"✅ AUTO-SCAN: ON (кожні {s['interval']}s)", reply_markup=_keyboard())


async def auto_off_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    s = _state(chat_id)
    s["autoscan"] = False
    await update.message.reply_text("⛔ AUTO-SCAN: OFF", reply_markup=_keyboard())


async def auto_status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    s = _state(chat_id)
    await update.message.reply_text(
        f"AutoScan={'ON' if s['autoscan'] else 'OFF'}; "
        f"Autotrade={'ON' if s['autotrade'] else 'OFF'}; "
        f"Mode={s['mode']} · Side={s['side']} · Interval={s['interval']}s",
        reply_markup=_keyboard(),
    )


async def alp_status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    s = _state(chat_id)

    acc = await get_account()
    cash = acc.get("cash")
    buying_power = acc.get("buying_power")
    equity = acc.get("equity")

    await update.message.reply_text(
        "📦 Alpaca:\n"
        f"• status={acc.get('status')}\n"
        f"• cash=${float(cash):,.2f}\n"
        f"• buying_power=${float(buying_power):,.2f}\n"
        f"• equity=${float(equity):,.2f}\n"
        f"Mode={s['mode']} · Autotrade={'ON' if s['autotrade'] else 'OFF'} · "
        f"AutoScan={'ON' if s['autoscan'] else 'OFF'} · Side={s['side']}",
        reply_markup=_keyboard(),
    )


async def signals_crypto_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    app = ctx.application
    s = _state(chat_id)

    # короткий звіт і спроба торгувати
    syms = await pick_crypto_symbols()
    await update.message.reply_text(
        "🛰️ Сканер (крипта):\n"
        f"• Активних USD-пар: {len(syms)}\n"
        f"• Використаємо для торгівлі (лімітом): {ALPACA_MAX_CRYPTO}\n"
        f"• Перші 25: {', '.join(syms[:25])}"
    )
    await scan_and_maybe_trade(chat_id, app)


async def signals_stocks_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    app = ctx.application
    s = _state(chat_id)

    syms = await pick_stock_symbols()
    await update.message.reply_text(
        "📡 Сканер (акції):\n"
        f"• Символів у списку: {len(syms)}\n"
        f"• Використаємо для торгівлі (лімітом): {ALPACA_MAX_STOCKS}\n"
        f"• Перші 25: {', '.join(syms[:25])}"
    )
    await scan_and_maybe_trade(chat_id, app)


# ============== ФОНОВІ ЗАДАЧІ ==============

async def scanner_loop(app: Application) -> None:
    """Періодично запускає скан і перевірку закриття позицій для всіх чатів із autoscan=True."""
    while True:
        try:
            # робимо копію ключів, бо STATE може змінюватися під час ітерації
            for chat_id in list(STATE.keys()):
                s = _state(chat_id)
                if s["autoscan"]:
                    await scan_and_maybe_trade(chat_id, app)
                # монітор закриття
                await notify_closures(chat_id, app)
        except Exception as e:
            print("scanner_loop error:", e)
        await asyncio.sleep(5)  # дрібний інтервал між чатами

        # глобальна пауза між циклами
        await asyncio.sleep( max(5, SCAN_INTERVAL_SEC // 5) )


# ===== запуск =====

async def main():
    global HTTP
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        HTTP = session

        app = Application.builder().token(TG_TOKEN).build()

        app.add_handler(CommandHandler("start", start_cmd))
        app.add_handler(CommandHandler("alp_status", alp_status_cmd))
        app.add_handler(CommandHandler("alp_on", alp_on_cmd))
        app.add_handler(CommandHandler("alp_off", alp_off_cmd))
        app.add_handler(CommandHandler("auto_on", auto_on_cmd))
        app.add_handler(CommandHandler("auto_off", auto_off_cmd))
        app.add_handler(CommandHandler("auto_status", auto_status_cmd))
        app.add_handler(CommandHandler("signals_crypto", signals_crypto_cmd))
        app.add_handler(CommandHandler("signals_stocks", signals_stocks_cmd))
        app.add_handler(CommandHandler("scalp", scalp_cmd))
        app.add_handler(CommandHandler("aggressive", aggressive_cmd))
        app.add_handler(CommandHandler("safe", safe_cmd))

        # запускаємо фон
        app.create_task(scanner_loop(app))

        print("Bot started.")
        await app.run_polling(close_loop=False)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
