# bot_alpaca.py
# -*- coding: utf-8 -*-

import os
import json
from typing import Any, Dict, List, Tuple

import aiohttp
from aiohttp import ClientSession, ClientTimeout

from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# =========================
# ENV
# =========================
TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN", "") or os.getenv("TELEGRAM_TOKEN", "")).strip()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "").strip()

# Торгові запити (orders/account): базовий трейд-ендпоінт (БЕЗ /v2 наприкінці)
ALPACA_BASE_URL = (os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/"))
# Дані ринку (снапшоти/бари): data API
ALPACA_DATA_URL = (os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/"))

# Налаштування
ALPACA_NOTIONAL = float(os.getenv("ALPACA_NOTIONAL", "25") or 25.0)   # $ на ордер
TOP_N = int(os.getenv("TOP_N", "5") or 5)                              # скільки найсильніших брати (3–5)
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
        ["/signals_crypto", "/signals_stocks"],
        ["/aggressive", "/scalp", "/default"],
        ["/swing", "/safe", "/help"],
        ["/alp_on", "/alp_status", "/alp_off"],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# =========================
# HTTP HELPERS
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

async def http_json(method: str, url: str, *, json_body: Any = None, headers: Dict[str, str] = None) -> Any:
    to = ClientTimeout(total=30)
    async with ClientSession(timeout=to) as s:
        async with s.request(method.upper(), url, headers=headers, data=(None if json_body is None else json.dumps(json_body))) as r:
            txt = await r.text()
            if r.status >= 400:
                raise RuntimeError(f"{method} {url} {r.status}: {txt}")
            if txt.strip():
                try:
                    return json.loads(txt)
                except Exception:
                    return txt
            return None

# =========================
# ALPACA TRADING
# =========================
async def alp_account() -> Dict[str, Any]:
    url = trade_url("account")
    return await http_json("GET", url, headers=alp_headers())

async def place_notional_order(symbol: str, side: str, notional: float) -> Dict[str, Any]:
    payload = {
        "symbol": symbol,
        "side": side,                   # "buy"/"sell"
        "type": "market",
        "time_in_force": "gtc",
        "notional": str(float(notional)),
    }
    url = trade_url("orders")
    return await http_json("POST", url, json_body=payload, headers=alp_headers())

# =========================
# DATA: КРИПТА
# =========================
async def crypto_active_usd_symbols() -> List[str]:
    """
    Беремо активні крипто-активи, фільтруємо тільки */USD (щоб не ловити 403 через баланс USDT/USDC),
    і повертаємо список символів як у торгівлі (AAVE/USD, BTC/USD, ...).
    """
    url = trade_url("assets?status=active&asset_class=crypto")
    assets = await http_json("GET", url, headers=alp_headers())
    syms = []
    for a in assets:
        sym = str(a.get("symbol", ""))
        if sym.endswith("/USD") and a.get("tradable", False):
            syms.append(sym)
    # трішки сортуємо для стабільності
    syms.sort()
    return syms

async def crypto_snapshots(symbols: List[str]) -> Dict[str, Any]:
    """
    Data API v1beta3: /crypto/us/snapshots?symbols=BTC/USD,ETH/USD,...
    Повертає dict з ключем 'snapshots' (per symbol).
    """
    if not symbols:
        return {"snapshots": {}}
    # робимо чанки по ~150, щоб не перевищувати ліміт довжини URL
    out = {"snapshots": {}}
    chunk = 150
    headers = alp_headers()
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i+chunk]
        sy = ",".join(part)
        url = data_url(f"v1beta3/crypto/us/snapshots?symbols={sy}")
        r = await http_json("GET", url, headers=headers)
        snaps = r.get("snapshots", {})
        out["snapshots"].update(snaps)
    return out

def strength_from_snapshot(s: Dict[str, Any]) -> float:
    """
    Оцінка "сили" інструмента:
      + daily change %
      + булеві бали за 1-хв свічку (close>open) та за відносно вузький спред
    """
    score = 0.0
    # daily change
    daily = s.get("dailyBar") or s.get("daily_bar") or {}
    d_o = float(daily.get("o", 0) or 0)
    d_c = float(daily.get("c", 0) or 0)
    if d_o > 0 and d_c > 0:
        score += (d_c - d_o) / d_o * 100.0  # %
    # 1-хв свічка
    m1 = s.get("minuteBar") or s.get("minute_bar") or {}
    m_o = float(m1.get("o", 0) or 0)
    m_c = float(m1.get("c", 0) or 0)
    if m_c > m_o > 0:
        score += 1.0
    # спред
    q = s.get("latestQuote") or s.get("latest_quote") or {}
    a = float(q.get("ap", 0) or 0)
    b = float(q.get("bp", 0) or 0)
    if a > 0 and b > 0:
        sp = (a - b) / a
        if sp < 0.002:  # <0.2% — бонус
            score += 0.5
    return score

async def scan_crypto_top() -> Tuple[str, List[str]]:
    """
    Повертає текстовий звіт і ТОП символів для торгівлі (до TOP_N).
    """
    all_usd = await crypto_active_usd_symbols()
    if not all_usd:
        return "❌ Крипта: немає активних USD-пар.", []

    snaps = await crypto_snapshots(all_usd)
    ss = snaps.get("snapshots", {})
    ranked: List[Tuple[str, float]] = []
    for sym in all_usd:
        s = ss.get(sym)
        if not s:
            continue
        ranked.append((sym, strength_from_snapshot(s)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    picks = [sym for sym, _ in ranked[:max(1, min(TOP_N, 5))]]

    top25 = ", ".join([sym for sym, _ in ranked[:25]]) or "—"
    rep = (
        "🛰️ Сканер (крипта):\n"
        f"• Активних USD-пар: {len(all_usd)}\n"
        f"• Використаємо для торгівлі (лімітом): {len(picks)}\n"
        f"• Перші 25: {top25}"
    )
    return rep, picks

# =========================
# DATA: АКЦІЇ (watchlist)
# =========================
STOCKS_WATCH = [
    "AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","GOOG","AVGO","ASML",
    "NFLX","AMD","COST","PEP","LLY","JPM","V","MA","JNJ","WMT",
]

async def stocks_snapshots(symbols: List[str]) -> Dict[str, Any]:
    """
    Data API v2: /v2/stocks/snapshots?symbols=AAPL,MSFT,...
    """
    if not symbols:
        return {"snapshots": {}}
    out = {"snapshots": {}}
    chunk = 150
    headers = alp_headers()
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i+chunk]
        sy = ",".join(part)
        url = data_url(f"v2/stocks/snapshots?symbols={sy}")
        r = await http_json("GET", url, headers=headers)
        snaps = r.get("snapshots", {})
        out["snapshots"].update(snaps)
    return out

def stock_strength_from_snapshot(s: Dict[str, Any]) -> float:
    # така ж формула, як для крипти
    return strength_from_snapshot(s)

async def scan_stocks_top() -> Tuple[str, List[str]]:
    snaps = await stocks_snapshots(STOCKS_WATCH)
    ss = snaps.get("snapshots", {})
    ranked: List[Tuple[str, float]] = []
    for sym in STOCKS_WATCH:
        s = ss.get(sym)
        if not s:
            continue
        ranked.append((sym, stock_strength_from_snapshot(s)))
    if not ranked:
        return "❌ Акції: немає даних по снапшотах.", []

    ranked.sort(key=lambda x: x[1], reverse=True)
    picks = [sym for sym, _ in ranked[:max(1, min(TOP_N, 5))]]
    top25 = ", ".join([sym for sym, _ in ranked[:25]]) or "—"
    rep = (
        "📈 Сканер (акції):\n"
        f"• У watchlist: {len(STOCKS_WATCH)}\n"
        f"• Використаємо для торгівлі (лімітом): {len(picks)}\n"
        f"• Перші 25: {top25}"
    )
    return rep, picks

# =========================
# КОМАНДИ БОТА
# =========================
async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    stedef(u.effective_chat.id)
    await u.message.reply_text(
        "👋 Бот готовий. Команди:\n"
        "• /signals_crypto — скан крипти (USD-пари)\n"
        "• /signals_stocks — скан акцій (watchlist)\n"
        "• /alp_on /alp_off — автотрейд\n"
        "• /alp_status — стан акаунту\n",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_keyboard()
    )

async def help_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    await start_cmd(u, c)

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
            f"• equity=${float(acc.get('equity',0)):,.2f}\n"
            f"Mode={stedef(u.effective_chat.id).get('mode')} · "
            f"Autotrade={'ON' if stedef(u.effective_chat.id).get('autotrade') else 'OFF'}"
        )
    except Exception as e:
        txt = f"❌ Alpaca error: {e}"
    await u.message.reply_text(txt, reply_markup=main_keyboard())

# ----- SIG/TRADE HELPERS
async def _report_and_trade(u: Update, rep: str, picks: List[str]) -> None:
    # звіт (розбиваємо якщо довгий)
    chunks = [rep[i:i+3500] for i in range(0, len(rep), 3500)] or [rep]
    for ch in chunks:
        await u.message.reply_text(ch)

    st = stedef(u.effective_chat.id)
    if not st.get("autotrade") or not picks:
        return

    # Торгуємо тільки BUY ринковими ордерами по ALPACA_NOTIONAL
    for sym in picks:
        try:
            await place_notional_order(sym, "buy", ALPACA_NOTIONAL)
            await u.message.reply_text(f"🟢 ORDER OK: {sym} ${ALPACA_NOTIONAL:.2f}")
        except Exception as e:
            await u.message.reply_text(f"🔴 ORDER FAIL {sym}: {e}")

# ----- /signals_crypto
async def signals_crypto(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        rep, picks = await scan_crypto_top()
        await _report_and_trade(u, rep, picks)
    except Exception as e:
        await u.message.reply_text(f"🔴 crypto scan error: {e}")

# ----- /signals_stocks
async def signals_stocks(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        rep, picks = await scan_stocks_top()
        await _report_and_trade(u, rep, picks)
    except Exception as e:
        await u.message.reply_text(f"🔴 stocks scan error: {e}")

# =========================
# MAIN
# =========================
def main():
    if not TG_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задано")

    app = Application.builder().token(TG_TOKEN).build()

    # Handlers
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

    app.run_polling()

if __name__ == "__main__":
    main()
