# bot_alpaca.py
import os
import asyncio
import aiohttp
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    KeyboardButton,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)

# =========================
# ENV
# =========================
TG_TOKEN         = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ALP_BASE_URL     = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").strip().rstrip("/")
ALP_DATA_URL     = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").strip().rstrip("/")
ALP_KEY          = os.getenv("ALPACA_API_KEY", "").strip()
ALP_SECRET       = os.getenv("ALPACA_API_SECRET", "").strip()

ALP_ENABLE       = bool(int(os.getenv("ALPACA_ENABLE", "1")))
ALP_NOTIONAL     = float(os.getenv("ALPACA_NOTIONAL", "50"))
MAX_STOCKS       = int(os.getenv("ALPACA_MAX_STOCKS", "150"))
MAX_CRYPTO       = int(os.getenv("ALPACA_MAX_CRYPTO", "80"))
SCAN_EVERY_SEC   = int(os.getenv("SCAN_EVERY_SEC", "120"))

# Data API endpoints
STOCK_BARS_URL   = f"{ALP_DATA_URL}/v2/stocks/bars"
# crypto бари – у Alpaca це v1beta3
CRYPTO_BARS_URL  = f"{ALP_DATA_URL}/v1beta3/crypto/us/bars"

# =========================
# Глобальний стан на чати
# =========================
def default_state():
    return {
        "autotrade": False,      # вмикається /alp_on
        "mode": "default",       # профіль
        "last_scan_txt": "",
        "last_picks": [],
        "last_picks_c": [],
        "limits": {
            "max_open": 20,      # ліміт позицій
        }
    }

STATE: Dict[int, Dict[str, Any]] = {}

# =========================
# Утиліти
# =========================
def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def fmt_usd(x: float) -> str:
    return f"${x:,.2f}"

async def http_json(session: aiohttp.ClientSession, method: str, url: str, **kw) -> Any:
    headers = kw.pop("headers", {})
    headers.update({
        "APCA-API-KEY-ID": ALP_KEY,
        "APCA-API-SECRET-KEY": ALP_SECRET,
        "Content-Type": "application/json"
    })
    async with session.request(method, url, headers=headers, **kw) as r:
        if r.status >= 400:
            txt = await r.text()
            raise RuntimeError(f"{method} {url} -> {r.status}: {txt}")
        if "application/json" in r.headers.get("Content-Type", ""):
            return await r.json()
        return await r.text()

async def alp_account(session: aiohttp.ClientSession) -> Dict[str, Any]:
    return await http_json(session, "GET", f"{ALP_BASE_URL}/v2/account")

async def alp_clock(session: aiohttp.ClientSession) -> Dict[str, Any]:
    # повертає {'is_open': bool, ...}
    return await http_json(session, "GET", f"{ALP_BASE_URL}/v2/clock")

async def alp_positions(session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    return await http_json(session, "GET", f"{ALP_BASE_URL}/v2/positions")

async def alp_orders_open(session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    return await http_json(session, "GET", f"{ALP_BASE_URL}/v2/orders?status=open&nested=true")

async def alp_submit_order(session: aiohttp.ClientSession, symbol: str, side: str, notional: float) -> Dict[str, Any]:
    payload = {
        "symbol": symbol,
        "side": side,               # "buy" / "sell"
        "type": "market",
        "time_in_force": "gtc",
        "notional": round(notional, 2)
    }
    return await http_json(session, "POST", f"{ALP_BASE_URL}/v2/orders", json=payload)

async def fetch_assets(session: aiohttp.ClientSession) -> Tuple[List[str], List[str]]:
    """Повертає (symbols_stocks, symbols_crypto) без вайтліста."""
    assets = await http_json(session, "GET", f"{ALP_BASE_URL}/v2/assets?status=active")
    stocks: List[str] = []
    crypto: List[str] = []
    for a in assets:
        if not a.get("tradable"): 
            continue
        klass = a.get("class") or a.get("asset_class")
        sym = a.get("symbol", "")
        if klass == "us_equity":
            stocks.append(sym)
        elif klass == "crypto":
            # торгуємо лише USD (у Alpaca: BTC/USD, ETH/USD, SOL/USD ...)
            if sym.endswith("/USD"):
                crypto.append(sym)
    # обмеження за ENV
    stocks = stocks[:MAX_STOCKS]
    crypto = crypto[:MAX_CRYPTO]
    return stocks, crypto

# =========================
# Індікатори
# =========================
def sma(values: List[float], window: int) -> float:
    if len(values) < window:
        return float("nan")
    return sum(values[-window:]) / window

def rsi(values: List[float], period: int = 14) -> float:
    if len(values) <= period:
        return float("nan")
    gains, losses = 0.0, 0.0
    for i in range(len(values)-period, len(values)-1):
        ch = values[i+1] - values[i]
        if ch > 0: gains += ch
        else: losses -= ch
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))

# =========================
# Дані (bars)
# =========================
async def get_stock_bars(session: aiohttp.ClientSession, symbols: List[str], tf="15Min", limit=100) -> Dict[str, List[float]]:
    """returns {symbol: [closes...]}"""
    out: Dict[str, List[float]] = {}
    if not symbols:
        return out

    # chunks до 50 символів
    step = 50
    for i in range(0, len(symbols), step):
        group = symbols[i:i+step]
        params = {
            "symbols": ",".join(group),
            "timeframe": tf,
            "limit": str(limit),
            "adjustment": "raw"
        }
        data = await http_json(session, "GET", STOCK_BARS_URL, params=params)
        bars = data.get("bars", {})
        for sym, arr in bars.items():
            closes = [b["c"] for b in arr]
            if closes:
                out[sym] = closes
    return out

async def get_crypto_bars(session: aiohttp.ClientSession, symbols: List[str], tf="15Min", limit=100) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    if not symbols:
        return out
    step = 30
    for i in range(0, len(symbols), step):
        group = symbols[i:i+step]
        params = {
            "symbols": ",".join(group),
            "timeframe": tf,
            "limit": str(limit)
        }
        data = await http_json(session, "GET", CRYPTO_BARS_URL, params=params)
        bars = data.get("bars", {})
        for sym, arr in bars.items():
            closes = [b["c"] for b in arr]
            if closes:
                out[sym] = closes
    return out

# =========================
# Скоринг
# =========================
def score_series(closes: List[float]) -> float:
    """Momentum + RSI + SMA alignment."""
    if len(closes) < 30:
        return -1e9
    mom = (closes[-1] / closes[0]) - 1.0   # за вікно даних
    r = rsi(closes, 14)
    s_fast = sma(closes, 10)
    s_slow = sma(closes, 30)
    trend = 0.0
    if not math.isnan(s_fast) and not math.isnan(s_slow):
        trend = (s_fast - s_slow) / s_slow
    # штраф за перекупленість
    overbought_penalty = 0.0
    if not math.isnan(r) and r > 75:
        overbought_penalty = -0.02
    # загальний бал
    return 0.7*mom + 0.3*trend + overbought_penalty

def pick_best(series: Dict[str, List[float]], top_k=5) -> List[Tuple[str, float]]:
    scored: List[Tuple[str, float]] = []
    for sym, closes in series.items():
        s = score_series(closes)
        if s != s or s < -1e8:
            continue
        scored.append((sym, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# =========================
# Торгова логіка
# =========================
async def place_orders_if_allowed(
    session: aiohttp.ClientSession,
    st: Dict[str, Any],
    picks: List[Tuple[str, float]],
    picks_c: List[Tuple[str, float]]
):
    if not st.get("autotrade"):
        return

    # ліміти / захист
    positions = await alp_positions(session)
    open_orders = await alp_orders_open(session)
    have = {p["symbol"] for p in positions}
    pending = {o["symbol"] for o in open_orders}
    total_open = len(have) + len(pending)
    max_open = st["limits"]["max_open"]

    to_trade: List[str] = []
    # беремо ТОП по акціях + ТОП по крипті
    for sym, _ in picks[:3]:
        to_trade.append(sym)
    for sym, _ in picks_c[:3]:
        to_trade.append(sym)

    placed = []
    for sym in to_trade:
        if total_open >= max_open:
            break
        if sym in have or sym in pending:
            continue
        try:
            resp = await alp_submit_order(session, sym, "buy", ALP_NOTIONAL)
            placed.append(sym)
            total_open += 1
        except Exception as e:
            # ігноруємо окремі фейли, рухаємось далі
            pass

    return placed

# =========================
# Скан
# =========================
async def scan_all(st: Dict[str, Any]) -> Tuple[str, List[Tuple[str,float]], List[Tuple[str,float]]]:
    txt_lines: List[str] = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as s:
        # статус акаунту
        acc = await alp_account(s)
        clk = await alp_clock(s)
        is_open = bool(clk.get("is_open"))

        txt_lines.append(f"📊 Account: status={acc.get('status')} · cash={fmt_usd(float(acc.get('cash',0)))} · buying_power={fmt_usd(float(acc.get('buying_power',0)))}")
        txt_lines.append(f"🕒 Market open: {'YES' if is_open else 'NO'} · {now_utc()}")

        # активи
        stocks, crypto = await fetch_assets(s)
        txt_lines.append(f"🔎 Assets: stocks={len(stocks)} · crypto={len(crypto)}")

        # тягнемо бари
        bars_s = await get_stock_bars(s, stocks, tf="15Min", limit=60) if stocks else {}
        bars_c = await get_crypto_bars(s, crypto, tf="15Min", limit=60) if crypto else {}

        # ранжуємо
        picks_s = pick_best(bars_s, top_k=10 if is_open else 0)  # акції лише якщо відкритий ринок
        picks_c = pick_best(bars_c, top_k=10)                    # крипта завжди

        # звіт
        if picks_s:
            txt_lines.append("🏛 Top stocks:")
            for sym, sc in picks_s[:5]:
                txt_lines.append(f" • {sym}: score={sc:.3f}")
        else:
            txt_lines.append("🏛 Top stocks: —")

        if picks_c:
            txt_lines.append("💠 Top crypto:")
            for sym, sc in picks_c[:5]:
                txt_lines.append(f" • {sym}: score={sc:.3f}")
        else:
            txt_lines.append("💠 Top crypto: —")

        # автотрейд
        placed = await place_orders_if_allowed(s, st, picks_s, picks_c)
        if placed:
            txt_lines.append("🟢 Orders placed: " + ", ".join(placed))
        elif st.get("autotrade"):
            txt_lines.append("ℹ️ Autotrade ON, але підходящих символів для нових ордерів не знайдено (або ліміти).")

    rep = "\n".join(txt_lines)
    return rep, picks_s, picks_c

# =========================
# Telegram UI
# =========================
MAIN_KB = ReplyKeyboardMarkup(
    [
        [KeyboardButton("/alp_status"), KeyboardButton("/signals")],
        [KeyboardButton("/alp_on"), KeyboardButton("/alp_off")],
        [KeyboardButton("/aggressive"), KeyboardButton("/default"), KeyboardButton("/safe")],
        [KeyboardButton("/help")]
    ],
    resize_keyboard=True
)

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    await u.message.reply_text(
        "👋 Готово. Бот сканує всі доступні активи в Alpaca.\n"
        "• /alp_on — увімкнути автотрейд\n"
        "• /alp_off — вимкнути автотрейд\n"
        "• /alp_status — стан акаунту\n"
        "• /signals — ручний скан + (за потреби) автотрейд\n",
        reply_markup=MAIN_KB
    )

async def help_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "Команди:\n"
        "/alp_status — акаунт + стан ринку\n"
        "/signals — скан і (якщо увімкнено) автотрейд\n"
        "/alp_on | /alp_off — увімк/вимк автотрейд\n"
        "/aggressive | /default | /safe — профіль ризику\n",
        reply_markup=MAIN_KB
    )

async def profile_aggr(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    st["mode"] = "aggressive"
    st["limits"]["max_open"] = 40
    await u.message.reply_text("✅ Mode: AGGRESSIVE", reply_markup=MAIN_KB)

async def profile_default(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    st["mode"] = "default"
    st["limits"]["max_open"] = 25
    await u.message.reply_text("✅ Mode: DEFAULT", reply_markup=MAIN_KB)

async def profile_safe(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    st["mode"] = "safe"
    st["limits"]["max_open"] = 10
    await u.message.reply_text("✅ Mode: SAFE", reply_markup=MAIN_KB)

async def autotrade_on_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    st["autotrade"] = True if ALP_ENABLE else False
    await u.message.reply_text("✅ Alpaca AUTOTRADE: ON", reply_markup=MAIN_KB)

async def autotrade_off_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    st["autotrade"] = False
    await u.message.reply_text("🟨 Alpaca AUTOTRADE: OFF", reply_markup=MAIN_KB)

async def status_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as s:
            acc = await alp_account(s)
            clk = await alp_clock(s)
        txt = (
            f"💼 Alpaca: status={acc.get('status','?')}\n"
            f"• cash={fmt_usd(float(acc.get('cash',0)))} · buying_power={fmt_usd(float(acc.get('buying_power',0)))} · "
            f"equity={fmt_usd(float(acc.get('equity',0)))}\n"
            f"🕒 Market open: {'YES' if clk.get('is_open') else 'NO'}\n"
            f"{now_utc()}"
        )
    except Exception as e:
        txt = f"❌ Alpaca error: {e}"
    await u.message.reply_text(txt, reply_markup=MAIN_KB)

async def signals_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = STATE.setdefault(u.effective_chat.id, default_state())
    try:
        rep, picks_s, picks_c = await scan_all(st)
        st["last_scan_txt"] = rep
        st["last_picks"] = picks_s
        st["last_picks_c"] = picks_c
        # надсилаємо частинами
        chunks = [rep[i:i+3500] for i in range(0, len(rep), 3500)]
        for ch in chunks:
            await u.message.reply_text(ch, parse_mode=ParseMode.MARKDOWN, reply_markup=MAIN_KB)
    except Exception as e:
        await u.message.reply_text(f"❌ Scan error: {e}", reply_markup=MAIN_KB)

# =========================
# Періодичний автоскан
# =========================
async def periodic_scan(app: Application):
    while True:
        try:
            # робимо копію ключів, щоб не ламати ітерацію під час оновлень
            for chat_id in list(STATE.keys()):
                st = STATE.get(chat_id)
                if not st:
                    continue
                try:
                    rep, picks_s, picks_c = await scan_all(st)
                    STATE[chat_id]["last_scan_txt"] = rep
                    STATE[chat_id]["last_picks"] = picks_s
                    STATE[chat_id]["last_picks_c"] = picks_c
                    # якщо автотрейд увімкнений – тихо працюємо, звіт не шлемо кожного разу
                except Exception as e:
                    # не падаємо на циклі – йдемо далі
                    pass
        except Exception:
            pass
        await asyncio.sleep(SCAN_EVERY_SEC)

# =========================
# main
# =========================
def build_app() -> Application:
    app = Application.builder().token(TG_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("alp_status", status_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))
    app.add_handler(CommandHandler("alp_on", autotrade_on_cmd))
    app.add_handler(CommandHandler("alp_off", autotrade_off_cmd))

    app.add_handler(CommandHandler("aggressive", profile_aggr))
    app.add_handler(CommandHandler("default", profile_default))
    app.add_handler(CommandHandler("safe", profile_safe))

    # приховуємо текстові повідомлення в никуда
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), help_cmd))
    return app

if __name__ == "__main__":
    if not TG_TOKEN or not ALP_KEY or not ALP_SECRET:
        raise SystemExit("Missing ENV: TELEGRAM_BOT_TOKEN / ALPACA_API_KEY / ALPACA_API_SECRET")

    application = build_app()
    # запускаємо фонового сканера
    application.job_queue.run_repeating(lambda ctx: None, interval=1)  # ініціалізація JobQueue
    application.create_task(periodic_scan(application))
    application.run_polling(allowed_updates=Update.ALL_TYPES, close_loop=False)
