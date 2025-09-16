# bot_alpaca.py
# -*- coding: utf-8 -*-
#
# Alpaca smart autotrade (crypto + stocks)
# - /signals_crypto: сканує ВСІ активні USD-криптопари на Alpaca, підтвердження 15m/30m/60m
# - /signals_stocks: сканує топ-ліквідні US-акції (список нижче), підтвердження 15m/30m/60m
# - /alp_on /alp_off /alp_status
# - Ордери: MARKET notional (USD). Крипта 24/7. Без перевірки торгових сесій.
#
# ENV (приклад):
# TELEGRAM_BOT_TOKEN=7282687464:AAHsIFXbRUCxYqJnm6iTUt5XLRlSUnthRtg
# ALPACA_API_KEY=PKLQDR19V7JS9Y15AXL3
# ALPACA_API_SECRET=GDcMGGXUC5uoPYb71mUfejHeZMZgmHm6D15rjXJG
# ALPACA_BASE_URL=https://paper-api.alpaca.markets
# ALPACA_DATA_URL=https://data.alpaca.markets
# ALPACA_NOTIONAL=50
# SCAN_EVERY_SEC=120
# ALPACA_MAX_STOCKS=400
# ALPACA_MAX_CRYPTO=200

import os
import json
from typing import Dict, Any, Tuple, List

import asyncio
import aiohttp
from aiohttp import ClientSession, ClientTimeout

from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# =========================
# ENV
# =========================
TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or "").strip()

ALPACA_API_KEY = (os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or "").strip()
ALPACA_API_SECRET = (os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY") or "").strip()

ALPACA_BASE_URL = (os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/"))
ALPACA_DATA_URL = (os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/"))

ALPACA_NOTIONAL = float(os.getenv("ALPACA_NOTIONAL", "50") or 50.0)
SCAN_EVERY_SEC = int(os.getenv("SCAN_EVERY_SEC", "120") or 120)

ALPACA_MAX_STOCKS = int(os.getenv("ALPACA_MAX_STOCKS", "400") or 400)   # скільки акцій максимум перевіряти за скан
ALPACA_MAX_CRYPTO = int(os.getenv("ALPACA_MAX_CRYPTO", "200") or 200)   # скільки криптопар максимум перевіряти

# =========================
# STATE
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
# UI
# =========================
def main_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        ["/aggressive", "/scalp", "/default"],
        ["/swing", "/safe", "/help"],
        ["/signals_crypto", "/signals_stocks"],
        ["/alp_on", "/alp_status", "/alp_off"],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# =========================
# Alpaca REST helpers
# =========================
def alp_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }

def alp_url(path: str) -> str:
    path = path.lstrip("/")
    return f"{ALPACA_BASE_URL}/v2/{path}"

async def alp_get(session: ClientSession, path: str) -> Any:
    async with session.get(alp_url(path), headers=alp_headers(), timeout=ClientTimeout(total=30)) as r:
        if r.status >= 400:
            raise RuntimeError(f"Alpaca GET {r.url} {r.status}: {await r.text()}")
        return await r.json()

async def alp_post(session: ClientSession, path: str, payload: Dict[str, Any]) -> Any:
    async with session.post(alp_url(path), headers=alp_headers(),
                            data=json.dumps(payload), timeout=ClientTimeout(total=30)) as r:
        if r.status >= 400:
            raise RuntimeError(f"Alpaca POST {r.url} {r.status}: {await r.text()}")
        return await r.json()

async def alp_account() -> Dict[str, Any]:
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        return await alp_get(s, "account")

# universal notional market order
async def place_notional_order(symbol: str, side: str, notional: float) -> Dict[str, Any]:
    payload = {
        "symbol": symbol,            # "AAPL" або "BTC/USD"
        "side": side,                # "buy" | "sell"
        "type": "market",
        "time_in_force": "gtc",
        "notional": str(float(notional)),
    }
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        return await alp_post(s, "orders", payload)

# =========================
# DATA fetchers (Alpaca Market Data)
# =========================
async def fetch_json_full(url: str) -> dict:
    async with ClientSession(timeout=ClientTimeout(total=40)) as s:
        async with s.get(url, headers=alp_headers()) as r:
            if r.status >= 400:
                raise RuntimeError(f"GET {url} {r.status}: {(await r.text())}")
            return await r.json()

# crypto bars: v1beta3 (symbols — без '/': BTCUSD)
async def get_bars_crypto(symbols: List[str], timeframe: str, limit: int=120) -> dict:
    # convert "BTC/USD" -> "BTCUSD"
    syms = ",".join([s.replace("/", "") for s in symbols])
    url = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/bars?symbols={syms}&timeframe={timeframe}&limit={limit}"
    return await fetch_json_full(url)

# stocks bars: v2
async def get_bars_stocks(symbols: List[str], timeframe: str, limit: int=200) -> dict:
    syms = ",".join(symbols)
    url = f"{ALPACA_DATA_URL}/v2/stocks/bars?symbols={syms}&timeframe={timeframe}&limit={limit}&adjustment=split"
    return await fetch_json_full(url)

# =========================
# Indicators
# =========================
def ema(values: List[float], period: int=20) -> List[float]:
    if not values:
        return []
    k = 2/(period+1)
    out = [values[0]]
    for v in values[1:]:
        out.append(out[-1] + k*(v - out[-1]))
    return out

def rsi(values: List[float], period: int=14) -> List[float]:
    if len(values) < period+1: return [50.0]*len(values)
    gains, losses = [], []
    for i in range(1, len(values)):
        d = values[i]-values[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_gain = sum(gains[:period])/period
    avg_loss = sum(losses[:period])/period
    out = [50.0]*period
    for i in range(period, len(values)-1):
        avg_gain = (avg_gain*(period-1)+gains[i])/period
        avg_loss = (avg_loss*(period-1)+losses[i])/period
        rs = (avg_gain/avg_loss) if avg_loss != 0 else 1e9
        out.append(100 - 100/(1+rs))
    out.append(out[-1] if out else 50.0)
    return out

# =========================
# Signal logic
# =========================
TIMEFRAMES = ["15Min", "30Min", "1Hour"]

def detect_signal_from_series(closes: List[float], highs: List[float]) -> bool:
    if len(closes) < 60:
        return False
    e20 = ema(closes, 20)[-1]
    e50 = ema(closes, 50)[-1]
    r = rsi(closes, 14)[-1]
    breakout = closes[-1] > max(highs[-20:])
    # LONG умови: тренд, "здоровий" RSI, пробій
    return (e20 > e50) and (45 <= r <= 70) and breakout

def aggregate_signals(bar_map: Dict[str, Dict[str, List[float]]]) -> bool:
    votes = 0
    for tf in TIMEFRAMES:
        series = bar_map.get(tf)
        if not series: 
            continue
        if detect_signal_from_series(series["c"], series["h"]):
            votes += 1
    return votes >= 2   # треба підтвердження 2 з 3

# =========================
# Scanners
# =========================
# curated liquid stocks (можеш розширити; великі списки — ризик лімітів)
STOCKS = [
    "AAPL","MSFT","NVDA","TSLA","META","AMZN","GOOGL","AMD","NFLX","AVGO",
    "JPM","BAC","XOM","CVX","PEP","KO","WMT","DIS","NKE","INTC",
    "ADBE","CRM","LIN","COST","V","MA","PYPL","MRNA","PFE","T"
][:ALPACA_MAX_STOCKS]

async def scan_crypto_confirmed(limit_signals: int=5) -> Tuple[str, List[str]]:
    # 1) усі активні crypto USD пари
    assets_url = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/assets?status=active"
    assets = await fetch_json_full(assets_url)
    usd_pairs = sorted({
        f"{a['symbol']}/USD" for a in (assets.get("assets") or [])
        if (a.get("class")=="crypto" and a.get("status")=="active" and a.get("symbol"))
    })
    if not usd_pairs:
        return "❌ Крипта: активних USD-пар не знайдено.", []
    # обмежимо верхньою межею на випадок дуже великих списків
    usd_pairs = usd_pairs[:ALPACA_MAX_CRYPTO]

    # 2) тягнемо бари по 3 TF
    bar_cache: Dict[str, Dict[str, Dict[str, List[float]]]] = {sym: {} for sym in usd_pairs}
    for tf in TIMEFRAMES:
        bars = await get_bars_crypto(usd_pairs, tf, limit=120)
        for raw_sym, arr in (bars.get("bars") or {}).items():
            # raw_sym: BTCUSD
            pretty = f"{raw_sym[:-3]}/{raw_sym[-3:]}" if raw_sym.endswith("USD") else raw_sym
            if arr:
                bar_cache[pretty][tf] = {
                    "t": [x["t"] for x in arr],
                    "c": [float(x["c"]) for x in arr],
                    "h": [float(x["h"]) for x in arr],
                }

    # 3) підтвердження 2/3 TF
    confirmed = []
    for sym, tfmap in bar_cache.items():
        if aggregate_signals(tfmap):
            c15 = tfmap.get("15Min", {}).get("c", [])
            score = (c15[-1]/c15[-2]-1) if len(c15) >= 2 else 0.0
            confirmed.append((sym, score))

    if not confirmed:
        return "🛰 Сканер (крипта): підтверджень немає.", []

    confirmed.sort(key=lambda x: x[1], reverse=True)
    picks = [s for s,_ in confirmed[:limit_signals]]

    txt = (
        "🛰 Сканер (крипта):\n"
        f"• Підтверджених інструментів: {len(confirmed)}\n"
        f"• До трейду: {len(picks)} → {', '.join(picks)}"
    )
    return txt, picks

async def scan_stocks_confirmed(limit_signals: int=5) -> Tuple[str, List[str]]:
    syms = STOCKS
    if not syms:
        return "❌ Акції: список порожній.", []
    syms = syms[:ALPACA_MAX_STOCKS]

    bar_cache: Dict[str, Dict[str, Dict[str, List[float]]]] = {sym: {} for sym in syms}
    for tf in TIMEFRAMES:
        bars = await get_bars_stocks(syms, tf, limit=200)
        for sym, arr in (bars.get("bars") or {}).items():
            if arr:
                bar_cache[sym][tf] = {
                    "t": [x["t"] for x in arr],
                    "c": [float(x["c"]) for x in arr],
                    "h": [float(x["h"]) for x in arr],
                }

    confirmed = []
    for sym, tfmap in bar_cache.items():
        if aggregate_signals(tfmap):
            c15 = tfmap.get("15Min", {}).get("c", [])
            score = (c15[-1]/c15[-2]-1) if len(c15) >= 2 else 0.0
            confirmed.append((sym, score))

    if not confirmed:
        return "🛰 Сканер (акції): підтверджень немає.", []

    confirmed.sort(key=lambda x: x[1], reverse=True)
    picks = [s for s,_ in confirmed[:limit_signals]]

    txt = (
        "🛰 Сканер (акції):\n"
        f"• Підтверджених інструментів: {len(confirmed)}\n"
        f"• До трейду: {len(picks)} → {', '.join(picks)}"
    )
    return txt, picks

# =========================
# Commands
# =========================
async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "👋 Бот готовий.\n"
        "• /signals_crypto — крипта (USD) з підтвердженням 15m/30m/60m\n"
        "• /signals_stocks — акції з підтвердженням 15m/30m/60m\n"
        "• /alp_on /alp_off /alp_status — автотрейд у Alpaca (USD notional)\n"
        "Крипта торгується 24/7; сесії не перевіряємо.",
        reply_markup=main_keyboard(),
        parse_mode=ParseMode.MARKDOWN
    )

async def help_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "Команди:\n"
        "• /signals_crypto — скан крипти (всі USD пари на Alpaca)\n"
        "• /signals_stocks — скан топ-акцій (список у коді)\n"
        "• /alp_on — увімкнути автотрейд (номінал USD з ALPACA_NOTIONAL)\n"
        "• /alp_off — вимкнути автотрейд\n"
        "• /alp_status — акаунт Alpaca\n"
        "• /aggressive /scalp /default /swing /safe — режими (інформативно)",
        reply_markup=main_keyboard()
    )

async def aggressive_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):  stedef(u.effective_chat.id)["mode"]="aggressive"; await u.message.reply_text("✅ Mode: AGGRESSIVE", reply_markup=main_keyboard())
async def scalp_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):       stedef(u.effective_chat.id)["mode"]="scalp";     await u.message.reply_text("✅ Mode: SCALP", reply_markup=main_keyboard())
async def default_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):     stedef(u.effective_chat.id)["mode"]="default";   await u.message.reply_text("✅ Mode: DEFAULT", reply_markup=main_keyboard())
async def swing_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):       stedef(u.effective_chat.id)["mode"]="swing";     await u.message.reply_text("✅ Mode: SWING", reply_markup=main_keyboard())
async def safe_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):        stedef(u.effective_chat.id)["mode"]="safe";      await u.message.reply_text("✅ Mode: SAFE", reply_markup=main_keyboard())

async def alp_on_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    stedef(u.effective_chat.id)["autotrade"] = True
    await u.message.reply_text("✅ Alpaca AUTOTRADE: ON", reply_markup=main_keyboard())

async def alp_off_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    stedef(u.effective_chat.id)["autotrade"] = False
    await u.message.reply_text("⏹ Alpaca AUTOTRADE: OFF", reply_markup=main_keyboard())

async def alp_status_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
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

async def signals_crypto_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stedef(u.effective_chat.id)
    rep, picks = await scan_crypto_confirmed(limit_signals=5)
    st["last_scan_txt"] = rep
    await u.message.reply_text(rep, parse_mode=ParseMode.MARKDOWN)
    if st.get("autotrade") and picks:
        for sym in picks:
            try:
                await place_notional_order(sym, "buy", ALPACA_NOTIONAL)
                await u.message.reply_text(f"🟢 ORDER OK: {sym} ${ALPACA_NOTIONAL:.2f}")
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym}: {e}")

async def signals_stocks_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = stedef(u.effective_chat.id)
    rep, picks = await scan_stocks_confirmed(limit_signals=5)
    st["last_scan_txt"] = rep
    await u.message.reply_text(rep, parse_mode=ParseMode.MARKDOWN)
    if st.get("autotrade") and picks:
        for sym in picks:
            try:
                await place_notional_order(sym, "buy", ALPACA_NOTIONAL)
                await u.message.reply_text(f"🟢 ORDER OK: {sym} ${ALPACA_NOTIONAL:.2f}")
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym}: {e}")

# =========================
# Periodic background job (optional)
# =========================
async def periodic_scan_job(ctx: ContextTypes.DEFAULT_TYPE):
    for chat_id, st in list(STATE.items()):
        try:
            rep_c, picks_c = await scan_crypto_confirmed(limit_signals=3)
            rep_s, picks_s = await scan_stocks_confirmed(limit_signals=2)
            txt = rep_c + "\n\n" + rep_s
            st["last_scan_txt"] = txt
            await ctx.bot.send_message(chat_id, txt)
            if st.get("autotrade"):
                for sym in (picks_c + picks_s):
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
# MAIN
# =========================
def main():
    if not TG_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задано")
    if not (ALPACA_API_KEY and ALPACA_API_SECRET):
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET не задано")

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

    app.add_handler(CommandHandler("signals_crypto", signals_crypto_cmd))
    app.add_handler(CommandHandler("signals_stocks", signals_stocks_cmd))

    # фон: крипта+акції (3+2) кожні SCAN_EVERY_SEC
    app.job_queue.run_repeating(periodic_scan_job, interval=SCAN_EVERY_SEC, first=5)

    app.run_polling()

if __name__ == "__main__":
    main()
