import os
import json
import time
import hmac
import math
import hashlib
import logging
import threading
from datetime import datetime, timedelta, timezone

import requests
from apscheduler.schedulers.background import BackgroundScheduler

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes, CallbackContext
)
from telegram.error import BadRequest

# ---------------------------
# ЛОГИ
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("bot")

# ---------------------------
# ENV
# ---------------------------
TG_TOKEN         = os.getenv("TELEGRAM_BOT_TOKEN", "")
ADMIN_ID         = int(os.getenv("ADMIN_ID", "0"))

# Оплата (залишаємо як було — не чіпаємо логіку)
WALLET_ADDRESS   = os.getenv("WALLET_ADDRESS", "")
TRON_API_KEY     = os.getenv("TRON_API_KEY", "")

# Bybit (UTA)
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_BASE       = os.getenv("BYBIT_BASE", "https://api.bybit.com")

# Торгові налаштування за замовчуванням (можна змінювати командами)
DEFAULT_SL_PCT   = float(os.getenv("SL_PCT", "3"))
DEFAULT_TP_PCT   = float(os.getenv("TP_PCT", "5"))
DEFAULT_SIZE     = float(os.getenv("SIZE_USDT", "4"))
DEFAULT_LEV      = int(os.getenv("LEVERAGE", "3"))
TRADE_ENABLED_ENV= os.getenv("TRADE_ENABLED", "1").strip()  # "1" або "0"
DEFAULT_INTERVAL = int(os.getenv("HEARTBEAT_MIN", "15"))  # хвилини автоскану
STRONG_VOTE      = int(os.getenv("STRONG_VOTE", "2"))     # скільки голосів

# Whitelist (коми), або спеціальний режим "TOP30"
TRADE_WHITELIST  = os.getenv("TRADE_WHITELIST", "TOP30").strip()

# ---------------------------
# СТАН
# ---------------------------
class State:
    def __init__(self):
        self.sl_pct        = DEFAULT_SL_PCT
        self.tp_pct        = DEFAULT_TP_PCT
        self.size_usdt     = DEFAULT_SIZE
        self.leverage      = DEFAULT_LEV
        self.auto_on       = False
        self.auto_minutes  = DEFAULT_INTERVAL
        self.trade_enabled = (TRADE_ENABLED_ENV == "1")
        self.filter_mode   = "TOP30" if TRADE_WHITELIST.upper() == "TOP30" else "WHITELIST"
        self.whitelist     = [] if self.filter_mode == "TOP30" else self._parse_whitelist(TRADE_WHITELIST)
        self.last_signals  = []
        self.lock          = threading.Lock()

    @staticmethod
    def _parse_whitelist(s):
        tickers = []
        for t in s.split(","):
            t = t.strip().upper()
            if not t:
                continue
            if not t.endswith("USDT"):
                t = t + "USDT"
            tickers.append(t)
        return tickers

state = State()

# ---------------------------
# БЕЗПЕЧНІ ВІДПРАВКИ БЕЗ MARKDOWN
# ---------------------------
def say(update: Update, text: str):
    try:
        return update.message.reply_text(text, disable_web_page_preview=True)
    except BadRequest:
        return update.message.reply_text(str(text), disable_web_page_preview=True)

def bot_say(context: CallbackContext, chat_id: int, text: str):
    try:
        return context.bot.send_message(chat_id=chat_id, text=text, disable_web_page_preview=True)
    except BadRequest:
        return context.bot.send_message(chat_id=chat_id, text=str(text), disable_web_page_preview=True)

# ---------------------------
# COINGECKO (топ-30 + тех.оцінка)
# ---------------------------
CG_BASE = "https://api.coingecko.com/api/v3"

def cg_get_markets_top30():
    """
    Повертає список словників по топ-капіталізації (до 30)
    """
    url = f"{CG_BASE}/coins/markets"
    params = dict(
        vs_currency="usd",
        order="market_cap_desc",
        per_page=30,
        page=1,
        sparkline=False,
        price_change_percentage="24h"
    )
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def rsi_vector(prices, period):
    # дуже приблизний RSI (для демонстрації)
    if len(prices) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(1, period+1):
        diff = prices[-i] - prices[-i-1]
        if diff >= 0:
            gains.append(diff)
        else:
            losses.append(-diff)
    avg_gain = sum(gains)/period if gains else 0.000001
    avg_loss = sum(losses)/period if losses else 0.000001
    rs = avg_gain/avg_loss
    rsi = 100 - (100/(1+rs))
    return max(0, min(100, rsi))

def tech_vote(item):
    """
    Дуже простий скоринговий підхід:
    - береться % зміни за 24h, орієнтовний RSI (імітуємо даними з current_price),
      напрямок тренду.
    Повертає:
      direction: "LONG" або "SHORT"
      score: float
      indicators: dict
    """
    # Синтетичні дані для RSI (бо без платних API історію не тягнемо)
    # Беремо current_price і «імітуємо» три таймфрейми від 24h change
    price = float(item["current_price"])
    ch24  = float(item.get("price_change_percentage_24h", 0.0) or 0.0)
    # Умовний напрям: якщо 24h сильно червоний — схиляємось до SHORT
    direction = "SHORT" if ch24 < -2.0 else ("LONG" if ch24 > 2.0 else "FLAT")

    # Імітуємо RSI як функцію від зміни за 24h
    rsi15 = max(10.0, min(90.0, 50 + ch24))
    rsi30 = max(10.0, min(90.0, 50 + ch24/2))
    rsi60 = max(10.0, min(90.0, 50 + ch24/3))

    # MACD/Trend — бінарні прапорці від ch24
    macd_up = 1 if ch24 > 0 else 0
    trend_up = 1 if ch24 > 0 else 0

    # підрахуємо примітивний score
    score = (abs(ch24)/10) + (macd_up + trend_up)*0.3 + ( (70-rsi60) if direction=="SHORT" else (rsi60-30) )/100

    return {
        "direction": direction if direction != "FLAT" else ("LONG" if rsi60>55 else "SHORT"),
        "score": round(score, 2),
        "ind": {
            "RSI15": round(rsi15, 1),
            "RSI30": round(rsi30, 1),
            "RSI60": round(rsi60, 1),
            "MACD": "UP" if macd_up else "DOWN",
            "Trend": "UP" if trend_up else "DOWN",
        },
        "price": price,
        "ch24": round(ch24, 2),
    }

def compose_strong_signals(max_take=2):
    """
    Повертає список сигналів (до 2) зі списку (TOP30 або whitelist).
    Елемент: dict(symbol, dir, price, sl_px, tp_px, score, ind)
    """
    data = cg_get_markets_top30()  # список по CG
    candidates = []
    for it in data:
        symbol_cg = it["symbol"].upper()  # e.g. "BTC"
        symbol = f"{symbol_cg}USDT"
        if state.filter_mode == "WHITELIST" and symbol not in state.whitelist:
            continue

        t = tech_vote(it)
        direction = t["direction"]
        price     = t["price"]
        sl_pct    = state.sl_pct/100.0
        tp_pct    = state.tp_pct/100.0

        # розрахунок SL/TP
        if direction == "LONG":
            sl_px = price * (1 - sl_pct)
            tp_px = price * (1 + tp_pct)
        else:
            sl_px = price * (1 + sl_pct)
            tp_px = price * (1 - tp_pct)

        candidates.append({
            "symbol": symbol,
            "dir": direction,
            "price": round(price, 6),
            "sl_px": round(sl_px, 6),
            "tp_px": round(tp_px, 6),
            "score": t["score"],
            "ind": t["ind"],
            "ch24": t["ch24"],
        })

    # сортуємо за score (спадання)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    strong = candidates[:max_take]
    state.last_signals = strong
    return strong

# ---------------------------
# BYBIT (Unified v5) — підпис і запит
# ---------------------------
def bybit_signature(ts, method, path, query, body):
    param_str = "" if not query else "&".join([f"{k}={v}" for k,v in sorted(query.items())])
    body_str  = "" if not body else (json.dumps(body, separators=(',', ':')) if isinstance(body, (dict, list)) else str(body))
    sign_str  = str(ts) + BYBIT_API_KEY + "5000" + param_str + body_str  # recv_window=5000
    return hmac.new(BYBIT_API_SECRET.encode(), sign_str.encode(), hashlib.sha256).hexdigest()

def bybit_request(method, endpoint, query=None, body=None):
    url = BYBIT_BASE + endpoint
    ts  = int(time.time() * 1000)
    sign= bybit_signature(ts, method, endpoint, query or {}, body)

    headers = {
        "X-BAPI-API-KEY": BYBIT_API_KEY,
        "X-BAPI-SIGN": sign,
        "X-BAPI-TIMESTAMP": str(ts),
        "X-BAPI-RECV-WINDOW": "5000",
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*"
    }
    if method == "GET":
        r = requests.get(url, params=query, headers=headers, timeout=20)
    else:
        r = requests.post(url, params=query, json=body, headers=headers, timeout=20)

    # повертаємо як текст + json якщо можна
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}

def bybit_set_leverage(symbol, leverage):
    # POST /v5/position/set-leverage
    body = {
        "category": "linear",
        "symbol": symbol,
        "buyLeverage": str(leverage),
        "sellLeverage": str(leverage)
    }
    code, data = bybit_request("POST", "/v5/position/set-leverage", body=body)
    return code, data

def bybit_place_order(symbol, side, qty, sl_px=None, tp_px=None):
    # POST /v5/order/create (linear)
    body = {
        "category": "linear",
        "symbol": symbol,
        "side": side,             # "Buy" або "Sell"
        "orderType": "Market",
        "qty": str(qty),
        "timeInForce": "GoodTillCancel",
    }
    if tp_px is not None:
        body["takeProfit"] = str(tp_px)
        body["tpTriggerBy"] = "LastPrice"
    if sl_px is not None:
        body["stopLoss"] = str(sl_px)
        body["slTriggerBy"] = "LastPrice"

    code, data = bybit_request("POST", "/v5/order/create", body=body)
    return code, data

def bybit_calc_qty(symbol, price, size_usdt, lev):
    # орієнтовна кількість контрактів (за USDT)
    notional = size_usdt * lev
    qty = notional / price
    # округлимо до 3 знаків (на практиці треба брати step з /v5/market/instruments-info)
    return max(0.001, round(qty, 3))

# ---------------------------
# КОМАНДИ
# ---------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = [
        "👋 Готовий!",
        "Команди:",
        "/signals — сканувати зараз (+автотрейд, якщо включений)",
        "/auto_on 15 — автопуш кожні N хв (5–120)",
        "/auto_off — вимкнути автопуш",
        "/trade_on — увімкнути автотрейд",
        "/trade_off — вимкнути автотрейд",
        "/set_size 5 — розмір угоди (USDT)",
        "/set_lev 3 — плече",
        "/set_risk 3 5 — SL% TP%",
        "/status — стан",
    ]
    say(update, "\n".join(lines))

def status_text():
    lines = [
        f"Статус: {'ON' if state.auto_on else 'OFF'} · кожні {state.auto_minutes} хв.",
        f"SL={state.sl_pct:.2f}% · TP={state.tp_pct:.2f}%",
        f"TRADE_ENABLED={'ON' if state.trade_enabled else 'OFF'} · SIZE={state.size_usdt:.2f} USDT · LEV={state.leverage}",
        f"Фільтр: {'TOP30' if state.filter_mode=='TOP30' else 'WHITELIST'}",
        "UTC: " + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"),
    ]
    return "\n".join(lines)

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    say(update, status_text())

async def set_size_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        v = float(context.args[0])
        if v <= 0:
            raise ValueError
        state.size_usdt = v
        say(update, f"✅ Розмір угоди встановлено: {v:.2f} USDT")
    except Exception:
        say(update, "⚠️ Приклад: /set_size 5")

async def set_lev_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        v = int(context.args[0])
        if v < 1 or v > 20:
            raise ValueError
        state.leverage = v
        say(update, f"✅ Плече встановлено: x{v}")
    except Exception:
        say(update, "⚠️ Приклад: /set_lev 3 (1–20)")

async def set_risk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        sl = float(context.args[0])
        tp = float(context.args[1])
        if sl <= 0 or tp <= 0:
            raise ValueError
        state.sl_pct = sl
        state.tp_pct = tp
        say(update, f"✅ Ризик встановлено: SL={sl:.2f}% · TP={tp:.2f}%")
    except Exception:
        say(update, "⚠️ Приклад: /set_risk 3 5")

async def trade_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.trade_enabled = True
    say(update, "🤖 Автоторгівля УВІМКНЕНА")

async def trade_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.trade_enabled = False
    say(update, "⏸️ Автоторгівлю ВИМКНЕНО")

async def auto_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        n = int(context.args[0]) if context.args else state.auto_minutes
        n = max(5, min(120, n))
        state.auto_minutes = n
        state.auto_on = True
        scheduler_reschedule(n)
        say(update, f"Автоскан увімкнено: кожні {n} хв.")
        # одразу пробний прогін
        await run_scan(update, context, push_header=False)
    except Exception:
        say(update, "⚠️ Приклад: /auto_on 15")

async def auto_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.auto_on = False
    scheduler.pause()
    say(update, "Автоскан вимкнено")

async def signals_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await run_scan(update, context, push_header=True)

# ---------------------------
# СКАН + ТОРГІВЛЯ
# ---------------------------
def format_signal(sig):
    sym = sig["symbol"]
    dr  = sig["dir"]
    p   = sig["price"]
    sl  = sig["sl_px"]
    tp  = sig["tp_px"]
    sc  = sig["score"]
    ind = sig["ind"]
    ch  = sig["ch24"]
    return (
        f"• {sym}: {dr} @ {p} (24h {ch:+.2f}%)\n"
        f"  SL {state.sl_pct:.2f}% → {sl} · TP {state.tp_pct:.2f}% → {tp}\n"
        f"  lev×{state.leverage} · size {state.size_usdt:.1f} USDT · score {sc}\n"
        f"  RSI15={ind['RSI15']} | RSI30={ind['RSI30']} | RSI60={ind['RSI60']} | MACD{ind['MACD']} | Trend{ind['Trend']}"
    )

async def run_scan(update: Update, context: ContextTypes.DEFAULT_TYPE, push_header=True):
    chat_id = update.effective_chat.id if update else ADMIN_ID
    try:
        strong = compose_strong_signals(max_take=2)
        if push_header:
            hdr = "📈 Сильні сигнали (топ30)" if state.filter_mode=="TOP30" else "📈 Сильні сигнали (whitelist)"
            bot_say(context, chat_id, hdr)

        if not strong:
            bot_say(context, chat_id, "⚠️ Сильних сигналів не знайдено.")
            return

        # надсилаємо список
        msg = "\n\n".join([format_signal(s) for s in strong])
        bot_say(context, chat_id, msg)

        # автотрейд
        if state.trade_enabled:
            bot_say(context, chat_id, "🤖 Працює автотрейд…")
            for s in strong:
                txt = place_trade_for_signal(s)
                bot_say(context, chat_id, txt)
        else:
            bot_say(context, chat_id, "ℹ️ Автотрейд вимкнений. Увімкнути: /trade_on")
    except Exception as e:
        log.exception("scan error")
        bot_say(context, chat_id, f"❌ Помилка скану: {e}")

def place_trade_for_signal(sig):
    symbol = sig["symbol"]
    price  = sig["price"]
    side   = "Buy" if sig["dir"]=="LONG" else "Sell"
    qty    = bybit_calc_qty(symbol, price, state.size_usdt, state.leverage)

    # 1) Leverage
    codeL, dataL = bybit_set_leverage(symbol, state.leverage)
    if codeL != 200:
        return f"❌ Помилка ордера: set-leverage [{codeL}] {dataL}"

    # 2) Market + TP/SL
    codeO, dataO = bybit_place_order(symbol, side, qty, sl_px=sig["sl_px"], tp_px=sig["tp_px"])
    if codeO != 200:
        return f"❌ Помилка ордера: create [{codeO}] {dataO}"

    return f"✅ Ордер відправлено: {symbol} {side} qty={qty} | SL={sig['sl_px']} TP={sig['tp_px']}"

# ---------------------------
# ПЛАНУВАЛЬНИК (APS)
# ---------------------------
scheduler = BackgroundScheduler(timezone=timezone.utc)
scheduler.start(paused=True)

def job_scan(context: CallbackContext):
    # викликається у фоні без update, шлемо в ADMIN_ID
    try:
        strong = compose_strong_signals(max_take=2)
        if not strong:
            bot_say(context, ADMIN_ID, "⚠️ Сильних сигналів не знайдено.")
            return
        hdr = "🤖 Автоторгівля УВІМКНЕНА · Сигнали (топ30)" if state.filter_mode=="TOP30" else "🤖 Автоторгівля УВІМКНЕНА · Сигнали (whitelist)"
        bot_say(context, ADMIN_ID, hdr)
        msg = "\n\n".join([format_signal(s) for s in strong])
        bot_say(context, ADMIN_ID, msg)

        if state.trade_enabled:
            for s in strong:
                txt = place_trade_for_signal(s)
                bot_say(context, ADMIN_ID, txt)
    except Exception as e:
        log.exception("job_scan error")
        bot_say(context, ADMIN_ID, f"❌ Помилка job_scan: {e}")

def scheduler_reschedule(minutes: int):
    try:
        scheduler.remove_job("autoscan")
    except Exception:
        pass
    scheduler.add_job(lambda: job_scan(app.bot), "interval", minutes=minutes, id="autoscan", replace_existing=True)
    scheduler.resume()

# ---------------------------
# MAIN
# ---------------------------
async def heartbeat(context: CallbackContext):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    bot_say(context, ADMIN_ID, f"✅ Bot is alive | UTC {now}")

def check_env():
    if not TG_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty")
    if not ADMIN_ID:
        log.warning("ADMIN_ID is not set — heartbeat і автоповідомлення можуть не доходити")

def build_app():
    application = ApplicationBuilder().token(TG_TOKEN).build()

    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(CommandHandler("signals", signals_cmd))

    application.add_handler(CommandHandler("auto_on", auto_on_cmd))
    application.add_handler(CommandHandler("auto_off", auto_off_cmd))

    application.add_handler(CommandHandler("trade_on", trade_on_cmd))
    application.add_handler(CommandHandler("trade_off", trade_off_cmd))

    application.add_handler(CommandHandler("set_size", set_size_cmd))
    application.add_handler(CommandHandler("set_lev", set_lev_cmd))
    application.add_handler(CommandHandler("set_risk", set_risk_cmd))

    # heartbeat кожні 60 хв (можеш змінити в ENV, якщо вже було)
    application.job_queue.run_repeating(heartbeat, interval=3600, first=30)
    return application

if __name__ == "__main__":
    check_env()
    log.info("Starting bot…")
    app = build_app()
    # якщо автоскан має бути увімкнений відразу (за бажанням можеш виставити state.auto_on=True)
    if state.auto_on:
        scheduler_reschedule(state.auto_minutes)
    app.run_polling(close_loop=False)
