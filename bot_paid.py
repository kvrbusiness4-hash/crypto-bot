# -*- coding: utf-8 -*-
# bot_paid.py

import os, math, time, aiohttp, asyncio, json, traceback, hmac, hashlib
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta, timezone

from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ================== ENV ==================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))

# трейдинг: за замовч. вимкнено
TRADE_ENABLED = os.getenv("TRADE_ENABLED", "OFF").upper() == "ON"
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# ризик / розмір / плече (можна міняти командами)
DEFAULT_SL_PCT = float(os.getenv("SL_PCT", "3"))      # 3%
DEFAULT_TP_PCT = float(os.getenv("TP_PCT", "5"))      # 5%
DEFAULT_SIZE_USDT = float(os.getenv("SIZE_USDT", "20"))  # 20 USDT
DEFAULT_LEVERAGE = int(os.getenv("LEVERAGE", "5"))

# heartbeat
HEARTBEAT_MIN = int(os.getenv("HEARTBEAT_MIN", "60"))

# ===== CoinGecko markets (топ-30) =====
MARKET_URL = ("https://api.coingecko.com/api/v3/coins/markets"
              "?vs_currency=usd&order=market_cap_desc&per_page=120&page=1"
              "&sparkline=true&price_change_percentage=24h")

# STATE: chat_id -> settings
STATE: Dict[int, Dict[str, object]] = {}

# ================== INDICATORS ==================
def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1:
        return series[:]
    k = 2 / (period + 1)
    out = [series[0]]
    for x in series[1:]:
        out.append(out[-1] + k * (x - out[-1]))
    return out

def rsi(series: List[float], period: int = 14) -> List[float]:
    if len(series) < period + 1:
        return [50.0] * len(series)
    gains, losses = [], []
    for i in range(1, len(series)):
        ch = series[i] - series[i-1]
        gains.append(max(0.0, ch))
        losses.append(max(0.0, -ch))
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    out = [100.0 if avg_l == 0 else 100 - 100/(1 + avg_g/avg_l)]
    for i in range(period, len(gains)):
        avg_g = (avg_g*(period-1) + gains[i]) / period
        avg_l = (avg_l*(period-1) + losses[i]) / period
        out.append(100.0 if avg_l == 0 else 100 - 100/(1 + avg_g/avg_l))
    return [50.0]*(len(series)-len(out)) + out

def macd(series: List[float], fast=12, slow=26, signal=9) -> Tuple[List[float], List[float]]:
    if not series or slow + signal >= len(series):
        n=len(series); return [0.0]*n, [0.0]*n
    efast = ema(series, fast)
    eslow = ema(series, slow)
    line = [a-b for a,b in zip(efast, eslow)]
    sig = ema(line, signal)
    return line, sig

# ================== SIGNALS ==================
def decide_signal(prices: List[float]) -> Tuple[str, float, float, str]:
    """
    returns: (direction 'LONG/SHORT/NONE', sl_pct, tp_pct, explain)
    """
    if len(prices) < 40:
        return "NONE", 0, 0, "Недостатньо даних"

    rs = rsi(prices, 14)
    m_line, m_sig = macd(prices)
    r_now = rs[-1]
    macd_hist = m_line[-1] - m_sig[-1]

    direction = "NONE"
    reason = []
    if r_now < 32 and macd_hist > 0:
        direction = "LONG"; reason.append(f"RSI={r_now:.1f}<32 & MACD↑")
    elif r_now > 68 and macd_hist < 0:
        direction = "SHORT"; reason.append(f"RSI={r_now:.1f}>68 & MACD↓")
    else:
        return "NONE", 0, 0, f"RSI={r_now:.1f}, MACD_hist={macd_hist:.4f}"

    return direction, DEFAULT_SL_PCT, DEFAULT_TP_PCT, " | ".join(reason)

def build_signals_text(rows: List[dict], limit: int = 2) -> Tuple[str, List[dict]]:
    """
    Вибираємо 1-2 найсильніші сигнали з топ-30.
    Сила = |RSI-50| + |MACD_hist|/нормалізація.
    """
    scored = []
    for r in rows:
        series = r.get("sparkline_in_7d", {}).get("price", [])[-120:]
        if not series:
            continue
        direction, sl_pct, tp_pct, why = decide_signal(series)
        if direction == "NONE":
            continue
        rs = rsi(series, 14)[-1]
        m_line, m_sig = macd(series)
        hist = abs(m_line[-1] - m_sig[-1])
        score = abs(rs - 50) + min(hist, 5)  # простий скоринг
        scored.append({
            "symbol": (r.get("symbol") or "").upper()+"USDT",
            "name": r.get("name",""),
            "price": r.get("current_price", 0.0),
            "direction": direction,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "why": why,
            "score": score
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    picked = scored[:max(1, min(2, limit))]
    # текст
    lines = ["📈 Сильні сигнали:"]
    for p in picked:
        side = "LONG" if p["direction"]=="LONG" else "SHORT"
        lines.append(
            f"• {p['symbol']}: {side} @ {p['price']:.6g}\n"
            f"  SL: {p['sl_pct']:.2f}% · TP: {p['tp_pct']:.2f}%\n"
            f"  {p['why']}"
        )
    if len(picked)==0:
        lines = ["⚠️ Сильних сигналів не знайдено."]
    return "\n".join(lines), picked

# ================== BYBIT REST (hedge/linear) ==================
BYBIT_HOST = "https://api.bybit.com"
def _ts() -> str:
    return str(int(time.time()*1000))

def _sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

async def bybit_private(session: aiohttp.ClientSession, path: str, body: dict):
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise RuntimeError("BYBIT keys are not set")
    body = body.copy()
    body["api_key"] = BYBIT_API_KEY
    body["timestamp"] = _ts()
    body["recv_window"] = "5000"
    payload = "&".join([f"{k}={body[k]}" for k in sorted(body.keys())])
    body["sign"] = _sign(BYBIT_API_SECRET, payload)
    async with session.post(BYBIT_HOST+path, data=body, timeout=30) as r:
        t = await r.text()
        return json.loads(t)

async def bybit_set_leverage(session: aiohttp.ClientSession, symbol: str, lev: int):
    try:
        return await bybit_private(session, "/v5/position/set-leverage", {
            "category":"linear", "symbol":symbol, "buyLeverage":str(lev), "sellLeverage":str(lev)
        })
    except Exception:
        return None

async def bybit_place(session: aiohttp.ClientSession, symbol: str, side: str,
                      qty_usdt: float, price: float, sl_pct: float, tp_pct: float, lev: int):
    """
    Ринковий ордер + SL/TP через розрахунок від ціни входу.
    """
    await bybit_set_leverage(session, symbol, lev)
    # qty в контрактах (≈ USDT/price для більшості лінійних пар)
    qty = max(0.0001, round(qty_usdt / price, 6))
    order = await bybit_private(session, "/v5/order/create", {
        "category":"linear",
        "symbol":symbol,
        "side":"Buy" if side=="LONG" else "Sell",
        "orderType":"Market",
        "qty":str(qty),
        "timeInForce":"ImmediateOrCancel",
        "reduceOnly":"false",
        "positionIdx":"0"
    })
    # SL/TP
    if side=="LONG":
        sl_price = price * (1 - sl_pct/100)
        tp_price = price * (1 + tp_pct/100)
        tp_side, sl_side = "Sell", "Sell"
    else:
        sl_price = price * (1 + sl_pct/100)
        tp_price = price * (1 - tp_pct/100)
        tp_side, sl_side = "Buy", "Buy"

    await bybit_private(session, "/v5/order/create", {
        "category":"linear","symbol":symbol, "side":tp_side, "orderType":"TakeProfit",
        "triggerDirection":"2","qty":str(qty), "triggerPrice":f"{tp_price:.6f}", "reduceOnly":"true"
    })
    await bybit_private(session, "/v5/order/create", {
        "category":"linear","symbol":symbol, "side":sl_side, "orderType":"StopLoss",
        "triggerDirection":"2","qty":str(qty), "triggerPrice":f"{sl_price:.6f}", "reduceOnly":"true"
    })
    return order

# ================== HELPERS ==================
def chat_state(chat_id:int) -> Dict[str,object]:
    st = STATE.setdefault(chat_id, {})
    st.setdefault("auto_on", False)
    st.setdefault("every", 15)
    st.setdefault("sl_pct", DEFAULT_SL_PCT)
    st.setdefault("tp_pct", DEFAULT_TP_PCT)
    st.setdefault("size_usdt", DEFAULT_SIZE_USDT)
    st.setdefault("lev", DEFAULT_LEVERAGE)
    st.setdefault("use_whitelist", False)   # False => топ-30
    st.setdefault("whitelist", ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","ADAUSDT"])
    return st

def kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [["/signals","/status"],
         ["/auto_on 15","/auto_off"],
         ["/set_size 20","/set_lev 5"],
         ["/set_risk 3 5","/mysub"]],
        resize_keyboard=True
    )

def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

# ================== COMMANDS ==================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    chat_state(chat_id)
    await update.message.reply_text(
        "👋 Готовий!\n\nКоманди:\n"
        "• /signals — сканувати (і авто-трейд, якщо увімкнено)\n"
        "• /auto_on 15 — автопуш кожні N хв (5–120)\n"
        "• /auto_off — вимкнути автопуш\n"
        "• /set_size <USDT> — сума угоди\n"
        "• /set_lev <x> — плече\n"
        "• /set_risk <SL%> <TP%>\n"
        "• /whitelist_on /whitelist_off — режим фільтра\n"
        "• /status — стан",
        reply_markup=kb()
    )

async def set_size_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = chat_state(chat_id)
    try:
        size = float(context.args[0])
        st["size_usdt"] = max(1.0, size)
        await update.message.reply_text(f"✅ Розмір угоди встановлено: {st['size_usdt']:.2f} USDT")
    except Exception:
        await update.message.reply_text("Приклад: /set_size 15")

async def set_lev_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = chat_state(chat_id)
    try:
        lev = int(context.args[0])
        st["lev"] = max(1, min(50, lev))
        await update.message.reply_text(f"✅ Плече встановлено: x{st['lev']}")
    except Exception:
        await update.message.reply_text("Приклад: /set_lev 5")

async def set_risk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = chat_state(chat_id)
    try:
        sl = float(context.args[0]); tp = float(context.args[1])
        st["sl_pct"] = max(0.1, sl)
        st["tp_pct"] = max(0.1, tp)
        await update.message.reply_text(f"✅ Ризик: SL={st['sl_pct']:.2f}% · TP={st['tp_pct']:.2f}%")
    except Exception:
        await update.message.reply_text("Приклад: /set_risk 3 5")

async def whitelist_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    st = chat_state(update.effective_chat.id)
    st["use_whitelist"] = True
    await update.message.reply_text("✅ Увімкнено whitelist (лише ваш список).")

async def whitelist_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    st = chat_state(update.effective_chat.id)
    st["use_whitelist"] = False
    await update.message.reply_text("✅ Вимкнено whitelist. Сканую ТОП-30 ринків.")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = chat_state(chat_id)
    txt = (f"Статус: {'ON' if st['auto_on'] else 'OFF'} · кожні {st['every']} хв.\n"
           f"SL={st['sl_pct']:.2f}% · TP={st['tp_pct']:.2f}%\n"
           f"TRADE_ENABLED={'ON' if TRADE_ENABLED else 'OFF'} · SIZE={st['size_usdt']:.2f} USDT · LEV={st['lev']}\n"
           f"Фільтр: {'WHITELIST' if st['use_whitelist'] else 'TOP30'}\n"
           f"UTC: {now_utc_str()}")
    await update.message.reply_text(txt)

async def auto_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = chat_state(chat_id)
    try:
        minutes = int(context.args[0]) if context.args else 15
        minutes = max(5, min(120, minutes))
        st["auto_on"] = True
        st["every"] = minutes
        name = f"auto_{chat_id}"
        for j in context.application.job_queue.get_jobs_by_name(name):
            j.schedule_removal()
        context.application.job_queue.run_repeating(
            auto_push_job, interval=timedelta(minutes=minutes), first=1, name=name, data={"chat_id":chat_id}
        )
        await update.message.reply_text(f"Автопуш увімкнено: кожні {minutes} хв.")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Помилка: {e}")

async def auto_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = chat_state(chat_id)
    st["auto_on"] = False
    name = f"auto_{chat_id}"
    for j in context.application.job_queue.get_jobs_by_name(name):
        j.schedule_removal()
    await update.message.reply_text("Автопуш вимкнено.")

async def signals_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await send_signals_flow(context, chat_id)

async def auto_push_job(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    st = chat_state(chat_id)
    if not st["auto_on"]:
        return
    await send_signals_flow(context, chat_id)

async def send_signals_flow(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    st = chat_state(chat_id)
    try:
        rows = await fetch_market_rows()
        # фільтрація
        if st["use_whitelist"]:
            wl = set(st["whitelist"])
            rows = [r for r in rows if (r.get("symbol","").upper()+"USDT") in wl]
        else:
            # ТОП-30 за маркет-кап
            rows = rows[:30]

        txt, picked = build_signals_text(rows, limit=2)
        # надсилаємо текст
        await context.bot.send_message(chat_id=chat_id, text=txt, parse_mode=ParseMode.MARKDOWN)

        # трейдинг (опційно)
        if TRADE_ENABLED and picked:
            async with aiohttp.ClientSession() as s:
                for p in picked:
                    symbol = p["symbol"]
                    side = p["direction"]
                    price = p["price"] or 0.0
                    sl_pct = st["sl_pct"]; tp_pct = st["tp_pct"]
                    size = st["size_usdt"]; lev = st["lev"]
                    try:
                        resp = await bybit_place(s, symbol, side, size*lev, price, sl_pct, tp_pct, lev)
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=(f"✅ Відкрито {side} {symbol} · {size} USDT ×{lev}\n"
                                  f"SL={sl_pct:.2f}% · TP={tp_pct:.2f}%\n"
                                  f"Resp: {str(resp)[:200]}...")
                        )
                    except Exception as e:
                        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Trade error {symbol}: {e}")
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Автопуш помилка: {e}")

# ================== MARKET FETCH ==================
async def fetch_market_rows() -> List[dict]:
    async with aiohttp.ClientSession() as s:
        async with s.get(MARKET_URL, timeout=30) as r:
            if r.status != 200:
                raise RuntimeError(f"Market error {r.status}")
            return await r.json()

# ================== HEARTBEAT ==================
async def heartbeat(_: ContextTypes.DEFAULT_TYPE):
    if not ADMIN_ID:
        return
    try:
        now = now_utc_str()
        await _.bot.send_message(ADMIN_ID, f"✅ Bot is alive | UTC {now}")
    except Exception as e:
        print(f"[heartbeat] send failed: {e}")

# ================== APP ==================
def build_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))
    app.add_handler(CommandHandler("auto_on", auto_on_cmd))
    app.add_handler(CommandHandler("auto_off", auto_off_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("set_risk", set_risk_cmd))
    app.add_handler(CommandHandler("set_size", set_size_cmd))
    app.add_handler(CommandHandler("set_lev", set_lev_cmd))
    app.add_handler(CommandHandler("whitelist_on", whitelist_on_cmd))
    app.add_handler(CommandHandler("whitelist_off", whitelist_off_cmd))

    jq = app.job_queue
    jq.run_repeating(heartbeat, interval=timedelta(minutes=HEARTBEAT_MIN), first=10, name="heartbeat")

    return app

if __name__ == "__main__":
    print("Starting bot…")
    app = build_app()
    app.run_polling()
