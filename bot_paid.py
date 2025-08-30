# -*- coding: utf-8 -*-
# Telegram → Scan top30 → pick 1–2 strong → (optional) auto-trade on Bybit (USDT Perp)
# Commands: /start /signals /auto_on N /auto_off /status /trade_on /trade_off
#           /set_size 5  /set_lev 3  /set_risk 0.8 1.6  /set_whitelist BTC,ETH,...
# Env vars: TELEGRAM_BOT_TOKEN, ADMIN_ID, BYBIT_API_KEY, BYBIT_API_SECRET,
#           DEFAULT_SIZE_USDT, DEFAULT_LEVERAGE, RISK_SL_PCT, RISK_TP_PCT,
#           AUTO_SCAN_MIN, TRADE_ENABLED (ON/OFF), SYMBOL_WHITELIST (comma)
# Run:  python bot_bybit_auto.py

import os, math, hmac, hashlib, time, json, asyncio, aiohttp
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ================= ENV / GLOBALS =================
TG_TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_ID        = int(os.getenv("ADMIN_ID", "0"))
BYBIT_KEY       = os.getenv("BYBIT_API_KEY", "").strip()
BYBIT_SECRET    = os.getenv("BYBIT_API_SECRET", "").strip()
TRADE_ENABLED   = os.getenv("TRADE_ENABLED", "OFF").upper() == "ON"   # стартовий прапор
AUTO_SCAN_MIN   = int(os.getenv("AUTO_SCAN_MIN", "15"))               # інтервал автоскану хв
DEFAULT_SIZE    = float(os.getenv("DEFAULT_SIZE_USDT", "5"))          # сума угоди (USDT)
DEFAULT_LEV     = int(os.getenv("DEFAULT_LEVERAGE", "3"))             # плече
RISK_SL_PCT     = float(os.getenv("RISK_SL_PCT", "0.8"))              # SL %
RISK_TP_PCT     = float(os.getenv("RISK_TP_PCT", "1.6"))              # TP %
WHITELIST_ENV   = os.getenv("SYMBOL_WHITELIST", "").strip()           # "BTC,ETH,OP"
SYMBOL_WHITELIST= [s.strip().upper() for s in WHITELIST_ENV.split(",") if s.strip()]

BYBIT_BASE      = "https://api.bybit.com"  # Unified V5 mainnet
CATEGORY        = "linear"                 # USDT Perps

# Авто-стан по чатах (ти — один адмін, але лишимо гнучко)
STATE: Dict[int, Dict[str, object]] = {}  # {"auto_on":bool,"every":int,"trade_on":bool,"size":float,"lev":int,"sl":float,"tp":float}

# CoinGecko
MARKET_URL = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=market_cap_desc&per_page=120&page=1"
    "&sparkline=true&price_change_percentage=24h"
)
STABLES = {"USDT","USDC","DAI","TUSD","FDUSD","USDD","PYUSD","EURS","EURT","BUSD"}

# ================== TA (RSI/MACD/EMA) ==================
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
        return []
    gains, losses = [], []
    for i in range(1, len(series)):
        ch = series[i] - series[i-1]
        gains.append(max(0.0, ch))
        losses.append(max(0.0, -ch))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out = [0.0] * (period)
    if avg_loss == 0:
        out.append(100.0)
    else:
        out.append(100.0 - (100.0 / (1.0 + (avg_gain/(avg_loss+1e-9)))))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        if avg_loss == 0:
            out.append(100.0)
        else:
            rs = avg_gain/(avg_loss+1e-9)
            out.append(100.0 - (100.0/(1.0+rs)))
    return out

def macd(series: List[float], fast:int=12, slow:int=26, signal:int=9):
    if len(series) < slow + signal:
        return [], []
    ef = ema(series, fast)
    es = ema(series, slow)
    macd_line = [a-b for a,b in zip(ef[-len(es):], es)]
    sig = ema(macd_line, signal)
    L = min(len(macd_line), len(sig))
    return macd_line[-L:], sig[-L:]

def decide_signal(prices: List[float], p24: Optional[float]) -> Tuple[str, float, float, str]:
    """
    -> (direction LONG/SHORT/NONE, sl_pct, tp_pct, note)
    sl_pct/tp_pct — повертаємо динамічні % (вола + контекст), але при вході в угоду
    використаємо саме ті, що стоять у STATE (налаштовувані), якщо ти їх поміняв.
    """
    explain: List[str] = []
    if not prices or len(prices) < 40:
        return "NONE", 0.0, 0.0, "недостатньо даних"

    series = prices[-120:]
    px = series[-1]

    ema50 = ema(series, 50)
    ema200 = ema(series, min(200, len(series)//2 if len(series) >= 200 else 100))
    trend = 0
    if ema50 and ema200:
        if ema50[-1] > ema200[-1]: trend = 1
        elif ema50[-1] < ema200[-1]: trend = -1

    rsi15 = rsi(series, 7)
    rsi30 = rsi(series, 14)
    rsi60 = rsi(series, 28)
    m_line, m_sig = macd(series)

    votes = 0
    def vote_rsi(x):
        if x is None: return 0
        if x <= 30: return +1
        if x >= 70: return -1
        return 0
    if rsi15: votes += vote_rsi(rsi15[-1]); explain.append(f"RSI15={rsi15[-1]:.1f}")
    if rsi30: votes += vote_rsi(rsi30[-1]); explain.append(f"RSI30={rsi30[-1]:.1f}")
    if rsi60: votes += vote_rsi(rsi60[-1]); explain.append(f"RSI60={rsi60[-1]:.1f}")

    if m_line and m_sig:
        if m_line[-1] > m_sig[-1]: votes += 1; explain.append("MACD↑")
        elif m_line[-1] < m_sig[-1]: votes -= 1; explain.append("MACD↓")

    if trend > 0: votes += 1; explain.append("Trend↑")
    elif trend < 0: votes -= 1; explain.append("Trend↓")

    direction = "NONE"
    if votes >= 2: direction = "LONG"
    elif votes <= -2: direction = "SHORT"

    # Волатильність — рекомендація SL/TP (як підказка)
    tail = series[-48:] if len(series) >= 48 else series
    if len(tail) >= 2:
        mean = sum(tail)/len(tail)
        var = sum((x-mean)**2 for x in tail)/len(tail)
        stdev = math.sqrt(var)
        vol = (stdev/px) * 100.0
    else:
        vol = 1.0
    ctx = abs(p24 or 0.0)
    rec_sl = max(0.5, min(3.0, 0.6*vol + ctx/3.0))
    rec_tp = max(0.8, min(5.0, 1.1*vol + ctx/2.5))

    return direction, rec_sl, rec_tp, " | ".join(explain)

# ================== Market fetch ==================
async def fetch_market(session: aiohttp.ClientSession) -> List[dict]:
    async with session.get(MARKET_URL, timeout=25) as r:
        r.raise_for_status()
        return await r.json()

def is_good_symbol(item: dict) -> bool:
    sym = (item.get("symbol") or "").upper()
    name = (item.get("name") or "").upper()
    if sym in STABLES or any(s in name for s in STABLES):
        return False
    return True

def cg_to_bybit_symbol(cg_symbol: str) -> str:
    # CG symbol 'btc' → Bybit 'BTCUSDT'
    return cg_symbol.upper() + "USDT"

# ================== Bybit REST (Unified v5) ==================
def _ts_ms() -> str:
    return str(int(time.time()*1000))

def _sign(payload: str) -> str:
    return hmac.new(BYBIT_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

async def bybit_private(session: aiohttp.ClientSession, method: str, path: str, body: dict):
    if not BYBIT_KEY or not BYBIT_SECRET:
        raise RuntimeError("Bybit API keys not set.")
    ts = _ts_ms()
    recv_window = "5000"
    q = ""  # no query for these endpoints
    body_str = json.dumps(body)
    payload = ts + BYBIT_KEY + recv_window + body_str
    sign = _sign(payload)
    headers = {
        "X-BAPI-API-KEY": BYBIT_KEY,
        "X-BAPI-TIMESTAMP": ts,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN": sign,
        "Content-Type": "application/json",
    }
    url = BYBIT_BASE + path
    async with session.request(method, url, headers=headers, data=body_str, timeout=25) as r:
        data = await r.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit error {data.get('retCode')}: {data.get('retMsg')}")
        return data.get("result")

async def bybit_set_leverage(session, symbol: str, leverage: int):
    body = {"category": CATEGORY, "symbol": symbol, "buyLeverage": str(leverage), "sellLeverage": str(leverage)}
    return await bybit_private(session, "POST", "/v5/position/set-leverage", body)

async def bybit_set_tpsl_mode(session, symbol: str):
    body = {"category": CATEGORY, "symbol": symbol, "tpSlMode": "Full"}
    return await bybit_private(session, "POST", "/v5/position/set-tpsl-mode", body)

async def bybit_create_order(session, symbol: str, side: str, qty: float, take_profit: float, stop_loss: float):
    body = {
        "category": CATEGORY,
        "symbol": symbol,
        "side": side,                 # "Buy" / "Sell"
        "orderType": "Market",
        "qty": f"{qty}",
        "tpslMode": "Full",
        "takeProfit": f"{take_profit}",
        "stopLoss": f"{stop_loss}",
        "timeInForce": "IOC",
        "reduceOnly": False
    }
    return await bybit_private(session, "POST", "/v5/order/create", body)

# ================== Signal build & (optional) trade ==================
async def build_signals_and_maybe_trade(chat_id: int, take_trades: bool) -> str:
    st = STATE.setdefault(chat_id, {
        "auto_on": False, "every": AUTO_SCAN_MIN,
        "trade_on": TRADE_ENABLED, "size": DEFAULT_SIZE,
        "lev": DEFAULT_LEV, "sl": RISK_SL_PCT, "tp": RISK_TP_PCT
    })
    size_usdt = float(st["size"]); lev = int(st["lev"])
    use_sl = float(st["sl"]); use_tp = float(st["tp"])

    txt_lines: List[str] = []
    async with aiohttp.ClientSession() as s:
        market = await fetch_market(s)
        # відберемо топ-30 по cap (це перші 30 у відповіді)
        candidates = [m for m in market if is_good_symbol(m)][:30]
        if SYMBOL_WHITELIST:
            # якщо whitelist заданий – беремо тільки ці символи
            wl = set(SYMBOL_WHITELIST)
            candidates = [m for m in candidates if (m.get("symbol") or "").upper() in wl]

        scored = []
        for it in candidates:
            prices = (((it.get("sparkline_in_7d") or {}).get("price")) or [])
            p24 = it.get("price_change_percentage_24h")
            direction, rec_sl, rec_tp, note = decide_signal(prices, p24)
            if direction == "NONE":
                continue
            score = 2.0
            if p24 is not None:
                score += min(2, abs(p24)/10.0)
            scored.append((score, direction, rec_sl, rec_tp, note, it))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:2]  # 1–2 сильні
        if not top:
            return "⚠️ Немає сильних підтверджених сигналів зараз."

        # можлива торгівля
        for sc, direction, rec_sl, rec_tp, note, it in top:
            sym_cg = (it.get("symbol") or "").upper()
            px = float(it.get("current_price") or 0.0)
            bybit_sym = cg_to_bybit_symbol(sym_cg)
            p24 = it.get("price_change_percentage_24h") or 0.0

            # використовуємо твої налаштування SL/TP (%)
            sl_pct = use_sl
            tp_pct = use_tp
            if direction == "LONG":
                sl_price = px * (1 - sl_pct/100.0)
                tp_price = px * (1 + tp_pct/100.0)
                side = "Buy"
            else:
                sl_price = px * (1 + sl_pct/100.0)
                tp_price = px * (1 - tp_pct/100.0)
                side = "Sell"

            # qty у монетах
            qty = max(0.0001, round(size_usdt / px, 6))

            line = (
                f"• {sym_cg}: *{direction}* @ {px:.4f} (24h {p24:+.2f}%)\n"
                f"  SL {sl_pct:.2f}% → `{sl_price:.6f}` · TP {tp_pct:.2f}% → `{tp_price:.6f}`\n"
                f"  lev×{lev} · size {size_usdt} USDT · score {sc:.2f}\n"
                f"  {note}"
            )
            txt_lines.append(line)

            # якщо дозволено — створимо ордер
            if take_trades and (chat_id == ADMIN_ID) and BYBIT_KEY and BYBIT_SECRET:
                try:
                    await bybit_set_leverage(s, bybit_sym, lev)
                    await bybit_set_tpsl_mode(s, bybit_sym)
                    await bybit_create_order(s, bybit_sym, side, qty, tp_price, sl_price)
                    txt_lines.append(f"  ✅ Ордер відправлено на Bybit ({bybit_sym}) qty={qty}")
                except Exception as e:
                    txt_lines.append(f"  ❌ Помилка ордера: {e}")

    header = "📈 *Сильні сигнали (топ30)*\n"
    if take_trades:
        header = "🤖 *Автоторгівля УВІМКНЕНА* · Сигнали (топ30)\n"
    return header + "\n\n" + "\n\n".join(txt_lines)

# ================== Telegram UI ==================
KB = ReplyKeyboardMarkup(
    [["/signals", "/status"], ["/auto_on 15", "/auto_off"], ["/trade_on", "/trade_off"]],
    resize_keyboard=True
)

def ensure_state(chat_id: int):
    STATE.setdefault(chat_id, {
        "auto_on": False, "every": AUTO_SCAN_MIN,
        "trade_on": TRADE_ENABLED, "size": DEFAULT_SIZE,
        "lev": DEFAULT_LEV, "sl": RISK_SL_PCT, "tp": RISK_TP_PCT
    })

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_state(update.effective_chat.id)
    await update.message.reply_text(
        "👋 Готовий.\n"
        "Команди:\n"
        "• /signals — сканувати зараз\n"
        "• /auto_on 15 — автоскан кожні N хв (5–120)\n"
        "• /auto_off — вимкнути автоскан\n"
        "• /trade_on /trade_off — увімк/вимк автоторгівлю\n"
        "• /set_size 5 — сума угоди (USDT)\n"
        "• /set_lev 3 — плече\n"
        "• /set_risk 0.8 1.6 — SL% TP%\n"
        "• /set_whitelist BTC,ETH — обмежити монети (optional)\n"
        "• /status — показати налаштування",
        reply_markup=KB
    )

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_state(update.effective_chat.id)
    st = STATE[update.effective_chat.id]
    wl = (", ".join(SYMBOL_WHITELIST)) if SYMBOL_WHITELIST else "ALL(top30)"
    await update.message.reply_text(
        "⚙️ *Статус*\n"
        f"AutoScan: {'ON' if st['auto_on'] else 'OFF'} · кожні {st['every']} хв\n"
        f"AutoTrade: {'ON' if st['trade_on'] else 'OFF'} (дозволено лише ADMIN_ID)\n"
        f"Size: {st['size']} USDT · Lev: x{st['lev']}\n"
        f"Risk: SL {st['sl']}% · TP {st['tp']}%\n"
        f"Whitelist: {wl}",
        parse_mode=ParseMode.MARKDOWN
    )

async def signals_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_state(update.effective_chat.id)
    st = STATE[update.effective_chat.id]
    take_trades = bool(st["trade_on"]) and (update.effective_user.id == ADMIN_ID)
    txt = await build_signals_and_maybe_trade(update.effective_chat.id, take_trades)
    await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

async def auto_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_state(update.effective_chat.id)
    st = STATE[update.effective_chat.id]
    minutes = st["every"]
    if context.args:
        try:
            minutes = max(5, min(120, int(context.args[0])))
        except:
            pass
    st["every"] = minutes
    st["auto_on"] = True

    name = f"auto_{update.effective_chat.id}"
    for j in context.application.job_queue.get_jobs_by_name(name):
        j.schedule_removal()
    context.application.job_queue.run_repeating(
        auto_job, interval=minutes*60, first=5,
        name=name, data={"chat_id": update.effective_chat.id}
    )
    await update.message.reply_text(f"Автоскан увімкнено: кожні {minutes} хв.")

async def auto_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_state(update.effective_chat.id)
    st = STATE[update.effective_chat.id]
    st["auto_on"] = False
    name = f"auto_{update.effective_chat.id}"
    for j in context.application.job_queue.get_jobs_by_name(name):
        j.schedule_removal()
    await update.message.reply_text("Автоскан вимкнено.")

async def trade_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_state(update.effective_chat.id)
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("Лише для адміністратора.")
        return
    STATE[update.effective_chat.id]["trade_on"] = True
    await update.message.reply_text("🤖 Автоторгівлю УВІМКНЕНО.")

async def trade_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_state(update.effective_chat.id)
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("Лише для адміністратора.")
        return
    STATE[update.effective_chat.id]["trade_on"] = False
    await update.message.reply_text("🛑 Автоторгівлю ВИМКНЕНО.")

async def set_size_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_state(update.effective_chat.id)
    if not context.args:
        await update.message.reply_text("Використання: /set_size 5")
        return
    try:
        v = max(1e-3, float(context.args[0]))
        STATE[update.effective_chat.id]["size"] = v
        await update.message.reply_text(f"OK. Size = {v} USDT")
    except:
        await update.message.reply_text("Невірне число.")

async def set_lev_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_state(update.effective_chat.id)
    if not context.args:
        await update.message.reply_text("Використання: /set_lev 3")
        return
    try:
        lv = max(1, min(50, int(context.args[0])))
        STATE[update.effective_chat.id]["lev"] = lv
        await update.message.reply_text(f"OK. Leverage = x{lv}")
    except:
        await update.message.reply_text("Невірне число.")

async def set_risk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_state(update.effective_chat.id)
    if len(context.args) < 2:
        await update.message.reply_text("Використання: /set_risk 0.8 1.6  (SL% TP%)")
        return
    try:
        sl = max(0.1, float(context.args[0]))
        tp = max(0.1, float(context.args[1]))
        STATE[update.effective_chat.id]["sl"] = sl
        STATE[update.effective_chat.id]["tp"] = tp
        await update.message.reply_text(f"OK. Risk: SL {sl}% · TP {tp}%")
    except:
        await update.message.reply_text("Невірні числа.")

async def set_whitelist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SYMBOL_WHITELIST
    if not context.args:
        SYMBOL_WHITELIST = []
        await update.message.reply_text("Whitelist очищено. Використовую всі з топ-30.")
        return
    raw = " ".join(context.args)
    parts = [p.strip().upper() for p in raw.replace(",", " ").split() if p.strip()]
    SYMBOL_WHITELIST = parts
    await update.message.reply_text("OK. Whitelist: " + ", ".join(SYMBOL_WHITELIST))

# авто-джоб
async def auto_job(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    st = STATE.get(chat_id, {})
    if not st or not st.get("auto_on"):
        return
    take_trades = bool(st.get("trade_on", False)) and (chat_id == ADMIN_ID)
    try:
        txt = await build_signals_and_maybe_trade(chat_id, take_trades)
        await context.bot.send_message(chat_id=chat_id, text=txt, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Auto error: {e}")

# Heartbeat (щогодини)
async def heartbeat(context: ContextTypes.DEFAULT_TYPE):
    if not ADMIN_ID:
        return
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    await context.bot.send_message(chat_id=ADMIN_ID, text=f"✅ Bot alive · {now} UTC")

# ================== MAIN ==================
def main():
    if not TG_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN")
        return

    app = Application.builder().token(TG_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))
    app.add_handler(CommandHandler("auto_on", auto_on_cmd))
    app.add_handler(CommandHandler("auto_off", auto_off_cmd))
    app.add_handler(CommandHandler("trade_on", trade_on_cmd))
    app.add_handler(CommandHandler("trade_off", trade_off_cmd))
    app.add_handler(CommandHandler("set_size", set_size_cmd))
    app.add_handler(CommandHandler("set_lev", set_lev_cmd))
    app.add_handler(CommandHandler("set_risk", set_risk_cmd))
    app.add_handler(CommandHandler("set_whitelist", set_whitelist_cmd))

    # heartbeat щогодини
    app.job_queue.run_repeating(heartbeat, interval=3600, first=15)

    print("Bot running… (top30 scan + optional Bybit auto-trade)")
    app.run_polling()

if __name__ == "__main__":
    main()
