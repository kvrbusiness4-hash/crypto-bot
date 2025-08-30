# bot_trade.py
# -*- coding: utf-8 -*-

import os, math, hmac, hashlib, time, json, aiohttp, asyncio
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ======================= ENV =======================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Сила сигналів / фільтри (можеш міняти у Railway Variables)
STRONG_VOTES = int(os.getenv("STRONG_VOTES", "3"))                  # мінімум голосів індикаторів
REQUIRE_TREND_CONFIRM = os.getenv("REQUIRE_TREND_CONFIRM", "1") == "1"
MIN_24H_VOLUME_USD = float(os.getenv("MIN_24H_VOLUME_USD", "0"))

# SL/TP у % (дефолти; /set_risk перепише для чату)
SL_PCT_DEFAULT = float(os.getenv("SL_PCT_DEFAULT", "1.0"))          # 1%
TP_PCT_DEFAULT = float(os.getenv("TP_PCT_DEFAULT", "2.0"))          # 2%

# Автотрейд
TRADE_ENABLED = os.getenv("TRADE_ENABLED", "0") == "1"              # 1=вмикає торгівлю
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_BASE = os.getenv("BYBIT_BASE", "https://api.bybit.com")       # real
TRADE_SIZE_USDT = float(os.getenv("TRADE_SIZE_USDT", "20"))         # розмір позиції в USDT
LEVERAGE = int(os.getenv("LEVERAGE", "5"))                          # плече
LOT_DECIMALS = int(os.getenv("LOT_DECIMALS", "3"))                  # округлення кількості
# Кома-розділений whitelist символів Bybit (USDT-perp): "BTCUSDT,ETHUSDT,SOLUSDT"
TRADE_WHITELIST = [s.strip().upper() for s in os.getenv("TRADE_WHITELIST", "BTCUSDT,ETHUSDT,SOLUSDT").split(",") if s.strip()]

# ========= CoinGecko =========
MARKET_URL = ("https://api.coingecko.com/api/v3/coins/markets"
              "?vs_currency=usd&order=market_cap_desc&per_page=120&page=1"
              "&sparkline=true&price_change_percentage=24h")
STABLES = {"USDT","USDC","DAI","TUSD","FDUSD","USDD","PYUSD","EURS","EURT","BUSD"}

# ======= State (in-memory) =======
# chat_id -> {"auto_on": bool, "every": int, "sl_pct": float, "tp_pct": float}
STATE: Dict[int, Dict[str, float | int | bool]] = {}

# ======================= Indicators =======================
def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1: return series[:]
    k = 2 / (period + 1)
    out = [series[0]]
    for x in series[1:]:
        out.append(out[-1] + k * (x - out[-1]))
    return out

def rsi(series: List[float], period: int = 14) -> List[float]:
    if len(series) < period + 1: return []
    gains, losses = [], []
    for i in range(1, len(series)):
        ch = series[i] - series[i-1]
        gains.append(max(0.0, ch))
        losses.append(max(0.0, -ch))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [0.0] * period
    if avg_loss == 0: rsis.append(100.0)
    else: rsis.append(100.0 - (100.0 / (1.0 + (avg_gain/(avg_loss+1e-9)))))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        if avg_loss == 0: rsis.append(100.0)
        else:
            rs = avg_gain/(avg_loss+1e-9)
            rsis.append(100.0 - (100.0/(1.0+rs)))
    return rsis

def macd(series: List[float], fast:int=12, slow:int=26, signal:int=9) -> Tuple[List[float], List[float]]:
    if len(series) < slow + signal: return [], []
    ema_fast = ema(series, fast); ema_slow = ema(series, slow)
    macd_line = [a-b for a,b in zip(ema_fast[-len(ema_slow):], ema_slow)]
    sig = ema(macd_line, signal)
    L = min(len(macd_line), len(sig))
    return macd_line[-L:], sig[-L:]

# ======================= Risk (SL/TP %) =======================
def get_user_risk(chat_id: int) -> Tuple[float, float]:
    st = STATE.setdefault(chat_id, {"auto_on": False, "every": 15})
    sl = float(st.get("sl_pct", SL_PCT_DEFAULT))
    tp = float(st.get("tp_pct", TP_PCT_DEFAULT))
    return sl, tp

# ======================= Signal logic =======================
def decide_signal(prices: List[float], p24: Optional[float], sl_pct: float, tp_pct: float):
    """
    -> (direction LONG/SHORT/NONE, sl_price, tp_price, note, votes, trend)
    trend: +1 up, -1 down, 0 flat
    """
    explain: List[str] = []
    if not prices or len(prices) < 40:
        return "NONE", 0.0, 0.0, "недостатньо даних", 0, 0

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
    macd_line, macd_sig = macd(series)

    votes = 0
    def rsi_vote(last: float) -> int:
        if last is None: return 0
        if last <= 30: return +1
        if last >= 70: return -1
        return 0

    if rsi15: votes += rsi_vote(rsi15[-1]); explain.append(f"RSI15={rsi15[-1]:.1f}")
    if rsi30: votes += rsi_vote(rsi30[-1]); explain.append(f"RSI30={rsi30[-1]:.1f}")
    if rsi60: votes += rsi_vote(rsi60[-1]); explain.append(f"RSI60={rsi60[-1]:.1f}")

    if macd_line and macd_sig:
        if macd_line[-1] > macd_sig[-1]: votes += 1; explain.append("MACD↑")
        elif macd_line[-1] < macd_sig[-1]: votes -= 1; explain.append("MACD↓")
        else: explain.append("MACD·")

    if trend > 0: votes += 1; explain.append("Trend=UP")
    elif trend < 0: votes -= 1; explain.append("Trend=DOWN")
    else: explain.append("Trend=FLAT")

    direction = "NONE"
    if votes >= 2: direction = "LONG"
    elif votes <= -2: direction = "SHORT"

    # SL/TP у % → в ціни
    if direction == "LONG":
        sl_price = px * (1 - sl_pct/100.0)
        tp_price = px * (1 + tp_pct/100.0)
    elif direction == "SHORT":
        sl_price = px * (1 + sl_pct/100.0)
        tp_price = px * (1 - tp_pct/100.0)
    else:
        sl_price = 0.0; tp_price = 0.0

    return direction, round(sl_price, 6), round(tp_price, 6), " | ".join(explain), votes, trend

# ======================= Market fetch =======================
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

# ======================= Bybit v5 client =======================
class BybitV5:
    def __init__(self, base: str, key: str, secret: str):
        self.base = base.rstrip("/")
        self.key = key
        self.secret = secret.encode("utf-8")

    def _sign(self, ts: str, recv: str, body: str = "") -> str:
        # v5 signature = HMAC_SHA256( timestamp + apiKey + recvWindow + body )
        pre = f"{ts}{self.key}{recv}{body}"
        return hmac.new(self.secret, pre.encode("utf-8"), hashlib.sha256).hexdigest()

    async def _post(self, path: str, payload: dict):
        ts = str(int(time.time() * 1000))
        recv = "20000"
        body = json.dumps(payload)
        sign = self._sign(ts, recv, body)
        headers = {
            "Content-Type": "application/json",
            "X-BAPI-API-KEY": self.key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": recv,
            "X-BAPI-SIGN": sign
        }
        url = f"{self.base}{path}"
        async with aiohttp.ClientSession() as s:
            async with s.post(url, headers=headers, data=body, timeout=25) as r:
                data = await r.json()
                return r.status, data

    async def set_leverage(self, symbol: str, buy_leverage: int, sell_leverage: int):
        payload = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(buy_leverage),
            "sellLeverage": str(sell_leverage)
        }
        return await self._post("/v5/position/set-leverage", payload)

    async def create_order(self, symbol: str, side: str, qty: float,
                           sl_price: Optional[float], tp_price: Optional[float]):
        """
        MARKET order with SL/TP attached
        """
        payload = {
            "category": "linear",
            "symbol": symbol,
            "side": side,                 # "Buy" / "Sell"
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "IOC",
        }
        # Додаємо SL/TP якщо задані
        if sl_price: payload["stopLoss"] = str(sl_price)
        if tp_price: payload["takeProfit"] = str(tp_price)

        return await self._post("/v5/order/create", payload)

# ======================= Trade helper =======================
def base_to_bybit_symbol(base: str) -> Optional[str]:
    """Перетворює CG base (BTC) у Bybit USDT-perp (BTCUSDT) та перевіряє whitelist."""
    sym = (base or "").upper() + "USDT"
    return sym if sym in TRADE_WHITELIST else None

def round_qty(px: float, usdt: float) -> float:
    if px <= 0: return 0.0
    q = usdt / px
    fmt = f"{{:.{LOT_DECIMALS}f}}"
    return float(fmt.format(q))

async def maybe_trade(symbol_base: str, direction: str, px: float,
                      sl_price: float, tp_price: float) -> str:
    """
    Якщо TRADE_ENABLED=1 -> ставимо ордер на Bybit (market) + SL/TP.
    Повертає короткий текст-результат (для логів у чат).
    """
    if not TRADE_ENABLED:
        return "TRADE_DISABLED"

    if not (BYBIT_API_KEY and BYBIT_API_SECRET):
        return "Bybit keys not set"

    bybit_symbol = base_to_bybit_symbol(symbol_base)
    if not bybit_symbol:
        return f"{symbol_base}: not in whitelist"

    client = BybitV5(BYBIT_BASE, BYBIT_API_KEY, BYBIT_API_SECRET)

    # Set leverage (разово перед ордером; зайве — але просто)
    code, data = await client.set_leverage(bybit_symbol, LEVERAGE, LEVERAGE)
    if code != 200 or (data.get("retCode") not in (0, "0")):
        # продовжимо все одно — деякі акаунти вже мають виставлений левередж
        pass

    qty = round_qty(px, TRADE_SIZE_USDT)
    if qty <= 0:
        return "qty=0"

    side = "Buy" if direction == "LONG" else "Sell"
    code, data = await client.create_order(bybit_symbol, side, qty, sl_price, tp_price)
    if code == 200 and (data.get("retCode") in (0, "0")):
        oid = ((data.get("result") or {}).get("orderId")) or "OK"
        return f"ORDER OK {bybit_symbol} {side} qty={qty} SL={sl_price} TP={tp_price} id={oid}"
    else:
        return f"ORDER FAIL {bybit_symbol}: {data}"

# ======================= Signals builder =======================
async def build_signals_text(top_n: int, chat_id: int) -> str:
    sl_pct, tp_pct = get_user_risk(chat_id)
    text_lines: List[str] = []

    async with aiohttp.ClientSession() as s:
        try:
            market = await fetch_market(s)
        except Exception as e:
            return f"⚠️ Помилка ринку: {e}"

    candidates = [m for m in market if is_good_symbol(m)]
    if not candidates:
        return "⚠️ Ринок недоступний або немає кандидатів."

    scored = []
    for it in candidates:
        vol = float(it.get("total_volume") or 0.0)
        if MIN_24H_VOLUME_USD and vol < MIN_24H_VOLUME_USD:
            continue

        prices = (((it.get("sparkline_in_7d") or {}).get("price")) or [])
        p24 = it.get("price_change_percentage_24h")
        direction, sl, tp, note, votes, trend = decide_signal(prices, p24, sl_pct, tp_pct)

        if direction == "NONE": continue
        if votes < STRONG_VOTES: continue
        if REQUIRE_TREND_CONFIRM:
            if direction == "LONG" and trend <= 0: continue
            if direction == "SHORT" and trend >= 0: continue

        score = votes + min(2, abs(p24 or 0)/10.0)
        scored.append((score, direction, sl, tp, note, it, votes, trend))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [z for z in scored][:top_n]
    if not top:
        return "⚠️ Зараз сильних підтверджених сигналів немає."

    for _, direction, sl, tp, note, it, votes, trend in top:
        sym = (it.get("symbol") or "").upper()   # CG base (наприклад 'BTC')
        px = float(it.get("current_price"))
        p24 = it.get("price_change_percentage_24h") or 0.0

        # спробувати поставити ордер (якщо включено)
        trade_res = await maybe_trade(sym, direction, px, sl, tp)

        text_lines.append(
            f"• {sym}: *{direction}* @ {px}\n"
            f"  SL: `{sl}` ({sl_pct:.2f}%) · TP: `{tp}` ({tp_pct:.2f}%) · 24h: {p24:.2f}% · votes={votes} · trend={'UP' if trend>0 else 'DOWN' if trend<0 else 'FLAT'}\n"
            f"  {note}\n"
            f"  🤖 {trade_res}\n"
        )

    return "📈 *Сильні сигнали:*\n\n" + "\n".join(text_lines)

# ======================= Commands =======================
KB = ReplyKeyboardMarkup(
    [["/signals", "/auto_on 15"], ["/auto_off", "/status"], ["/set_risk 1 2"]],
    resize_keyboard=True
)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = STATE.setdefault(chat_id, {"auto_on": False, "every": 15})
    st.setdefault("sl_pct", SL_PCT_DEFAULT)
    st.setdefault("tp_pct", TP_PCT_DEFAULT)

    await update.message.reply_text(
        "👋 Готовий!\n\nКоманди:\n"
        "• /signals — сканувати зараз (+автотрейд, якщо включений)\n"
        "• /auto_on 15 — автопуш кожні N хв (5–120)\n"
        "• /auto_off — вимкнути автопуш\n"
        "• /set_risk <SL%> <TP%> — напр.: /set_risk 1 2\n"
        "• /status — стан",
        reply_markup=KB
    )

async def set_risk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if len(context.args) < 2:
        await update.message.reply_text("Використання: `/set_risk 1.0 2.0` (SL% TP%)", parse_mode=ParseMode.MARKDOWN)
        return
    try:
        sl = float(context.args[0]); tp = float(context.args[1])
        if sl <= 0 or tp <= 0 or sl > 50 or tp > 200: raise ValueError()
    except Exception:
        await update.message.reply_text("Помилка. Приклад: /set_risk 1 2 (допустимо: SL 0.1–50, TP 0.1–200)")
        return
    st = STATE.setdefault(chat_id, {"auto_on": False, "every": 15})
    st["sl_pct"] = sl; st["tp_pct"] = tp
    await update.message.reply_text(f"✅ Ризик встановлено: SL={sl:.2f}% · TP={tp:.2f}%")

async def signals_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    txt = await build_signals_text(top_n=3, chat_id=chat_id)
    for chunk in split_long(txt):
        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

async def auto_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = STATE.setdefault(chat_id, {"auto_on": False, "every": 15})
    minutes = 15
    if context.args:
        try:
            minutes = max(5, min(120, int(context.args[0])))
        except: pass
    st["auto_on"] = True; st["every"] = minutes

    name = f"auto_{chat_id}"
    for j in context.application.job_queue.get_jobs_by_name(name):
        j.schedule_removal()
    context.application.job_queue.run_repeating(
        auto_push_job, interval=minutes*60, first=5, name=name, data={"chat_id": chat_id}
    )
    await update.message.reply_text(f"Автопуш увімкнено: кожні {minutes} хв.", reply_markup=KB)

async def auto_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = STATE.setdefault(chat_id, {"auto_on": False, "every": 15})
    st["auto_on"] = False
    name = f"auto_{chat_id}"
    for j in context.application.job_queue.get_jobs_by_name(name):
        j.schedule_removal()
    await update.message.reply_text("Автопуш вимкнено.", reply_markup=KB)

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = STATE.setdefault(chat_id, {"auto_on": False, "every": 15})
    sl, tp = get_user_risk(chat_id)
    await update.message.reply_text(
        f"Статус: {'ON' if st['auto_on'] else 'OFF'} · кожні {st['every']} хв.\n"
        f"SL={sl:.2f}% · TP={tp:.2f}%\n"
        f"TRADE_ENABLED={'ON' if TRADE_ENABLED else 'OFF'} · SIZE={TRADE_SIZE_USDT} USDT · LEV={LEVERAGE}\n"
        f"Whitelist: {', '.join(TRADE_WHITELIST)}"
    )

# ======================= Auto-push =======================
async def auto_push_job(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    st = STATE.get(chat_id, {})
    if not st or not st.get("auto_on"): return
    try:
        txt = await build_signals_text(top_n=3, chat_id=chat_id)
        for chunk in split_long(txt):
            await context.bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Автопуш помилка: {e}")

# ======================= Utils & Main =======================
def split_long(text: str, chunk_len: int = 3500) -> List[str]:
    if not text: return [""]
    out = []
    while len(text) > chunk_len:
        out.append(text[:chunk_len]); text = text[chunk_len:]
    out.append(text); return out

def main():
    if not TELEGRAM_BOT_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN env var"); return
    print("Bot running | BASE=CoinGecko | Bybit auto-trade =", "ON" if TRADE_ENABLED else "OFF")
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("set_risk", set_risk_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))
    app.add_handler(CommandHandler("auto_on", auto_on_cmd))
    app.add_handler(CommandHandler("auto_off", auto_off_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    app.run_polling()

if __name__ == "__main__":
    main()
