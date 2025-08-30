import os, time, hmac, hashlib, json, math, asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple, Optional

import aiohttp
from aiohttp import ClientResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes, AIORateLimiter
)

UTC = timezone.utc

# ======== ENV ========
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))

BYBIT_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_BASE = os.getenv("BYBIT_BASE", "https://api.bybit.com").rstrip("/")

# runtime config (можна міняти з ТГ)
SIZE_USDT = float(os.getenv("SIZE_USDT", "5"))
LEVERAGE = int(os.getenv("LEVERAGE", "3"))
SL_PCT = float(os.getenv("SL_PCT", "3"))       # 3 -> 3%
TP_PCT = float(os.getenv("TP_PCT", "5"))       # 5 -> 5%
STRONG_VOTE = float(os.getenv("STRONG_VOTE", "2.2"))  # поріг сили
TRADE_ENABLED = os.getenv("TRADE_ENABLED", "1") in ("1","true","True","on","ON")
TRADE_WHITELIST = [s.strip().upper() for s in os.getenv("TRADE_WHITELIST","").split(",") if s.strip()]

FILTER = "TOP30"  # поки фіксовано — як ти просив

# ======== Utils ========
def now_ts_ms() -> str:
    return str(int(time.time()*1000))

def sign(params: Dict[str, Any]) -> str:
    query = "&".join([f"{k}={params[k]}" for k in sorted(params.keys())])
    return hmac.new(BYBIT_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()

def fmt_pct(x: float) -> str:
    return f"{x:.2f}%"

def safe_get_env_float(val: str, default: float) -> float:
    try:
        return float(val)
    except:
        return default

# ======== HTTP with JSON guard ========
async def ensure_json(resp: ClientResponse) -> Any:
    ct = resp.headers.get("Content-Type","").lower()
    txt = await resp.text()
    if "application/json" not in ct:
        # Bybit US-block або HTML капча/заглушка
        raise RuntimeError(f"Bybit non-JSON (possible IP block): {txt[:120]}")
    try:
        return json.loads(txt)
    except Exception as e:
        raise RuntimeError(f"JSON decode error: {e}; body:{txt[:120]}")

async def bybit_get(session: aiohttp.ClientSession, path: str, params: Dict[str, Any]) -> Any:
    url = f"{BYBIT_BASE}{path}"
    async with session.get(url, params=params, timeout=20) as r:
        return await ensure_json(r)

async def bybit_private_post(session: aiohttp.ClientSession, path: str, body: Dict[str, Any]) -> Any:
    url = f"{BYBIT_BASE}{path}"
    if not BYBIT_KEY or not BYBIT_SECRET:
        raise RuntimeError("BYBIT API keys missing")
    common = {
        "api_key": BYBIT_KEY,
        "timestamp": now_ts_ms(),
        "recv_window": "5000"
    }
    allp = {**common, **body}
    allp["sign"] = sign(allp)
    headers = {"Content-Type":"application/x-www-form-urlencoded"}
    async with session.post(url, data=allp, headers=headers, timeout=20) as r:
        return await ensure_json(r)

# ======== Market/Indicators ========
async def fetch_top30_linear(session: aiohttp.ClientSession) -> List[Dict[str,Any]]:
    """V5 tickers, category=linear; повертаємо топ-30 за 24h turnover"""
    data = await bybit_get(session, "/v5/market/tickers", {"category":"linear"})
    lst = data.get("result",{}).get("list",[])
    # Сортуємо за 24h turnover (quoteVolume(USDT) може називатись "turnover24h")
    def vol(x):
        try:
            return float(x.get("turnover24h","0"))
        except: return 0.0
    lst.sort(key=vol, reverse=True)
    return lst[:30]

async def fetch_ohlc(session: aiohttp.ClientSession, symbol: str, interval: str = "15") -> List[List[float]]:
    """V5 kline linear"""
    params = {"category":"linear", "symbol":symbol, "interval":interval, "limit":"200"}
    data = await bybit_get(session, "/v5/market/kline", params)
    kl = data.get("result",{}).get("list",[])
    # list entries: [start, open, high, low, close, volume, turnover]
    # Вертаю у зручному форматі float close.
    out=[]
    for row in kl[::-1]:
        try:
            out.append(float(row[4]))
        except: pass
    return out

def rsi(values: List[float], period: int=14) -> float:
    if len(values) < period+1: return 50.0
    gains=0.0; losses=0.0
    for i in range(-period,0):
        diff = values[i] - values[i-1]
        if diff>0: gains += diff
        else: losses -= diff
    if losses==0: return 70.0
    rs = gains/losses
    return 100 - (100/(1+rs))

def trend_score(r15: float, r30: float, r60: float) -> float:
    # простий скоринг: чим нижче RSI, тим сильніший short; чим вище — long
    # сила = |(rsi-50)| сумарно по 3 таймфреймам, масштаб 0..3.5
    s = (abs(r15-50)+abs(r30-50)+abs(r60-50))/30
    return round(s,2)

def side_from_rsi(r15: float, r30: float, r60: float) -> str:
    avg = (r15+r30+r60)/3
    return "LONG" if avg<50 else "SHORT"  # якщо перепроданість — LONG; перекупленість — SHORT

# ======== Trading ========
async def ensure_leverage(session: aiohttp.ClientSession, symbol: str, lev: int) -> None:
    body = {"category":"linear", "symbol":symbol, "buyLeverage":str(lev), "sellLeverage":str(lev)}
    await bybit_private_post(session, "/v5/position/set-leverage", body)

async def place_order(session: aiohttp.ClientSession, symbol: str, side: str, qty: float,
                      sl_price: float, tp_price: float) -> Tuple[bool,str]:
    body = {
        "category":"linear",
        "symbol":symbol,
        "side": side,                # BUY/LONG або SELL/SHORT у V5: side=Buy/Sell + positionIdx?
        "orderType":"Market",
        "qty": str(qty),
        "timeInForce":"IOC",
        "reduceOnly":"false",
        "tpTriggerBy":"LastPrice",
        "slTriggerBy":"LastPrice",
        "takeProfit": str(tp_price),
        "stopLoss": str(sl_price)
    }
    # V5 очікує side = Buy/Sell
    body["side"] = "Buy" if side.upper()=="LONG" else "Sell"
    resp = await bybit_private_post(session, "/v5/order/create", body)
    ret = resp.get("retCode", -1)
    if ret==0:
        return True, resp.get("result",{}).get("orderId","ok")
    return False, f'{ret} {resp.get("retMsg","")[:120]}'

def calc_tp_sl(entry: float, side: str, sl_pct: float, tp_pct: float) -> Tuple[float,float]:
    if side=="LONG":
        sl = entry * (1 - sl_pct/100)
        tp = entry * (1 + tp_pct/100)
    else:
        sl = entry * (1 + sl_pct/100)
        tp = entry * (1 - tp_pct/100)
    return round(sl,4), round(tp,4)

def calc_qty_usdt(entry: float, size_usdt: float, lev: int) -> float:
    # qty in coin ≈ (USDT * lev) / price
    if entry<=0: return 0.0
    return round((size_usdt*lev)/entry, 6)

# ======== Scanner ========
async def scan_market(session: aiohttp.ClientSession) -> List[Dict[str,Any]]:
    top = await fetch_top30_linear(session)
    picks=[]
    for t in top:
        sym = t.get("symbol","")
        if not sym.endswith("USDT"): 
            continue
        if TRADE_WHITELIST and sym not in TRADE_WHITELIST:
            continue
        # ціна
        try:
            last = float(t.get("lastPrice","0"))
        except:
            continue
        # індикатори
        close15 = await fetch_ohlc(session, sym, "15")
        close30 = await fetch_ohlc(session, sym, "30")
        close60 = await fetch_ohlc(session, sym, "60")
        r15 = rsi(close15,14); r30=rsi(close30,14); r60=rsi(close60,14)
        score = trend_score(r15,r30,r60)
        side = side_from_rsi(r15,r30,r60)
        picks.append({
            "symbol": sym, "price": last, "r15":r15, "r30":r30, "r60":r60,
            "score": score, "side": side
        })
    # сильні сигнали
    picks.sort(key=lambda x: x["score"], reverse=True)
    strong = [x for x in picks if x["score"]>=STRONG_VOTE]
    return strong[:2] if strong else []

# ======== Telegram Handlers ========
def status_text() -> str:
    return (f"Статус: {'ON' if AUTO_ON else 'OFF'} · кожні {AUTO_MIN} хв.\n"
            f"SL={fmt_pct(SL_PCT)} · TP={fmt_pct(TP_PCT)}\n"
            f"TRADE_ENABLED={'ON' if TRADE_ENABLED else 'OFF'} · SIZE={SIZE_USDT:.2f} USDT\n"
            f"· LEV={LEVERAGE}\nФільтр: {FILTER}\nUTC: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%SZ')}")

AUTO_ON = False
AUTO_MIN = 15
scheduler: Optional[AsyncIOScheduler] = None
app: Optional[Application] = None

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Команди:\n"
        "• /signals — сканувати зараз (+автотрейд якщо увімкнено)\n"
        f"• /auto_on {AUTO_MIN} — автопуш кожні N хв (5–120)\n"
        "• /auto_off — вимкнути автопуш\n"
        "• /set_size 5 — розмір угоди (USDT)\n"
        "• /set_lev 3 — плече\n"
        "• /set_risk 3 5 — SL/TP у %\n"
        "• /trade_on, /trade_off — вмикає/вимикає виставлення ордерів\n"
        "• /status — стан"
    )

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(status_text())

async def cmd_set_size(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global SIZE_USDT
    try:
        SIZE_USDT = float(ctx.args[0])
        await update.message.reply_text(f"✅ Розмір угоди встановлено: {SIZE_USDT:.2f} USDT")
    except:
        await update.message.reply_text("Формат: /set_size 5")

async def cmd_set_lev(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global LEVERAGE
    try:
        LEVERAGE = max(1, int(ctx.args[0]))
        await update.message.reply_text(f"✅ Плече: x{LEVERAGE}")
    except:
        await update.message.reply_text("Формат: /set_lev 3")

async def cmd_set_risk(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global SL_PCT, TP_PCT
    try:
        sl = safe_get_env_float(ctx.args[0], SL_PCT)
        tp = safe_get_env_float(ctx.args[1], TP_PCT)
        SL_PCT, TP_PCT = sl, tp
        await update.message.reply_text(f"✅ Ризик: SL={fmt_pct(SL_PCT)} · TP={fmt_pct(TP_PCT)}")
    except:
        await update.message.reply_text("Формат: /set_risk 3 5")

async def cmd_trade_on(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global TRADE_ENABLED
    TRADE_ENABLED = True
    await update.message.reply_text("Автоторгівля: УВІМКНЕНА ✅")

async def cmd_trade_off(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global TRADE_ENABLED
    TRADE_ENABLED = False
    await update.message.reply_text("Автоторгівля: ВИМКНЕНА ⛔")

async def cmd_auto_on(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global AUTO_ON, AUTO_MIN
    try:
        n = int(ctx.args[0]) if ctx.args else AUTO_MIN
        n = min(120, max(5, n))
        AUTO_MIN = n
        AUTO_ON = True
        scheduler.add_job(scan_and_maybe_trade, "interval", minutes=AUTO_MIN, next_run_time=datetime.now(UTC))
        await update.message.reply_text(f"Автоскан увімкнено: кожні {AUTO_MIN} хв.")
    except Exception as e:
        await update.message.reply_text(f"Помилка автоскану: {e}")

async def cmd_auto_off(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global AUTO_ON
    AUTO_ON = False
    for j in scheduler.get_jobs():
        j.remove()
    await update.message.reply_text("Автоскан вимкнено.")

def score_block(s: Dict[str,Any]) -> str:
    return (f"{s['symbol']}: *{s['side']}* @ {s['price']:.4f} "
            f"• score {s['score']}\n"
            f"RSI15={s['r15']:.1f} | RSI30={s['r30']:.1f} | RSI60={s['r60']:.1f}")

async def cmd_signals(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔎 Сканую ринок…")
    text=""
    async with aiohttp.ClientSession() as session:
        try:
            picks = await scan_market(session)
        except Exception as e:
            await update.message.reply_text(f"❌ Помилка сканера: {e}")
            return
        if not picks:
            await update.message.reply_text("⚠️ Сильних сигналів не знайдено.")
            return
        for p in picks:
            text += "• "+score_block(p)+"\n\n"
        await update.message.reply_text(f"📈 Сильні сигнали (топ30)\n\n{text}", parse_mode=ParseMode.MARKDOWN)

        if TRADE_ENABLED:
            results=[]
            for p in picks:
                symbol = p["symbol"]
                side = p["side"]
                price = p["price"]
                qty = calc_qty_usdt(price, SIZE_USDT, LEVERAGE)
                sl, tp = calc_tp_sl(price, side, SL_PCT, TP_PCT)
                try:
                    await ensure_leverage(session, symbol, LEVERAGE)
                    ok, msg = await place_order(session, symbol, side, qty, sl, tp)
                    if ok:
                        results.append(f"✅ Ордер {symbol} {side} qty={qty} SL={sl} TP={tp}")
                    else:
                        results.append(f"❌ {symbol}: {msg}")
                except Exception as e:
                    results.append(f"❌ {symbol}: {e}")
            await update.message.reply_text("\n".join(results))

async def scan_and_maybe_trade():
    # для JobQueue / scheduler
    try:
        dummy_update = None
        dummy_ctx = None
        # посилаємо сигнал в ADMIN чат якщо є
        async with aiohttp.ClientSession() as session:
            picks = await scan_market(session)
            if not picks:
                if app and ADMIN_ID:
                    await app.bot.send_message(ADMIN_ID, "⚠️ Сильних сигналів не знайдено.")
                return
            msg = "📈 Сильні сигнали (топ30)\n\n" + "\n\n".join(["• "+score_block(p) for p in picks])
            if app and ADMIN_ID:
                await app.bot.send_message(ADMIN_ID, msg, parse_mode=ParseMode.MARKDOWN)

            if TRADE_ENABLED:
                res=[]
                for p in picks:
                    symbol=p["symbol"]; side=p["side"]; price=p["price"]
                    qty = calc_qty_usdt(price, SIZE_USDT, LEVERAGE)
                    sl, tp = calc_tp_sl(price, side, SL_PCT, TP_PCT)
                    try:
                        await ensure_leverage(session, symbol, LEVERAGE)
                        ok, mid = await place_order(session, symbol, side, qty, sl, tp)
                        res.append(("✅" if ok else "❌")+f" {symbol} {side} qty={qty} SL={sl} TP={tp} {mid}")
                    except Exception as e:
                        res.append(f"❌ {symbol}: {e}")
                if app and ADMIN_ID:
                    await app.bot.send_message(ADMIN_ID, "\n".join(res))
    except Exception as e:
        if app and ADMIN_ID:
            await app.bot.send_message(ADMIN_ID, f"❌ Помилка автосканера: {e}")

# ======= Main ========
def main():
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    # Команди
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("set_size", cmd_set_size))
    application.add_handler(CommandHandler("set_lev", cmd_set_lev))
    application.add_handler(CommandHandler("set_risk", cmd_set_risk))
    application.add_handler(CommandHandler("trade_on", cmd_trade_on))
    application.add_handler(CommandHandler("trade_off", cmd_trade_off))
    application.add_handler(CommandHandler("signals", cmd_signals))
    application.add_handler(CommandHandler("auto_on", cmd_auto_on))
    application.add_handler(CommandHandler("auto_off", cmd_auto_off))

    # Scheduler для heartbeat чи автоскану (якщо треба)
    scheduler = AsyncIOScheduler(timezone=UTC)
    scheduler.start()
    scheduler.add_job(lambda: None, "interval", minutes=60)  # заглушка

    print("🚀 Bot is running...")
    application.run_polling(drop_pending_updates=True)
    

if __name__ == "__main__":
    main()
