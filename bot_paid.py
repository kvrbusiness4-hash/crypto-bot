import os
import asyncio
import math
import time
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from telegram import Update, BotCommand
from telegram.constants import ParseMode
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, ContextTypes
)

import aiohttp
from pybit.unified_trading import HTTP as BYBITHTTP


# ========= ENV & Defaults =========
TG_TOKEN         = os.getenv("TELEGRAM_BOT_TOKEN", "")
ADMIN_ID         = int(os.getenv("ADMIN_ID", "0"))

BYBIT_KEY        = os.getenv("BYBIT_API_KEY", "")
BYBIT_SECRET     = os.getenv("BYBIT_API_SECRET", "")
# base URL не обов'язковий; pybit сам вибере прод, але залишаю гачок
BYBIT_BASE       = os.getenv("BYBIT_BASE", "").strip() or None   # напр. "https://api.bybit.com"

# Ризик/розмір/плече (глобальні; можна змінювати командами)
SL_PCT           = float(os.getenv("SL_PCT", "3"))      # 3%
TP_PCT           = float(os.getenv("TP_PCT", "5"))      # 5%
SIZE_USDT        = float(os.getenv("SIZE_USDT", "5"))   # 5 USDT ноціонал
LEVERAGE         = int(os.getenv("LEVERAGE", "3"))      # x3
STRONG_VOTE_MIN  = float(os.getenv("STRONG_VOTE_MIN", "2.4"))  # поріг cили сигналу
TRADE_ENABLED    = os.getenv("TRADE_ENABLED", "0") in ("1","true","True","yes","on")

# Автоскан інтервал (хв)
AUTO_EVERY_MIN   = 15

# Проксі для Railway/US-IP
PROXY_ENABLED    = os.getenv("PROXY_ENABLED", "0") in ("1","true","True","yes","on")
PROXY_URL        = os.getenv("PROXY_URL", "").strip() or None

# Скільки ордерів брати з топових сигналів
ORDERS_TO_TAKE   = int(os.getenv("ORDERS_TO_TAKE", "2"))

# Константи
BYBIT_PUBLIC     = BYBIT_BASE or "https://api.bybit.com"
LINEAR_CATEGORY  = "linear"     # USDT perpetual
KLINE_INTERVAL   = "15"         # 15m для RSI
TOPN             = 30


# ========= Runtime Settings (in-memory) =========
runtime = {
    "sl_pct": SL_PCT,
    "tp_pct": TP_PCT,
    "size_usdt": SIZE_USDT,
    "lev": LEVERAGE,
    "auto_on": False,
    "auto_every_min": AUTO_EVERY_MIN,
    "trade_enabled": TRADE_ENABLED,
}

# ========= Helper: aiohttp session with (optional) proxy =========
_aiohttp_session: aiohttp.ClientSession | None = None

def _proxy_kw():
    if PROXY_ENABLED and PROXY_URL:
        return {"proxy": PROXY_URL}
    return {}

async def get_http() -> aiohttp.ClientSession:
    global _aiohttp_session
    if _aiohttp_session and not _aiohttp_session.closed:
        return _aiohttp_session
    timeout = aiohttp.ClientTimeout(total=20)
    _aiohttp_session = aiohttp.ClientSession(timeout=timeout)
    return _aiohttp_session


# ========= Bybit client (pybit) with proxies =========
def make_bybit_client() -> BYBITHTTP:
    proxies = None
    if PROXY_ENABLED and PROXY_URL:
        proxies = {"http": PROXY_URL, "https": PROXY_URL}
    client = BYBITHTTP(
        testnet=False,
        api_key=BYBIT_KEY,
        api_secret=BYBIT_SECRET,
        recv_window=5000,
        **({"proxies": proxies} if proxies else {}),
        **({"domain": BYBIT_BASE} if BYBIT_BASE else {})
    )
    return client


# ========= Math: RSI =========
def calc_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(1, period+1):
        diff = closes[-i] - closes[-i-1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period if losses != 0 else 1e-9
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)


# ========= Signals (Top30 by 24h turnover) =========
async def fetch_top30_symbols() -> list[str]:
    """
    Беремо tickers (linear), сортуємо за turnover24h (обіг), top30.
    """
    url = f"{BYBIT_PUBLIC}/v5/market/tickers?category={LINEAR_CATEGORY}"
    http = await get_http()
    async with http.get(url, **_proxy_kw()) as r:
        if r.status != 200:
            txt = await r.text()
            raise RuntimeError(f"tickers status={r.status} body={txt[:200]}")
        data = await r.json()
    rows = data.get("result", {}).get("list", []) or []
    # sort desc by turnover24h (float)
    def _key(x):
        try:
            return float(x.get("turnover24h") or 0.0)
        except:
            return 0.0
    rows.sort(key=_key, reverse=True)
    syms = [row["symbol"] for row in rows if row.get("symbol","").endswith("USDT")]
    return syms[:TOPN]


async def fetch_klines(symbol: str, limit: int = 60) -> list[float]:
    """
    15m клiни, беремо closes.
    """
    url = (f"{BYBIT_PUBLIC}/v5/market/kline"
           f"?category={LINEAR_CATEGORY}&symbol={symbol}&interval={KLINE_INTERVAL}&limit={limit}")
    http = await get_http()
    async with http.get(url, **_proxy_kw()) as r:
        if r.status != 200:
            txt = await r.text()
            raise RuntimeError(f"kline {symbol} status={r.status} body={txt[:200]}")
        data = await r.json()
    ls = data.get("result", {}).get("list", []) or []
    closes = [float(x[4]) for x in ls]  # [start,open,high,low,close,volume, ...]
    return closes[-(limit):]


async def last_price(symbol: str) -> float:
    url = f"{BYBIT_PUBLIC}/v5/market/tickers?category={LINEAR_CATEGORY}&symbol={symbol}"
    http = await get_http()
    async with http.get(url, **_proxy_kw()) as r:
        if r.status != 200:
            txt = await r.text()
            raise RuntimeError(f"price {symbol} status={r.status} body={txt[:200]}")
        data = await r.json()
    lst = data.get("result", {}).get("list", [])
    if not lst:
        raise RuntimeError("No ticker")
    return float(lst[0]["lastPrice"])


def score_signal(rsi15: float, rsi30: float, rsi60: float, trend: float) -> float:
    """
    Простий скорер (0..~3+). Чим менш/більше RSI й одностайніший тренд, тим краще.
    """
    bias = 0.0
    # перепроданість/перекупленість
    if rsi15 < 35 and rsi30 < 40 and rsi60 < 45:
        bias += 1.3
    if rsi15 > 65 and rsi30 > 60 and rsi60 > 55:
        bias += 1.3
    # тренд
    if abs(trend) > 0.003:
        bias += min(1.0, abs(trend) * 120)  # грубо масштабую trend -> 0..1
    # підтвердження таймфреймами
    align = 0
    if (rsi15<50 and rsi30<50 and rsi60<50) or (rsi15>50 and rsi30>50 and rsi60>50):
        align = 0.5
    return round(bias + align, 2)


async def build_signals() -> list[dict]:
    """
    Повертає список сигналів:
    {symbol, side, price, sl_price, tp_price, rsi15,rsi30,rsi60, trend, score}
    side вибирається за напрямком тренду/RSI.
    """
    out = []
    symbols = await fetch_top30_symbols()
    for sym in symbols:
        try:
            closes = await fetch_klines(sym, 120)
            if len(closes) < 40:
                continue
            pr = closes[-1]
            # тренд як нахил останніх N барів
            trend = (closes[-1] - closes[-20]) / closes[-20] if closes[-20] != 0 else 0.0

            # RSI на різних вікнах (беремо 15м, 30м, 60м — з однієї серії приблизно)
            rsi15 = calc_rsi(closes[-30:], 14)
            rsi30 = calc_rsi(closes[-60:], 14)
            rsi60 = calc_rsi(closes[-120:], 14) if len(closes)>=121 else rsi30

            sc = score_signal(rsi15, rsi30, rsi60, trend)

            # напрям: якщо тренд нижче 0 і RSI>50 -> short? краще: якщо тренд вниз і RSI вище 40 — short,
            # якщо тренд вверх і RSI нижче 60 — long
            if trend < 0:
                side = "Sell"  # short
                sl_price = pr * (1 + runtime["sl_pct"]/100.0)
                tp_price = pr * (1 - runtime["tp_pct"]/100.0)
            else:
                side = "Buy"   # long
                sl_price = pr * (1 - runtime["sl_pct"]/100.0)
                tp_price = pr * (1 + runtime["tp_pct"]/100.0)

            out.append({
                "symbol": sym,
                "price": pr,
                "side": side,
                "sl": round(sl_price, 6),
                "tp": round(tp_price, 6),
                "rsi15": rsi15,
                "rsi30": rsi30,
                "rsi60": rsi60,
                "trend": round(trend, 5),
                "score": sc,
            })
            await asyncio.sleep(1.1)   # поважаємо rate-limit
        except Exception as e:
            # не валимо весь цикл
            # print(f"signal err {sym}: {e}")
            await asyncio.sleep(0.3)
            continue

    # сортую за score (низхідно)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:max(ORDERS_TO_TAKE, 1)]


# ========= Trading (market + TP/SL + leverage) =========
def qty_from_usdt(price: float, usdt: float) -> float:
    if price <= 0:
        return 0.0
    # позиція на ~usdt ноціоналу
    q = usdt / price
    # округлення до 3 знаків — більшість символів приймає 0.001
    return max(0.001, float(f"{q:.3f}"))

async def place_trade(sig: dict) -> tuple[bool, str]:
    """
    Маркет вхід + встановлення плеча + TP/SL (Full)
    """
    cl = make_bybit_client()
    sym = sig["symbol"]
    side = "Buy" if sig["side"] == "Buy" else "Sell"
    pr   = sig["price"]

    # плечі на обидва боки
    try:
        cl.set_leverage(
            category=LINEAR_CATEGORY,
            symbol=sym,
            buyLeverage=str(runtime["lev"]),
            sellLeverage=str(runtime["lev"])
        )
    except Exception as e:
        return False, f"❌ set_leverage {sym}: {e}"

    qty = qty_from_usdt(pr, runtime["size_usdt"])
    try:
        resp = cl.place_order(
            category=LINEAR_CATEGORY,
            symbol=sym,
            side=side,
            orderType="Market",
            qty=str(qty),
            timeInForce="IOC",
            reduceOnly=False,
            tpSlMode="Full",
            takeProfit=f"{sig['tp']:.6f}",
            stopLoss=f"{sig['sl']:.6f}",
            positionIdx=0,      # one-way
        )
        # чекаємо відповідь від Bybit
        retCode = int(resp.get("retCode", -1))
        if retCode != 0:
            return False, f"❌ order {sym}: retCode={retCode}, msg={resp.get('retMsg')}"
        return True, f"✅ Ордер {sym} {side} • qty={qty} • SL={sig['sl']:.6f} • TP={sig['tp']:.6f}"
    except Exception as e:
        return False, f"❌ order {sym}: {e}"


# ========= Bot Handlers =========
HELP = (
    "Команди:\n"
    "/signals — просканувати зараз (топ30)\n"
    "/auto_on 15 — автоскан кожні N хв (5–120)\n"
    "/auto_off — вимкнути автоскан\n"
    "/set_risk <SL%> <TP%> — напр.: /set_risk 3 5\n"
    "/set_size <USD> — напр.: /set_size 5\n"
    "/set_lev <X> — напр.: /set_lev 3\n"
    "/trade_on — увімкнути автотрейд\n"
    "/trade_off — вимкнути автотрейд\n"
    "/status — стан\n"
)

def fmt_status() -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    return (
        f"Статус: {'ON' if runtime['auto_on'] else 'OFF'} • кожні {runtime['auto_every_min']} хв.\n"
        f"SL={runtime['sl_pct']:.2f}% • TP={runtime['tp_pct']:.2f}%\n"
        f"TRADE_ENABLED={'ON' if runtime['trade_enabled'] else 'OFF'} • "
        f"SIZE={runtime['size_usdt']:.2f} USDT • LEV={runtime['lev']}\n"
        f"Фільтр: TOP{TOPN}\n"
        f"UTC: {now}"
    )

async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user is None:
        return
    await update.message.reply_text("👋 Готовий!\n" + HELP)

async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(fmt_status())

async def set_risk_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        sl = float(ctx.args[0]); tp = float(ctx.args[1])
        runtime["sl_pct"] = max(0.1, min(sl, 50))
        runtime["tp_pct"] = max(0.1, min(tp, 100))
        await update.message.reply_text(f"✅ Ризик встановлено: SL={runtime['sl_pct']:.2f}% • TP={runtime['tp_pct']:.2f}%")
    except Exception:
        await update.message.reply_text("Формат: /set_risk 3 5")

async def set_size_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        sz = float(ctx.args[0])
        runtime["size_usdt"] = max(1.0, min(sz, 10000))
        await update.message.reply_text(f"✅ Розмір угоди встановлено: {runtime['size_usdt']:.2f} USDT")
    except Exception:
        await update.message.reply_text("Формат: /set_size 5")

async def set_lev_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        lv = int(ctx.args[0])
        runtime["lev"] = max(1, min(lv, 50))
        await update.message.reply_text(f"✅ Плече встановлено: x{runtime['lev']}")
    except Exception:
        await update.message.reply_text("Формат: /set_lev 3")

async def trade_on_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    runtime["trade_enabled"] = True
    await update.message.reply_text("🤖 Автоторгівля УВІМКНЕНА.")

async def trade_off_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    runtime["trade_enabled"] = False
    await update.message.reply_text("🛑 Автоторгівля ВИМКНЕНА.")

async def auto_on_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        mins = int(ctx.args[0]) if ctx.args else AUTO_EVERY_MIN
        mins = max(5, min(120, mins))
        runtime["auto_every_min"] = mins
        runtime["auto_on"] = True
        await update.message.reply_text(f"Автоскан увімкнено: кожні {mins} хв.")
    except Exception:
        await update.message.reply_text("Формат: /auto_on 15 (5–120)")

async def auto_off_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    runtime["auto_on"] = False
    await update.message.reply_text("Автоскан вимкнено.")

async def signals_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await scan_and_maybe_trade(update, ctx, manual=True)


# ========= Scan + Maybe trade =========
async def scan_and_maybe_trade(update: Update | None, ctx: ContextTypes.DEFAULT_TYPE | None, manual: bool = False):
    chat_id = update.effective_chat.id if update and update.effective_chat else ADMIN_ID
    try:
        sigs = await build_signals()
        if not sigs:
            txt = "⚠️ Сильних сигналів не знайдено."
            if ctx: await ctx.bot.send_message(chat_id, txt)
            return

        lines = ["📈 Сильні сигнали (топ30)"]
        for s in sigs:
            direction = "LONG" if s["side"]=="Buy" else "SHORT"
            lines.append(
                f"• {s['symbol']}: {direction} @ {s['price']:.6f} "
                f"SL {runtime['sl_pct']:.2f}% → {s['sl']:.6f} • TP {runtime['tp_pct']:.2f}% → {s['tp']:.6f}\n"
                f"lev×{runtime['lev']} • size {runtime['size_usdt']:.1f} USDT • score {s['score']}\n"
                f"RSI15={s['rsi15']} | RSI30={s['rsi30']} | RSI60={s['rsi60']} | Trend {('↑' if s['trend']>0 else '↓')}"
            )
        if ctx:
            await ctx.bot.send_message(chat_id, "\n\n".join(lines))

        if not runtime["trade_enabled"]:
            return

        # placing up to ORDERS_TO_TAKE
        for s in sigs[:ORDERS_TO_TAKE]:
            ok, msg = await place_trade(s)
            if ctx: await ctx.bot.send_message(chat_id, msg)
            await asyncio.sleep(1.2)

    except Exception as e:
        if ctx:
            await ctx.bot.send_message(chat_id, f"❌ Помилка скану: {e}")


# ========= Scheduler heartbeat =========
async def heartbeat(app: Application):
    chat_id = ADMIN_ID
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    await app.bot.send_message(chat_id, f"✅ Heartbeat · Uptime 10s · UTC {now}")


async def auto_loop(app: Application):
    # Легкий цикл: раз на N хвилин, якщо увімкнено
    while True:
        try:
            if runtime["auto_on"]:
                await scan_and_maybe_trade(None, app, manual=False)
        except Exception:
            pass
        await asyncio.sleep(runtime["auto_every_min"] * 60)


# ========= Main =========
async def main():
    if not TG_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty")
    application = (
        ApplicationBuilder()
        .token(TG_TOKEN)
        .build()
    )

    # Команди (кнопки)
    await application.bot.set_my_commands([
        BotCommand("signals", "сканувати зараз"),
        BotCommand("auto_on", "автоскан кожні N хв"),
        BotCommand("auto_off", "вимкнути автоскан"),
        BotCommand("set_risk", "встановити SL/TP у %"),
        BotCommand("set_size", "встановити розмір угоди (USDT)"),
        BotCommand("set_lev", "встановити плече (x)"),
        BotCommand("trade_on", "увімкнути автотрейд"),
        BotCommand("trade_off", "вимкнути автотрейд"),
        BotCommand("status", "стан"),
    ])

    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(CommandHandler("set_risk", set_risk_cmd))
    application.add_handler(CommandHandler("set_size", set_size_cmd))
    application.add_handler(CommandHandler("set_lev", set_lev_cmd))
    application.add_handler(CommandHandler("trade_on", trade_on_cmd))
    application.add_handler(CommandHandler("trade_off", trade_off_cmd))
    application.add_handler(CommandHandler("auto_on", auto_on_cmd))
    application.add_handler(CommandHandler("auto_off", auto_off_cmd))
    application.add_handler(CommandHandler("signals", signals_cmd))

    # Паралельний автолооп
    application.job_queue.run_once(lambda *_: None, when=0)  # прогріти JobQueue
    asyncio.create_task(auto_loop(application))

    # Heartbeat кожні 10 хв (видно, що бот живий)
    scheduler = AsyncIOScheduler()
    scheduler.add_job(lambda: heartbeat(application), "interval", seconds=600, id="heartbeat")
    scheduler.start()

    print("Starting bot…")
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    await application.idle()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        if _aiohttp_session and not _aiohttp_session.closed:
            asyncio.run(_aiohttp_session.close())
