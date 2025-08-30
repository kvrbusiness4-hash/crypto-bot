
import os
import asyncio
import math
import time
import datetime as dt
from typing import List, Tuple, Optional

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message

from apscheduler.schedulers.asyncio import AsyncIOScheduler

# ---- Bybit (sync SDK, запускаємо в to_thread) ----
from pybit.unified_trading import HTTP

# -------------------- UTIL -------------------- #
def env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)

def env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return bool(default)

def now_utc_iso():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

# ----------------- CONFIG (ENV) ---------------- #
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_B_TOKEN") or os.getenv("TELEGRAM_BOT")
ADMIN_ID = os.getenv("ADMIN_ID")

BYBIT_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_SEC = os.getenv("BYBIT_API_SECRET", "")

# Торгові параметри (дефолти можна міняти в Railway Variables)
SIZE_USDT = env_float("SIZE_USDT", env_float("TRADE_AMOUNT", 5.0))   # розмір угоди, USDT
LEVERAGE  = env_int("LEVERAGE", 3)                                   # плече
SL_PCT    = env_float("SL_PCT", 3.0)                                  # SL %
TP_PCT    = env_float("TP_PCT", 5.0)                                  # TP %
TRADE_ENABLED = env_bool("TRADE_ENABLED", False)

# Фільтр: TOP30 або WHITELIST (через env TRADE_WHITELIST = "BTCUSDT,ETHUSDT,...")
FILTER_MODE = os.getenv("FILTER_MODE", "TOP30").upper()
TRADE_WHITELIST = [s.strip().upper() for s in os.getenv("TRADE_WHITELIST", "").replace(" ", "").split(",") if s.strip()]

# Автоскан (за замовчуванням 15 хв), можна змінити командою /auto_on N
AUTO_ENABLED = False
AUTO_INTERVAL_MIN = 15

# Скільки «сильних» сигналів брати
MAX_STRONG_ORDERS = 2

# ----------------- AIROGRAM CORE ---------------- #
bot = Bot(TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# ----------------- BYBIT SESSION ---------------- #
_bybit = HTTP(
    testnet=False,
    api_key=BYBIT_KEY,
    api_secret=BYBIT_SEC,
)

async def bybit_to_thread(fn, *args, **kwargs):
    def _call():
        return fn(*args, **kwargs)
    return await asyncio.to_thread(_call)

# ----------------- RSI / MARKET DATA ------------- #
def calc_rsi(prices: List[float], period: int = 14) -> float:
    if len(prices) <= period:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        ch = prices[-i] - prices[-i-1]
        if ch >= 0:
            gains += ch
        else:
            losses -= ch
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - (100.0 / (1.0 + rs))

async def bybit_tickers_linear() -> List[dict]:
    """Отримати тікери USDT-перп, відсортувати за оборотом (turnover24h)"""
    res = await bybit_to_thread(_bybit.get_tickers, category="linear")
    lst = (res or {}).get("result", {}).get("list", []) or []
    # фільтруємо тільки *USDT пари
    lst = [x for x in lst if x.get("symbol","").upper().endswith("USDT")]
    # сортуємо за обсягом
    def vol(x): 
        try: return float(x.get("turnover24h","0"))
        except: return 0.0
    lst.sort(key=vol, reverse=True)
    return lst

async def bybit_kline_closes(symbol: str, interval: str = "60", limit: int = 60) -> List[float]:
    res = await bybit_to_thread(
        _bybit.get_kline,
        category="linear",
        symbol=symbol,
        interval=interval,
        limit=limit
    )
    arr = (res or {}).get("result", {}).get("list", []) or []
    # unified v5 повертає: [start,open,high,low,close,volume,turnover]
    closes = [float(it[4]) for it in arr]
    closes.reverse()  # від старих до нових
    return closes

# ----------------- SIGNALS ---------------------- #
LATEST_SIGNALS: List[dict] = []  # кеш для /signals

async def build_signals() -> List[dict]:
    """Повертає список сигналів з полями:
       symbol, side ('LONG'/'SHORT'), price, rsi, score, sl_price, tp_price"""
    # 1) вибір інструментів
    if FILTER_MODE == "WHITELIST" and TRADE_WHITELIST:
        tickers = []
        all_tickers = await bybit_tickers_linear()
        m = {t["symbol"].upper(): t for t in all_tickers}
        for s in TRADE_WHITELIST:
            if s in m:
                tickers.append(m[s])
    else:
        tickers = (await bybit_tickers_linear())[:30]

    signals = []
    for t in tickers:
        symbol = t.get("symbol","")
        try:
            closes = await bybit_kline_closes(symbol, interval="60", limit=60)
            if len(closes) < 20:
                continue
            last = float(closes[-1])
            rsi = calc_rsi(closes, 14)
            # напр. тренду — по простій ковзній:
            ma20 = sum(closes[-20:]) / 20.0
            trend_down = last < ma20
            trend_up   = last > ma20

            # "сила" сигналу: відхилення від 50 (чим далі — тим сильніше)
            strength = abs(rsi - 50.0)

            if rsi <= 40 and trend_up:
                side = "LONG"
            elif rsi >= 60 and trend_down:
                side = "SHORT"
            else:
                continue

            # SL/TP у цінах
            if side == "LONG":
                sl_price = last * (1.0 - SL_PCT/100.0)
                tp_price = last * (1.0 + TP_PCT/100.0)
            else:
                sl_price = last * (1.0 + SL_PCT/100.0)
                tp_price = last * (1.0 - TP_PCT/100.0)

            signals.append({
                "symbol": symbol,
                "side": side,
                "price": last,
                "rsi": rsi,
                "trend": "UP" if trend_up else "DOWN",
                "score": round(strength/10.0, 2),  # косметика
                "sl_price": sl_price,
                "tp_price": tp_price,
            })
        except Exception as e:
            print(f"build_signals: {symbol} error: {e}")

    # найсильніші 1–2
    signals.sort(key=lambda x: abs(x["rsi"]-50.0), reverse=True)
    return signals[:MAX_STRONG_ORDERS]

def fmt_signal(sig: dict) -> str:
    side = sig["side"]
    sym  = sig["symbol"]
    p    = sig["price"]
    r15  = sig["rsi"]
    trend= sig["trend"]
    sl   = sig["sl_price"]
    tp   = sig["tp_price"]
    return (f"• {sym}: {side} @ {p:.4f}\n"
            f"  SL {SL_PCT:.2f}% → {sl:.6f} · TP {TP_PCT:.2f}% → {tp:.6f}\n"
            f"  lev×{LEVERAGE} · size {SIZE_USDT:.1f} USDT · score {abs(r15-50)/10:.2f}\n"
            f"  RSI14={r15:.1f} | Trend={trend}")

def render_signals(signals: List[dict]) -> str:
    if not signals:
        return "⚠️ Сильних сигналів не знайдено."
    body = "\n\n".join(fmt_signal(s) for s in signals)
    hdr = "📈 Сильні сигнали (топ30)\n"
    return hdr + body

# ----------------- TRADING ---------------------- #
async def ensure_leverage(symbol: str, lev: int):
    # set_leverage для linear USDT
    try:
        resp = await bybit_to_thread(
            _bybit.set_leverage,
            category="linear",
            symbol=symbol,
            buyLeverage=lev,
            sellLeverage=lev
        )
        return resp
    except Exception as e:
        raise RuntimeError(f"set_leverage failed: {e}")

async def place_market_with_sl_tp(symbol: str, side: str, price: float, sl: float, tp: float):
    # qty = USDT * lev / price
    qty = max(0.001, SIZE_USDT * LEVERAGE / max(price, 1e-6))
    qty_str = f"{qty:.6f}".rstrip('0').rstrip('.')  # більш акуратно

    try:
        await ensure_leverage(symbol, LEVERAGE)
    except Exception as e:
        raise RuntimeError(f"Помилка встановлення плеча: {e}")

    try:
        resp = await bybit_to_thread(
            _bybit.place_order,
            category="linear",
            symbol=symbol,
            side="Buy" if side=="LONG" else "Sell",
            orderType="Market",
            qty=qty_str,
            timeInForce="IOC",
            reduceOnly=False,
            slTriggerBy="LastPrice",
            tpTriggerBy="LastPrice",
            stopLoss=f"{sl:.6f}",
            takeProfit=f"{tp:.6f}",
        )
        return resp
    except Exception as e:
        raise RuntimeError(f"Помилка ордера: {e}")

# ----------------- BACKGROUND JOB ---------------- #
LATEST_SIGNALS = []

async def scan_and_maybe_trade():
    global LATEST_SIGNALS
    try:
        LATEST_SIGNALS = await build_signals()
    except Exception as e:
        print("scan error:", e)
        LATEST_SIGNALS = []

    if not LATEST_SIGNALS:
        return

    if TRADE_ENABLED:
        text_lines = ["🤖 Автоторгівля УВІМКНЕНА · Сигнали (топ30)"]
    else:
        text_lines = ["📊 Автоскан УВІМКНЕНА · Сигнали (топ30)"]

    for sig in LATEST_SIGNALS:
        text_lines.append(fmt_signal(sig))

    # Надсилаємо у чат(и): якщо є ADMIN_ID — йому, інакше нікому (можеш додати збереження chat_id при /start)
    if ADMIN_ID:
        try:
            await bot.send_message(int(ADMIN_ID), "\n\n".join(text_lines))
        except Exception as e:
            print("send admin error:", e)

    # виконати торгівлю
    if TRADE_ENABLED:
        for sig in LATEST_SIGNALS:
            try:
                await place_market_with_sl_tp(
                    symbol=sig["symbol"],
                    side=sig["side"],
                    price=sig["price"],
                    sl=sig["sl_price"],
                    tp=sig["tp_price"]
                )
            except Exception as e:
                err = f"❌ Помилка ордера по {sig['symbol']}: {e}"
                print(err)
                if ADMIN_ID:
                    try:
                        await bot.send_message(int(ADMIN_ID), err)
                    except: ...

# ----------------- COMMANDS ---------------- #
@dp.message(Command("ping"))
async def ping_cmd(m: Message):
    await m.answer("pong ✅")

@dp.message(Command("start"))
async def start_cmd(m: Message):
    await m.answer(
        "👋 Готовий!\n\n"
        "Команди:\n"
        "• /signals — показати кеш топ30 (+автотрейд, якщо увімкнено)\n"
        "• /auto_on 15 — автоскан кожні N хв (5–120)\n"
        "• /auto_off — вимкнути автоскан\n"
        "• /trade_on — увімкнути автотрейд\n"
        "• /trade_off — вимкнути автотрейд\n"
        "• /set_size 5 — розмір угоди в USDT\n"
        "• /set_lev 3 — плече\n"
        "• /set_risk 3 5 — SL/TP у %\n"
        "• /status — стан"
    )

@dp.message(Command("status"))
async def status_cmd(m: Message):
    await m.answer(
        f"Статус: {'ON' if AUTO_ENABLED else 'OFF'} · кожні {AUTO_INTERVAL_MIN} хв.\n"
        f"SL={SL_PCT:.2f}% · TP={TP_PCT:.2f}%\n"
        f"TRADE_ENABLED={'ON' if TRADE_ENABLED else 'OFF'} · SIZE={SIZE_USDT:.2f} USDT · LEV={LEVERAGE}\n"
        f"Фільтр: {'WHITELIST' if (FILTER_MODE=='WHITELIST' and TRADE_WHITELIST) else 'TOP30'}\n"
        f"UTC: {now_utc_iso()}"
    )

@dp.message(Command("signals"))
async def signals_cmd(m: Message):
    if not LATEST_SIGNALS:
        await m.answer("Поки що немає свіжих сигналів. Зачекай, сканер оновить кеш.")
    else:
        await m.answer(render_signals(LATEST_SIGNALS))

@dp.message(Command("auto_on"))
async def auto_on_cmd(m: Message):
    global AUTO_ENABLED, AUTO_INTERVAL_MIN
    try:
        parts = m.text.strip().split()
        if len(parts) >= 2:
            n = int(parts[1])
            n = max(5, min(120, n))
            AUTO_INTERVAL_MIN = n
    except:
        pass
    AUTO_ENABLED = True
    await m.answer(f"Автоскан увімкнено: кожні {AUTO_INTERVAL_MIN} хв.")

@dp.message(Command("auto_off"))
async def auto_off_cmd(m: Message):
    global AUTO_ENABLED
    AUTO_ENABLED = False
    await m.answer("Автоскан вимкнено.")

@dp.message(Command("trade_on"))
async def trade_on_cmd(m: Message):
    global TRADE_ENABLED
    TRADE_ENABLED = True
    await m.answer("Автоторгівля: УВІМКНЕНО ✅")

@dp.message(Command("trade_off"))
async def trade_off_cmd(m: Message):
    global TRADE_ENABLED
    TRADE_ENABLED = False
    await m.answer("Автоторгівля: ВИМКНЕНО ⛔️")

@dp.message(Command("set_size"))
async def set_size_cmd(m: Message):
    global SIZE_USDT
    try:
        parts = m.text.strip().split()
        if len(parts) >= 2:
            SIZE_USDT = max(1.0, float(parts[1]))
            await m.answer(f"Розмір угоди встановлено: {SIZE_USDT:.2f} USDT")
        else:
            await m.answer("Використання: /set_size 5")
    except Exception as e:
        await m.answer(f"Помилка: {e}")

@dp.message(Command("set_lev"))
async def set_lev_cmd(m: Message):
    global LEVERAGE
    try:
        parts = m.text.strip().split()
        if len(parts) >= 2:
            LEVERAGE = max(1, int(float(parts[1])))
            await m.answer(f"Плече встановлено: x{LEVERAGE}")
        else:
            await m.answer("Використання: /set_lev 3")
    except Exception as e:
        await m.answer(f"Помилка: {e}")

@dp.message(Command("set_risk"))
async def set_risk_cmd(m: Message):
    global SL_PCT, TP_PCT
    try:
        parts = m.text.strip().split()
        if len(parts) >= 3:
            SL_PCT = max(0.2, float(parts[1]))
            TP_PCT = max(0.2, float(parts[2]))
            await m.answer(f"✅ Ризик встановлено: SL={SL_PCT:.2f}% · TP={TP_PCT:.2f}%")
        else:
            await m.answer("Використання: /set_risk <SL%> <TP%>  (наприклад: /set_risk 3 5)")
    except Exception as e:
        await m.answer(f"Помилка: {e}")

# ----------------- SCHEDULER ---------------- #
scheduler = AsyncIOScheduler(timezone="UTC")

async def scheduler_loop():
    # джоба скану — лише коли AUTO_ENABLED
    async def job_wrap():
        if AUTO_ENABLED:
            await scan_and_maybe_trade()
    scheduler.add_job(job_wrap, "interval",
                      minutes=1,  # частота перевірки прапору; сам скан стартує коли AUTO_ENABLED
                      coalesce=True,
                      max_instances=1,
                      misfire_grace_time=20)
    scheduler.start()

# ----------------- MAIN --------------------- #
async def main():
    print("Starting bot…")
    await scheduler_loop()
    # одразу зробимо перший скан, але без торгів (щоб був кеш для /signals)
    try:
        sigs = await build_signals()
        LATEST_SIGNALS.clear()
        LATEST_SIGNALS.extend(sigs)
    except Exception as e:
        print("initial scan error:", e)

    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Bot stopped.")
