import os, hmac, hashlib, time, json, math, asyncio, logging
from datetime import datetime, timezone
from urllib.parse import urlencode

import aiohttp
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, AIORateLimiter, CommandHandler, ContextTypes, JobQueue
)

# ---------- CONFIG from ENV ----------

BOT_TOKEN          = os.getenv("TELEGRAM_BOT_TOKEN", "")
ADMIN_ID           = int(os.getenv("ADMIN_ID", "0"))

BYBIT_API_KEY      = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET   = os.getenv("BYBIT_API_SECRET", "")
BYBIT_BASE_URL     = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")

# trading params (editable командою)
DEFAULT_SIZE_USDT  = float(os.getenv("SIZE_USDT", "5"))       # 5 USDT
DEFAULT_LEVERAGE   = int(os.getenv("LEVERAGE", "3"))          # x3
DEFAULT_SL_PCT     = float(os.getenv("SL_PCT", "3"))          # 3%
DEFAULT_TP_PCT     = float(os.getenv("TP_PCT", "5"))          # 5%
TRADE_ENABLED_DEF  = os.getenv("TRADE_ENABLED", "0") in ("1", "true", "True")

# scan
AUTO_MINUTES_DEF   = int(os.getenv("HEARTBEAT_MINUTES", "15"))  # інтервал автоскану

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
L = logging.getLogger("bot")

# ---------- GLOBAL STATE (runtime) ----------
state = {
    "size_usdt":  DEFAULT_SIZE_USDT,
    "leverage":   DEFAULT_LEVERAGE,
    "sl_pct":     DEFAULT_SL_PCT,
    "tp_pct":     DEFAULT_TP_PCT,
    "trade_on":   TRADE_ENABLED_DEF,
    "auto_job":   None,
    "auto_mins":  AUTO_MINUTES_DEF,
    "last_scan":  None,
    "filter":     "TOP30",   # поки один режим
}

# ---------- BYBIT HELPERS ----------
def _ts_ms() -> str:
    return str(int(time.time() * 1000))

def _sign(payload: str) -> str:
    return hmac.new(BYBIT_API_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

async def bybit_public(session: aiohttp.ClientSession, path: str, params: dict = None):
    url = f"{BYBIT_BASE_URL}{path}"
    if params:
        url += "?" + urlencode(params, doseq=True)
    async with session.get(url, timeout=30) as r:
        # деякі помилки Bybit віддають text/html -> страхуємося
        txt = await r.text()
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            raise RuntimeError(f"Bybit public non-JSON: {r.status} {txt[:120]}")

async def bybit_private(session: aiohttp.ClientSession, path: str, method: str = "POST", body: dict = None, query: dict = None):
    if not (BYBIT_API_KEY and BYBIT_API_SECRET):
        raise RuntimeError("Bybit keys are not set")

    if body is None:
        body = {}
    if query is None:
        query = {}

    t = _ts_ms()
    recv = "20000"
    body_str = json.dumps(body) if body else ""
    qs = urlencode(query) if query else ""
    prehash = t + BYBIT_API_KEY + recv + body_str
    sign = _sign(prehash)

    headers = {
        "X-BAPI-API-KEY": BYBIT_API_KEY,
        "X-BAPI-SIGN": sign,
        "X-BAPI-TIMESTAMP": t,
        "X-BAPI-RECV-WINDOW": recv,
        "Content-Type": "application/json",
    }

    url = f"{BYBIT_BASE_URL}{path}"
    if qs:
        url += "?" + qs

    if method == "POST":
        async with session.post(url, data=body_str, headers=headers, timeout=30) as r:
            txt = await r.text()
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                raise RuntimeError(f"Bybit private non-JSON: {r.status} {txt[:120]}")
    else:
        async with session.get(url, headers=headers, timeout=30) as r:
            txt = await r.text()
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                raise RuntimeError(f"Bybit private non-JSON: {r.status} {txt[:120]}")

async def ensure_leverage(session: aiohttp.ClientSession, symbol: str, lev: int):
    # category=linear, mode=REGULAR
    body = {"category": "linear", "symbol": symbol, "buyLeverage": str(lev), "sellLeverage": str(lev)}
    res = await bybit_private(session, "/v5/position/set-leverage", "POST", body=body)
    if str(res.get("retCode")) != "0":
        raise RuntimeError(f"set-leverage failed: {res}")

async def place_market_order(session: aiohttp.ClientSession, symbol: str, side: str, qty: str, sl_px: float, tp_px: float):
    # timeInForce GTC; takeProfit/stopLoss – у лінійних
    body = {
        "category": "linear",
        "symbol": symbol,
        "side": side,                # "Buy" / "Sell"
        "orderType": "Market",
        "qty": qty,
        "timeInForce": "GTC",
        "takeProfit": f"{tp_px:.6f}",
        "stopLoss": f"{sl_px:.6f}",
        "tpslMode": "Full",
    }
    res = await bybit_private(session, "/v5/order/create", "POST", body=body)
    return res

# ---------- ANALYTICS ----------
def pick_strong_from_tickers(tickers: list):
    """
    Дуже легкий фільтр: беремо пари з USDT з топ-кап (Bybit вже дає популярні),
    рахуємо «силу» як |price24hPcnt| + обсяг/вага.
    Віддаємо 1-2 найсильніших із TOP30.
    """
    # лишаємо лише *USDT
    rows = [t for t in tickers if t.get("symbol","").endswith("USDT")]
    # топ30 за turnover24h (обʼєм)
    rows.sort(key=lambda x: float(x.get("turnover24h","0") or 0.0), reverse=True)
    rows = rows[:30]

    scored = []
    for t in rows:
        try:
            ch = abs(float(t.get("price24hPcnt","0")))*100.0  # у %
            vol = float(t.get("turnover24h","0"))
            score = ch + math.log10(vol+1)
            # напрям: якщо 24hPcnt < 0 — short, >0 — long
            side = "Sell" if float(t.get("price24hPcnt","0")) < 0 else "Buy"
            last = float(t.get("lastPrice","0"))
            scored.append((score, t["symbol"], side, last))
        except:
            continue
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:2]  # 1-2 ордери

# ---------- SCAN + (optional) TRADE ----------
async def do_scan_and_maybe_trade(app: Application):
    state["last_scan"] = datetime.now(timezone.utc)
    async with aiohttp.ClientSession() as s:
        try:
            # Візьмемо ринок лінійних фʼючерсів
            data = await bybit_public(s, "/v5/market/tickers", {"category":"linear"})
        except Exception as e:
            L.error(f"initial scan error: {e}")
            return

        if str(data.get("retCode")) != "0":
            L.warning(f"tickers ret !=0: {data}")
            return

        tickers = data.get("result", {}).get("list", []) or []
        picks = pick_strong_from_tickers(tickers)

        # Розсилка сигналів у бот (лише один чат — адмін)
        if picks:
            msg_lines = ["📈 Сильні сигнали (топ30)"]
            for sc, sym, side, price in picks:
                sl = price * (1 - state["sl_pct"]/100) if side=="Buy" else price * (1 + state["sl_pct"]/100)
                tp = price * (1 + state["tp_pct"]/100) if side=="Buy" else price * (1 - state["tp_pct"]/100)
                msg_lines.append(f"• {sym}: *{('LONG' if side=='Buy' else 'SHORT')}* @ {price:.4f} "
                                 f"SL {state['sl_pct']:.2f}% → {sl:.6f} · TP {state['tp_pct']:.2f}% → {tp:.6f}\n"
                                 f"lev×{state['leverage']} · size {state['size_usdt']:.2f} USDT · score {sc:.2f}")
            try:
                await app.bot.send_message(chat_id=ADMIN_ID, text="\n".join(msg_lines), parse_mode=ParseMode.MARKDOWN)
            except Exception:
                pass

        # Автоторгівля
        if state["trade_on"] and picks and BYBIT_API_KEY and BYBIT_API_SECRET:
            for _, sym, side, price in picks:
                # розрахунок кількості: (size_usdt * lev)/price
                notional = state["size_usdt"] * state["leverage"]
                qty = max(0.001, round(notional / price, 3))  # округлимо до 0.001
                sl = price * (1 - state["sl_pct"]/100) if side=="Buy" else price * (1 + state["sl_pct"]/100)
                tp = price * (1 + state["tp_pct"]/100) if side=="Buy" else price * (1 - state["tp_pct"]/100)
                try:
                    async with aiohttp.ClientSession() as s2:
                        await ensure_leverage(s2, sym, state["leverage"])
                        res = await place_market_order(s2, sym, side, str(qty), sl, tp)
                    ok = str(res.get("retCode")) == "0"
                    text = "✅ Ордер виконано" if ok else f"❌ Помилка ордера: {res.get('retCode')}, {res.get('retMsg')}"
                except Exception as e:
                    text = f"❌ Помилка ордера: {e}"
                try:
                    await app.bot.send_message(chat_id=ADMIN_ID, text=f"{text}\n{sym} {side} qty={qty}")
                except Exception:
                    pass

# ---------- COMMANDS ----------
def user_ok(update: Update) -> bool:
    return (ADMIN_ID == 0) or (update.effective_user and update.effective_user.id == ADMIN_ID)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not user_ok(update): return
    await update.message.reply_text(
        "👋 Готовий!\n\nКоманди:\n"
        "/signals — сканувати зараз\n"
        "/auto_on 15 — автоскан кожні N хв (5–120)\n"
        "/auto_off — вимкнути автоскан\n"
        "/trade_on — увімкнути автоторгівлю\n"
        "/trade_off — вимкнути автоторгівлю\n"
        "/set_size 5 — сума угоди, USDT\n"
        "/set_lev 3 — плече\n"
        "/set_risk 3 5 — SL/TP у %\n"
        "/status — стан"
    )

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not user_ok(update): return
    on = "ON" if state["auto_job"] else "OFF"
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    txt = (f"Статус: {on} · кожні {state['auto_mins']} хв.\n"
           f"SL={state['sl_pct']:.2f}% · TP={state['tp_pct']:.2f}%\n"
           f"TRADE_ENABLED={'ON' if state['trade_on'] else 'OFF'} · SIZE={state['size_usdt']:.2f} USDT · LEV={state['leverage']}\n"
           f"Фільтр: {state['filter']}\nUTC: {utc}")
    await update.message.reply_text(txt)

async def set_size_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not user_ok(update): return
    try:
        v = float(context.args[0])
        if v <= 0: raise ValueError()
        state["size_usdt"] = v
        await update.message.reply_text(f"✅ Розмір угоди встановлено: {v:.2f} USDT")
    except:
        await update.message.reply_text("Формат: /set_size 5")

async def set_lev_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not user_ok(update): return
    try:
        lev = int(context.args[0])
        if lev < 1 or lev > 50: raise ValueError()
        state["leverage"] = lev
        await update.message.reply_text(f"✅ Плече встановлено: x{lev}")
    except:
        await update.message.reply_text("Формат: /set_lev 3  (1–50)")

async def set_risk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not user_ok(update): return
    try:
        sl = float(context.args[0]); tp = float(context.args[1])
        if sl <= 0 or tp <= 0: raise ValueError()
        state["sl_pct"] = sl; state["tp_pct"] = tp
        await update.message.reply_text(f"✅ Ризик: SL={sl:.2f}% · TP={tp:.2f}%")
    except:
        await update.message.reply_text("Формат: /set_risk 3 5")

async def signals_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not user_ok(update): return
    await update.message.reply_text("🔎 Сканую ринок…")
    await do_scan_and_maybe_trade(context.application)

async def auto_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not user_ok(update): return
    try:
        mins = int(context.args[0]) if context.args else state["auto_mins"]
        mins = max(5, min(120, mins))
        state["auto_mins"] = mins
    except:
        pass

    if state["auto_job"]:
        state["auto_job"].schedule_removal()
        state["auto_job"] = None

    job = context.job_queue.run_repeating(lambda ctx: asyncio.create_task(do_scan_and_maybe_trade(ctx.application)),
                                          interval=state["auto_mins"]*60, first=1)
    state["auto_job"] = job
    await update.message.reply_text(f"Автоскан увімкнено: кожні {state['auto_mins']} хв.")

async def auto_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not user_ok(update): return
    if state["auto_job"]:
        state["auto_job"].schedule_removal()
        state["auto_job"] = None
    await update.message.reply_text("Автоскан вимкнено.")

async def trade_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not user_ok(update): return
    state["trade_on"] = True
    await update.message.reply_text("Автоторгівля: УВІМКНЕНО ✅")

async def trade_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not user_ok(update): return
    state["trade_on"] = False
    await update.message.reply_text("Автоторгівля: ВИМКНЕНО ⛔️")

# ---------- MAIN ----------
def build_app() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty")

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .rate_limiter(AIORateLimiter())  # <— потребує extras у requirements
        .build()
    )

    app.add_handler(CommandHandler("start",      start_cmd))
    app.add_handler(CommandHandler("status",     status_cmd))
    app.add_handler(CommandHandler("signals",    signals_cmd))
    app.add_handler(CommandHandler("auto_on",    auto_on_cmd))
    app.add_handler(CommandHandler("auto_off",   auto_off_cmd))
    app.add_handler(CommandHandler("trade_on",   trade_on_cmd))
    app.add_handler(CommandHandler("trade_off",  trade_off_cmd))
    app.add_handler(CommandHandler("set_size",   set_size_cmd))
    app.add_handler(CommandHandler("set_lev",    set_lev_cmd))
    app.add_handler(CommandHandler("set_risk",   set_risk_cmd))
    return app

if __name__ == "__main__":
    L.info("Starting bot…")
    app = build_app()

    # якщо є job_queue
    if app.job_queue:
        app.job_queue.run_repeating(lambda ctx: L.info("heartbeat"), interval=600, first=30)

    app.run_polling(close_loop=False)
