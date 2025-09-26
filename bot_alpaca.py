# -*- coding: utf-8 -*-
import os, json, math, asyncio, time, re
from typing import Dict, Any, Tuple, List, Optional
from aiohttp import ClientSession, ClientTimeout
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes

# ========= ENV =========
TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_TOKEN") or "").strip()
ALPACA_API_KEY = (os.getenv("ALPACA_API_KEY") or "").strip()
ALPACA_API_SECRET = (os.getenv("ALPACA_API_SECRET") or "").strip()
ALPACA_BASE_URL = (os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_URL = (os.getenv("ALPACA_DATA_URL") or "https://data.alpaca.markets").rstrip("/")

ALPACA_BUDGET = float(os.getenv("ALPACA_NOTIONAL") or 50)   # бюджет у USD на одну угоду
ALPACA_TOP_N = int(os.getenv("ALPACA_TOP_N") or 2)
ALPACA_MAX_CRYPTO = int(os.getenv("ALPACA_MAX_CRYPTO") or 25)
ALPACA_MAX_STOCKS = int(os.getenv("ALPACA_MAX_STOCKS") or 50)
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC") or 300)
DEDUP_COOLDOWN_MIN = int(os.getenv("DEDUP_COOLDOWN_MIN") or 240)

STATE: Dict[int, Dict[str, Any]] = {}

MODE_PARAMS = {
    "aggressive": {"bars": ("15Min","30Min","1Hour"), "rsi_buy":55.0, "rsi_sell":45.0, "ema_fast":15, "ema_slow":30, "top_n":ALPACA_TOP_N, "tp_pct":0.015, "sl_pct":0.008},
    "scalp":      {"bars": ("5Min","15Min","1Hour"),  "rsi_buy":58.0, "rsi_sell":42.0, "ema_fast":9,  "ema_slow":21, "top_n":ALPACA_TOP_N, "tp_pct":0.010, "sl_pct":0.006},
    "default":    {"bars": ("15Min","30Min","1Hour"), "rsi_buy":56.0, "rsi_sell":44.0, "ema_fast":12, "ema_slow":26, "top_n":ALPACA_TOP_N, "tp_pct":0.012, "sl_pct":0.008},
    "swing":      {"bars": ("30Min","1Hour","1Day"),  "rsi_buy":55.0, "rsi_sell":45.0, "ema_fast":20, "ema_slow":40, "top_n":ALPACA_TOP_N, "tp_pct":0.020, "sl_pct":0.010},
    "safe":       {"bars": ("15Min","30Min","1Hour"), "rsi_buy":60.0, "rsi_sell":40.0, "ema_fast":15, "ema_slow":35, "top_n":max(1,ALPACA_TOP_N-1), "tp_pct":0.009, "sl_pct":0.006},
}

CRYPTO_USD_PAIRS = [
    "BTC/USD","ETH/USD","SOL/USD","LTC/USD","DOGE/USD","AVAX/USD","AAVE/USD","MKR/USD","DOT/USD",
    "LINK/USD","UNI/USD","PEPE/USD","XRP/USD","TRUMP/USD","CRV/USD","BCH/USD","BAT/USD","GRT/USD",
    "XTZ/USD","USDC/USD","USDT/USD","USDG/USD","YFI/USD","LDO/USD"
][:ALPACA_MAX_CRYPTO]

STOCKS_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","ADBE","CRM","ORCL","AMD","AMAT","INTC","CSCO","QCOM",
    "BAC","JPM","GS","BRK.B","V","MA","KO","PEP","MCD","NKE","SPY","QQQ","IWM","DIA","XLF","XLK","XLV","XLE","XLY","XLP",
][:ALPACA_MAX_STOCKS]

def map_tf(tf: str) -> str:
    t=(tf or "").strip()
    return "1Hour" if t.lower() in ("60min","60","1h","60мин","60мін") else t
def to_order_sym(s:str)->str: return s.replace("/","").upper()
def to_data_sym(s:str)->str:
    s=(s or "").replace(" ","").upper()
    if "/" in s: return s
    if s.endswith("USD"): return s[:-3]+"/USD"
    return s
def now_s()->float: return time.time()

RECENT_TRADES: Dict[str,float]={}
def skip_as_duplicate(market:str,sym:str,side:str)->bool:
    key=f"{market}|{to_order_sym(sym)}|{side.lower()}"
    last=RECENT_TRADES.get(key,0)
    if now_s()-last<DEDUP_COOLDOWN_MIN*60: return True
    RECENT_TRADES[key]=now_s(); return False

def _mode_conf(st): return MODE_PARAMS.get(st.get("mode") or "default", MODE_PARAMS["default"])
def stdef(chat_id:int)->Dict[str,Any]:
    st=STATE.setdefault(chat_id,{})
    st.setdefault("mode","aggressive"); st.setdefault("autotrade",False)
    st.setdefault("auto_scan",False); st.setdefault("side_mode","long")
    return st

def kb()->ReplyKeyboardMarkup:
    rows=[
        ["/aggressive","/scalp","/default"],
        ["/swing","/safe","/help"],
        ["/signals_crypto","/trade_crypto"],
        ["/signals_stocks","/trade_stocks"],
        ["/alp_on","/alp_status","/alp_off"],
        ["/auto_on","/auto_status","/auto_off"],
        ["/long_mode","/short_mode","/both_mode"],
    ]; return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def _alp_headers()->Dict[str,str]:
    return {"APCA-API-KEY-ID":ALPACA_API_KEY,"APCA-API-SECRET-KEY":ALPACA_API_SECRET,"Content-Type":"application/json"}

async def alp_get_json(path:str, params:Dict[str,Any]|None=None)->Any:
    url=f"{ALPACA_BASE_URL}{path}" if path.startswith("/v") else f"{ALPACA_DATA_URL}{path}"
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        async with s.get(url, headers=_alp_headers(), params=params) as r:
            t=await r.text()
            if r.status>=400: raise RuntimeError(f"GET {path} {r.status}: {t}")
            return json.loads(t) if t else {}

async def alp_post_json(path:str, payload:Dict[str,Any])->Any:
    url=f"{ALPACA_BASE_URL}{path}"
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        async with s.post(url, headers=_alp_headers(), data=json.dumps(payload)) as r:
            t=await r.text()
            if r.status>=400: raise RuntimeError(f"POST {path} {r.status}: {t}")
            return json.loads(t) if t else {}

async def alp_clock()->Dict[str,Any]: return await alp_get_json("/v2/clock")
async def alp_positions()->List[Dict[str,Any]]: return await alp_get_json("/v2/positions")
async def has_open_long(sym:str)->bool:
    try:
        p=await alp_get_json(f"/v2/positions/{to_order_sym(sym)}")
        return float(p.get("qty",0) or 0)>0
    except Exception: return False

# ===== DATA (bars) =====
async def get_bars_crypto(pairs:List[str], timeframe:str, limit:int=120)->Dict[str,Any]:
    tf=map_tf(timeframe); syms=",".join([to_data_sym(p) for p in pairs])
    path="/v1beta3/crypto/us/bars"; params={"symbols":syms,"timeframe":tf,"limit":str(limit),"sort":"asc"}
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        url=f"{ALPACA_DATA_URL}{path}"
        async with s.get(url, headers=_alp_headers(), params=params) as r:
            t=await r.text()
            if r.status>=400: raise RuntimeError(f"GET {url} {r.status}: {t}")
            return json.loads(t) if t else {}

async def get_bars_stocks(symbols:List[str], timeframe:str, limit:int=120)->Dict[str,Any]:
    tf=map_tf(timeframe); syms=",".join([s.upper() for s in symbols])
    path="/v2/stocks/bars"; params={"symbols":syms,"timeframe":tf,"limit":str(limit),"sort":"asc"}
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        url=f"{ALPACA_DATA_URL}{path}"
        async with s.get(url, headers=_alp_headers(), params=params) as r:
            t=await r.text()
            if r.status>=400: raise RuntimeError(f"GET {url} {r.status}: {t}")
            return json.loads(t) if t else {}

# ===== INDICATORS =====
def ema(vals:List[float], period:int)->List[float]:
    if not vals or period<=0: return []
    k=2.0/(period+1.0); out=[vals[0]]
    for v in vals[1:]: out.append(v*k + out[-1]*(1-k))
    return out

def rsi(vals:List[float], period:int)->float:
    if len(vals)<period+1: return 50.0
    g=l=0.0
    for i in range(-period,0):
        d=vals[i]-vals[i-1]
        if d>=0: g+=d
        else: l-=d
    if l==0: return 70.0
    rs=g/l
    return 100.0 - (100.0/(1+rs))

def rank_score(c15,c30,c60,rsi_buy,rsi_sell,ema_fast_p,ema_slow_p)->float:
    r1,r2,r3=rsi(c15,14),rsi(c30,14),rsi(c60,14)
    e_fast,e_slow=ema(c60,ema_fast_p),ema(c60,ema_slow_p)
    trend=(e_fast[-1]-e_slow[-1])/max(1e-9,abs(e_slow[-1])) if (e_fast and e_slow) else 0.0
    bias_long=sum(int(r>=rsi_buy) for r in (r1,r2,r3))
    bias_short=sum(int(r<=rsi_sell) for r in (r1,r2,r3))
    bias=max(bias_long,bias_short)
    return bias*100.0 + trend*50.0 - abs(50.0-r1)

def calc_sl_tp(side:str, price:float, conf:Dict[str,Any])->Tuple[Optional[float],Optional[float]]:
    tp_pct=float(conf.get("tp_pct",0.01)); sl_pct=float(conf.get("sl_pct",0.008))
    if side.lower()=="buy": return price*(1+tp_pct), price*(1-sl_pct)
    else: return price*(1-tp_pct), price*(1+sl_pct)

# ===== SCANS =====
async def scan_rank_crypto(st)->Tuple[str,List[Tuple[float,str,List[Dict[str,Any]]]]]:
    conf=_mode_conf(st); tf15,tf30,tf60=[map_tf(x) for x in conf["bars"]]
    pairs=CRYPTO_USD_PAIRS[:]; data_pairs=[to_data_sym(p) for p in pairs]
    bars15=await get_bars_crypto(data_pairs, tf15, 120)
    bars30=await get_bars_crypto(data_pairs, tf30, 120)
    bars60=await get_bars_crypto(data_pairs, tf60, 120)
    ranked=[]
    for sym in data_pairs:
        r15=(bars15.get("bars") or {}).get(sym,[])
        r30=(bars30.get("bars") or {}).get(sym,[])
        r60=(bars60.get("bars") or {}).get(sym,[])
        if not r15 or not r30 or not r60: continue
        c15=[float(x["c"]) for x in r15]; c30=[float(x["c"]) for x in r30]; c60=[float(x["c"]) for x in r60]
        ranked.append((rank_score(c15,c30,c60,conf["rsi_buy"],conf["rsi_sell"],conf["ema_fast"],conf["ema_slow"]), sym, r15))
    ranked.sort(reverse=True)
    rep=("🛰️ Сканер (крипта):\n"
         f"• Активних USD-пар: {len(data_pairs)}\n"
         f"• Використаємо для торгівлі (лімітом): {min(conf['top_n'], len(ranked))}\n"
         + (f"• Перші 25: "+", ".join([s for _,s,_ in ranked[:25]]) if ranked else "• Немає сигналів"))
    return rep, ranked

async def scan_rank_stocks(st)->Tuple[str,List[Tuple[float,str,List[Dict[str,Any]]]]]:
    conf=_mode_conf(st); tf15,tf30,tf60=[map_tf(x) for x in conf["bars"]]
    syms=STOCKS_UNIVERSE[:]
    b15=await get_bars_stocks(syms, tf15, 120)
    b30=await get_bars_stocks(syms, tf30, 120)
    b60=await get_bars_stocks(syms, tf60, 120)
    ranked=[]
    for s in syms:
        r15=(b15.get("bars") or {}).get(s,[])
        r30=(b30.get("bars") or {}).get(s,[])
        r60=(b60.get("bars") or {}).get(s,[])
        if not r15 or not r30 or not r60: continue
        c15=[float(x["c"]) for x in r15]; c30=[float(x["c"]) for x in r30]; c60=[float(x["c"]) for x in r60]
        ranked.append((rank_score(c15,c30,c60,conf["rsi_buy"],conf["rsi_sell"],conf["ema_fast"],conf["ema_slow"]), s, r15))
    ranked.sort(reverse=True)
    rep=("📡 Сканер (акції):\n"
         f"• Символів у списку: {len(syms)}\n"
         f"• Використаємо для торгівлі (лімітом): {min(conf['top_n'], len(ranked))}\n"
         + (f"• Перші 25: "+", ".join([s for _,s,_ in ranked[:25]]) if ranked else "• Немає сигналів"))
    return rep, ranked

# ===== ORDERS (qty-only) =====
def _floor_qty(x:float, dec:int=6)->float:
    if x<=0: return 0.0
    m=10**dec
    return math.floor(x*m)/m

async def place_market_buy_qty_crypto(sym:str, qty:float)->dict:
    payload={"symbol":to_order_sym(sym),"side":"buy","type":"market","time_in_force":"gtc","qty":f"{_floor_qty(qty):.6f}"}
    return await alp_post_json("/v2/orders", payload)

async def place_market_buy_qty_stock(sym:str, qty_int:int)->dict:
    payload={"symbol":to_order_sym(sym),"side":"buy","type":"market","time_in_force":"day","qty":str(int(qty_int))}
    return await alp_post_json("/v2/orders", payload)

async def get_order(order_id:str)->dict:
    return await alp_get_json(f"/v2/orders/{order_id}")

async def place_tp_sl_as_simple_sells(sym:str, filled_qty:float, tp:float|None, sl:float|None, is_crypto:bool):
    if filled_qty<=0: return
    qty = _floor_qty(filled_qty) if is_crypto else int(round(filled_qty))
    if tp is not None:
        await alp_post_json("/v2/orders", {"symbol":to_order_sym(sym),"side":"sell","type":"limit","time_in_force":"gtc","limit_price":f"{tp:.6f}","qty":f"{qty}"})
    if sl is not None:
        await alp_post_json("/v2/orders", {"symbol":to_order_sym(sym),"side":"sell","type":"stop","time_in_force":"gtc","stop_price":f"{sl:.6f}","qty":f"{qty}"})

def _parse_insufficient(msg:str)->Tuple[Optional[float],Optional[str]]:
    # returns (available, symbol) if found
    m_a=re.search(r'"available"\s*:\s*"?(?P<a>[\d\.]+)"?', msg)
    m_s=re.search(r'"symbol"\s*:\s*"(?P<s>[^"]+)"', msg)
    a=float(m_a.group("a")) if m_a else None
    s=m_s.group("s") if m_s else None
    return a, s

async def place_bracket_qty_order_crypto(sym:str, price:float, budget_usd:float, tp:float|None, sl:float|None)->Any:
    qty = _floor_qty((budget_usd/price)*0.99, 6)
    if qty<=0: raise RuntimeError("budget too small for qty")
    try:
        buy=await place_market_buy_qty_crypto(sym, qty)
    except RuntimeError as e:
        msg=str(e)
        if "insufficient balance" in msg:
            available, sym_code=_parse_insufficient(msg)
            if (sym_code or "").upper()=="USD" and (available is not None):
                qty=_floor_qty((available/price)*0.99, 6)
                buy=await place_market_buy_qty_crypto(sym, qty)
            else:
                raise
        else:
            raise

    order_id=buy.get("id",""); filled=0.0
    for _ in range(12):
        od=await get_order(order_id)
        st=od.get("status")
        if st in ("filled","partially_filled"):
            filled=float(od.get("filled_qty") or 0)
            if st=="filled": break
        await asyncio.sleep(0.7)
    await place_tp_sl_as_simple_sells(sym, filled, tp, sl, is_crypto=True)
    return buy

async def place_bracket_qty_order_stock(sym:str, price:float, budget_usd:float, tp:float|None, sl:float|None)->Any:
    qty_int = int(budget_usd // max(0.01, price))  # тільки цілі
    if qty_int<=0: raise RuntimeError("budget too small for 1 share")
    buy=await place_market_buy_qty_stock(sym, qty_int)
    order_id=buy.get("id",""); filled=0.0
    for _ in range(12):
        od=await get_order(order_id)
        st=od.get("status")
        if st in ("filled","partially_filled"):
            filled=float(od.get("filled_qty") or 0)
            if st=="filled": break
        await asyncio.sleep(0.7)
    await place_tp_sl_as_simple_sells(sym, filled, tp, sl, is_crypto=False)
    return buy

# ===== COMMANDS =====
async def start(u:Update,c:ContextTypes.DEFAULT_TYPE):
    stdef(u.effective_chat.id)
    await u.message.reply_text(
        "👋 Алпака-бот готовий.\n"
        "Крипта 24/7; акції — тільки в години ринку. Сканер/автотрейд може працювати у фоні.\n"
        "Увімкнути автотрейд: /alp_on · Зупинити: /alp_off · Стан: /alp_status\n"
        "Фоновий автоскан: /auto_on · /auto_off · /auto_status",
        reply_markup=kb()
    )

async def help_cmd(u:Update,c:ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "/signals_crypto — показати топ-N, а якщо Autotrade=ON — купити\n"
        "/trade_crypto — одразу купити топ-N\n"
        "/signals_stocks — топ-N акцій\n"
        "/trade_stocks — одразу купити топ-N акцій\n"
        "/alp_on /alp_off /alp_status — автотрейд\n"
        "/auto_on /auto_off /auto_status — автоскан\n"
        "/long_mode /short_mode /both_mode — напрям (short для крипти ігнорується)\n"
        "/aggressive /scalp /default /swing /safe — профілі",
        reply_markup=kb()
    )

async def set_mode(u:Update,c:ContextTypes.DEFAULT_TYPE,mode:str):
    st=stdef(u.effective_chat.id); st["mode"]=mode
    await u.message.reply_text(f"Режим встановлено: {mode.upper()}")

async def long_mode(u, c): stdef(u.effective_chat.id)["side_mode"]="long"; await u.message.reply_text("Режим входів: LONG")
async def short_mode(u, c): stdef(u.effective_chat.id)["side_mode"]="short"; await u.message.reply_text("Режим входів: SHORT (крипта ігнорує)")
async def both_mode(u, c): stdef(u.effective_chat.id)["side_mode"]="both"; await u.message.reply_text("Режим входів: BOTH (крипта все одно LONG)")

async def alp_on(u:Update,c:ContextTypes.DEFAULT_TYPE): stdef(u.effective_chat.id)["autotrade"]=True; await u.message.reply_text("✅ Alpaca AUTOTRADE: ON")
async def alp_off(u:Update,c:ContextTypes.DEFAULT_TYPE): stdef(u.effective_chat.id)["autotrade"]=False; await u.message.reply_text("⛔ Alpaca AUTOTRADE: OFF")

async def alp_status(u:Update,c:ContextTypes.DEFAULT_TYPE):
    try:
        acc=await alp_get_json("/v2/account"); st=stdef(u.effective_chat.id)
        txt=("📦 Alpaca:\n"
             f"• status={acc.get('status','UNKNOWN')}\n"
             f"• cash=${float(acc.get('cash',0)):.2f}\n"
             f"• buying_power=${float(acc.get('buying_power',0)):.2f}\n"
             f"• equity=${float(acc.get('equity',0)):.2f}\n"
             f"Mode={st.get('mode','default')} · Autotrade={'ON' if st.get('autotrade') else 'OFF'} · "
             f"AutoScan={'ON' if st.get('auto_scan') else 'OFF'} · Side={st.get('side_mode','long')} · "
             f"Budget=${ALPACA_BUDGET:.2f}")
        await u.message.reply_text(txt)
    except Exception as e:
        await u.message.reply_text(f"🔴 alp_status error: {e}")

# --- CRYPTO ---
async def signals_crypto(u:Update,c:ContextTypes.DEFAULT_TYPE):
    st=stdef(u.effective_chat.id)
    try:
        report, ranked=await scan_rank_crypto(st)
        await u.message.reply_text(report)
        if not st.get("autotrade") or not ranked: return
        picks=ranked[:_mode_conf(st)["top_n"]]
        for _, sym, arr in picks:
            side="buy"; px=float(arr[-1]["c"]); conf=_mode_conf(st); tp,sl=calc_sl_tp(side,px,conf)
            if await has_open_long(sym): await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}"); continue
            if skip_as_duplicate("CRYPTO",sym,side): await u.message.reply_text(f"⚪ SKIP (дубль): {sym} {side.upper()}"); continue
            try:
                await place_bracket_qty_order_crypto(sym, px, ALPACA_BUDGET, tp, sl)
                await u.message.reply_text(f"🟢 ORDER OK: {sym} BUY ≈${ALPACA_BUDGET:.2f}\nTP:{(tp or 0):.6f} SL:{(sl or 0):.6f}")
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")
    except Exception as e:
        await u.message.reply_text(f"🔴 signals_crypto error: {e}")

async def trade_crypto(u:Update,c:ContextTypes.DEFAULT_TYPE):
    st=stdef(u.effective_chat.id)
    try:
        _, ranked=await scan_rank_crypto(st)
        if not ranked: await u.message.reply_text("⚠️ Немає сигналів"); return
        picks=ranked[:_mode_conf(st)["top_n"]]
        for _, sym, arr in picks:
            side="buy"; px=float(arr[-1]["c"]); conf=_mode_conf(st); tp,sl=calc_sl_tp(side,px,conf)
            if await has_open_long(sym): await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}"); continue
            if skip_as_duplicate("CRYPTO",sym,side): await u.message.reply_text(f"⚪ SKIP (дубль): {sym} {side.upper()}"); continue
            try:
                await place_bracket_qty_order_crypto(sym, px, ALPACA_BUDGET, tp, sl)
                await u.message.reply_text(f"🟢 ORDER OK: {sym} BUY ≈${ALPACA_BUDGET:.2f}\nTP:{(tp or 0):.6f} SL:{(sl or 0):.6f}")
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")
    except Exception as e:
        await u.message.reply_text(f"🔴 trade_crypto error: {e}")

# --- STOCKS ---
async def signals_stocks(u:Update,c:ContextTypes.DEFAULT_TYPE):
    st=stdef(u.effective_chat.id)
    try:
        report, ranked=await scan_rank_stocks(st); await u.message.reply_text(report)
        if not st.get("autotrade") or not ranked: return
        try: clk=await alp_clock(); market_open=bool(clk.get("is_open"))
        except Exception: market_open=True
        if not market_open: await u.message.reply_text("⏸ Ринок акцій закритий — ордери не виставляю."); return
        picks=ranked[:_mode_conf(st)["top_n"]]
        for _, sym, arr in picks:
            side="buy"; px=float(arr[-1]["c"]); conf=_mode_conf(st); tp,sl=calc_sl_tp(side,px,conf)
            if await has_open_long(sym): await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}"); continue
            if skip_as_duplicate("STOCK",sym,side): await u.message.reply_text(f"⚪ SKIP (дубль): {sym} {side.upper()}"); continue
            try:
                await place_bracket_qty_order_stock(sym, px, ALPACA_BUDGET, tp, sl)
                await u.message.reply_text(f"🟢 ORDER OK: {sym} BUY ≈${ALPACA_BUDGET:.2f}\nTP:{(tp or 0):.6f} SL:{(sl or 0):.6f}")
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")
    except Exception as e:
        await u.message.reply_text(f"🔴 signals_stocks error: {e}")

async def trade_stocks(u:Update,c:ContextTypes.DEFAULT_TYPE):
    st=stdef(u.effective_chat.id)
    try:
        _, ranked=await scan_rank_stocks(st)
        if not ranked: await u.message.reply_text("⚠️ Немає сигналів"); return
        try: clk=await alp_clock(); market_open=bool(clk.get("is_open"))
        except Exception: market_open=True
        if not market_open: await u.message.reply_text("⏸ Ринок акцій закритий — ордери не виставляю."); return
        picks=ranked[:_mode_conf(st)["top_n"]]
        for _, sym, arr in picks:
            side="buy"; px=float(arr[-1]["c"]); conf=_mode_conf(st); tp,sl=calc_sl_tp(side,px,conf)
            if await has_open_long(sym): await u.message.reply_text(f"⚪ SKIP: вже є позиція по {to_order_sym(sym)}"); continue
            if skip_as_duplicate("STOCK",sym,side): await u.message.reply_text(f"⚪ SKIP (дубль): {sym} {side.upper()}"); continue
            try:
                await place_bracket_qty_order_stock(sym, px, ALPACA_BUDGET, tp, sl)
                await u.message.reply_text(f"🟢 ORDER OK: {sym} BUY ≈${ALPACA_BUDGET:.2f}\nTP:{(tp or 0):.6f} SL:{(sl or 0):.6f}")
            except Exception as e:
                await u.message.reply_text(f"🔴 ORDER FAIL {sym} BUY: {e}")
    except Exception as e:
        await u.message.reply_text(f"🔴 trade_stocks error: {e}")

# ===== AUTOSCAN =====
async def _auto_scan_once_for_chat(chat_id:int, ctx:ContextTypes.DEFAULT_TYPE):
    st=stdef(chat_id)
    if not st.get("auto_scan") or not st.get("autotrade"): return
    conf=_mode_conf(st); top_n=int(conf.get("top_n", max(1,ALPACA_TOP_N)))
    try: clk=await alp_clock(); market_open=bool(clk.get("is_open"))
    except Exception: market_open=True
    try: _, c_ranked=await scan_rank_crypto(st)
    except Exception as e: c_ranked=[]; await ctx.bot.send_message(chat_id, f"🔴 Крипто-скан помилка: {e}")
    try: _, s_ranked=await scan_rank_stocks(st)
    except Exception as e: s_ranked=[]; await ctx.bot.send_message(chat_id, f"🔴 Скан акцій помилка: {e}")

    combined=[]
    for sc,sym,arr in c_ranked: combined.append((sc,sym,"crypto",arr))
    for sc,sym,arr in s_ranked: combined.append((sc,sym,"stock",arr))
    combined.sort(reverse=True); picks=combined[:top_n]

    for score,sym,kind,arr in picks:
        if kind=="stock" and not market_open: continue
        if await has_open_long(sym): continue
        side="buy"; px=float(arr[-1]["c"]); tp,sl=calc_sl_tp(side,px,conf)
        if skip_as_duplicate("STOCK" if kind=="stock" else "CRYPTO", sym, side): continue
        try:
            if kind=="stock":
                await place_bracket_qty_order_stock(sym, px, ALPACA_BUDGET, tp, sl)
            else:
                await place_bracket_qty_order_crypto(sym, px, ALPACA_BUDGET, tp, sl)
            await ctx.bot.send_message(chat_id, f"🟢 AUTO BUY: {to_order_sym(sym)} · entry={px:.6f} · budget≈${ALPACA_BUDGET:.2f}")
        except Exception as e:
            await ctx.bot.send_message(chat_id, f"🔴 periodic autoscan error: {e}")

async def periodic_auto_scan(ctx:ContextTypes.DEFAULT_TYPE):
    for chat_id in list(STATE.keys()):
        try: await _auto_scan_once_for_chat(chat_id, ctx)
        except Exception as e:
            try: await ctx.bot.send_message(chat_id, f"🔴 periodic autoscan error: {e}")
            except Exception: pass

async def auto_on(u:Update,c:ContextTypes.DEFAULT_TYPE): st=stdef(u.effective_chat.id); st["auto_scan"]=True; await u.message.reply_text(f"✅ AUTO-SCAN: ON (кожні {SCAN_INTERVAL_SEC}s)")
async def auto_off(u:Update,c:ContextTypes.DEFAULT_TYPE): st=stdef(u.effective_chat.id); st["auto_scan"]=False; await u.message.reply_text("⛔ AUTO-SCAN: OFF")
async def auto_status(u:Update,c:ContextTypes.DEFAULT_TYPE):
    st=stdef(u.effective_chat.id)
    await u.message.reply_text(f"AutoScan={'ON' if st.get('auto_scan') else 'OFF'}; Autotrade={'ON' if st.get('autotrade') else 'OFF'}; Mode={st.get('mode','default')} · Side={st.get('side_mode','long')} · Interval={SCAN_INTERVAL_SEC}s")

async def aggressive(u,c): await set_mode(u,c,"aggressive")
async def scalp(u,c): await set_mode(u,c,"scalp")
async def default(u,c): await set_mode(u,c,"default")
async def swing(u,c): await set_mode(u,c,"swing")
async def safe(u,c): await set_mode(u,c,"safe")

def main()->None:
    if not TG_TOKEN: raise SystemExit("No TELEGRAM_BOT_TOKEN provided")
    app=Application.builder().token(TG_TOKEN).build()
    app.add_handler(CommandHandler("start", start)); app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("aggressive", aggressive)); app.add_handler(CommandHandler("scalp", scalp))
    app.add_handler(CommandHandler("default", default)); app.add_handler(CommandHandler("swing", swing)); app.add_handler(CommandHandler("safe", safe))
    app.add_handler(CommandHandler("long_mode", long_mode)); app.add_handler(CommandHandler("short_mode", short_mode)); app.add_handler(CommandHandler("both_mode", both_mode))
    app.add_handler(CommandHandler("alp_on", alp_on)); app.add_handler(CommandHandler("alp_off", alp_off)); app.add_handler(CommandHandler("alp_status", alp_status))
    app.add_handler(CommandHandler("signals_crypto", signals_crypto)); app.add_handler(CommandHandler("trade_crypto", trade_crypto))
    app.add_handler(CommandHandler("signals_stocks", signals_stocks)); app.add_handler(CommandHandler("trade_stocks", trade_stocks))
    app.add_handler(CommandHandler("auto_on", auto_on)); app.add_handler(CommandHandler("auto_off", auto_off)); app.add_handler(CommandHandler("auto_status", auto_status))
    app.job_queue.run_repeating(periodic_auto_scan, interval=SCAN_INTERVAL_SEC, first=10)
    print("Bot started."); app.run_polling(close_loop=False)

if __name__=="__main__": main()
