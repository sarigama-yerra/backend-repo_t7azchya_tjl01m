import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
import requests
import time

from database import create_document, get_documents
from schemas import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisRecord,
    ChartAnalysisRequest,
    ChartAnalysisResult,
    ChartAnalysisRecord,
    TradePlan,
    WalletTrackRequest,
    WalletRecord,
    WalletStats,
    LeaderboardEntry,
    MACDInput,
    MACDResult,
    MACDSignal,
    Candle,
    MACDAlertRecord,
)

app = FastAPI(title="Solana Meme Coin Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Solana Meme Coin Analyzer Backend"}

@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        from database import db
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response

# Utility: simple heuristic scoring for meme coins on Solana

def score_token(ds: Dict[str, Any]) -> Dict[str, Any]:
    """Return score (0-100) and metrics based on DexScreener data structure"""
    metrics: Dict[str, Any] = {}
    score = 50.0

    # Liquidity
    liq = ds.get("liquidity", {})
    liquidity_usd = (liq.get("usd") or 0) if isinstance(liq, dict) else 0
    metrics["liquidity_usd"] = liquidity_usd
    if liquidity_usd > 200000: score += 18
    elif liquidity_usd > 50000: score += 10
    elif liquidity_usd > 10000: score += 5
    else: score -= 8

    # FDV / Market cap
    fdv = ds.get("fdv") or 0
    metrics["fdv"] = fdv
    if fdv and fdv < 5_000_000: score += 6
    elif fdv and fdv > 50_000_000: score -= 6

    # 24h volume and price change
    vol24 = ds.get("volume", {}).get("h24") if isinstance(ds.get("volume"), dict) else None
    metrics["volume_h24"] = vol24 or 0
    if vol24 and vol24 > 200_000: score += 10
    elif vol24 and vol24 > 50_000: score += 6
    elif vol24 and vol24 < 5_000: score -= 6

    change_h24 = ds.get("priceChange", {}).get("h24") if isinstance(ds.get("priceChange"), dict) else None
    metrics["price_change_h24"] = change_h24
    if change_h24 is not None:
        if -10 <= change_h24 <= 60: score += 5
        if change_h24 < -40: score -= 8

    # Age
    age = ds.get("age") or {}
    age_days = age.get("days") if isinstance(age, dict) else None
    metrics["age_days"] = age_days
    if age_days is not None:
        if age_days > 30: score += 5
        elif age_days < 2: score -= 6

    # Holders if available
    metrics["pair_created_at"] = ds.get("pairCreatedAt")

    score = max(0, min(100, score))
    verdict = (
        "Promising momentum" if score >= 75 else
        "Worth watching" if score >= 60 else
        "High risk / low liquidity" if score >= 40 else
        "Extremely speculative"
    )
    return {"score": score, "metrics": metrics, "verdict": verdict}

@app.post("/analyze", response_model=AnalysisResult)
def analyze_token(req: AnalysisRequest):
    """
    Analyze a Solana token by address or search term using public DexScreener API.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    # Try address lookup on solana first
    result_data: Dict[str, Any] | None = None
    base = "https://api.dexscreener.com/latest/dex"  # public

    # Attempt: search endpoint
    try:
        resp = requests.get(f"{base}/search?q={query}", timeout=10)
        if resp.status_code == 200:
            j = resp.json() or {}
            pairs = j.get("pairs") or []
            # Filter for solana chain
            sol_pairs = [p for p in pairs if (p.get("chainId") == "solana" or p.get("chainId") == "solana" or p.get("chainId") == "solana")]
            # Fallback to all if none
            target = sol_pairs[0] if sol_pairs else (pairs[0] if pairs else None)
            if target:
                result_data = target
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)[:80]}")

    if not result_data:
        raise HTTPException(status_code=404, detail="No token data found for your query")

    s = score_token(result_data)

    name = result_data.get("baseToken", {}).get("name") if isinstance(result_data.get("baseToken"), dict) else None
    symbol = result_data.get("baseToken", {}).get("symbol") if isinstance(result_data.get("baseToken"), dict) else None
    token_address = result_data.get("baseToken", {}).get("address") if isinstance(result_data.get("baseToken"), dict) else None
    chain = result_data.get("chainId")

    analysis = AnalysisResult(
        token_address=token_address,
        name=name,
        symbol=symbol,
        chain=chain,
        score=s["score"],
        verdict=s["verdict"],
        metrics=s["metrics"],
        insights=[
            "Liquidity and 24h volume drive most of the score",
            "FDV sanity check penalizes inflated valuations",
            "Very new pairs tend to be riskier",
        ],
    )

    try:
        rec = AnalysisRecord(query=query, result=analysis)
        create_document("analysisrecord", rec)
    except Exception:
        # Non-fatal if database not configured
        pass

    return analysis

# --- Chart AI: analyze uploaded chart image and propose trade plan ---

def _infer_chart_context(filename: str, file_size: int, timeframe: str, notes: Optional[str]) -> Dict[str, Any]:
    """Lightweight heuristics to infer trend/momentum from metadata and notes."""
    tf = timeframe.lower().strip()
    trend = "sideways"
    momentum = "neutral"

    if notes:
        n = notes.lower()
        if any(k in n for k in ["breakout", "uptrend", "bull", "higher high"]):
            trend = "up"
            momentum = "increasing"
        elif any(k in n for k in ["breakdown", "downtrend", "bear", "lower low"]):
            trend = "down"
            momentum = "decreasing"
        elif "range" in n or "consolid" in n:
            trend = "sideways"
            momentum = "compressing"

    # If no notes, nudge based on timeframe
    if trend == "sideways":
        if tf in ("1m", "3m", "5m", "15m"):
            momentum = "increasing"
        elif tf in ("1h", "4h", "1d"):
            momentum = "neutral"

    # Confidence: more context => higher; larger files (higher resolution) => slightly higher
    base_conf = 0.5
    if notes:
        base_conf += 0.15
    if file_size > 500_000:
        base_conf += 0.1
    base_conf = max(0.35, min(0.9, base_conf))

    return {"trend": trend, "momentum": momentum, "confidence": base_conf}


def _propose_trade(trend: str, timeframe: str) -> TradePlan:
    tf = timeframe.lower().strip()
    # default parameters (percent offsets)
    rr = 2.0 if tf in ("1m", "3m", "5m", "15m") else 2.5
    if trend == "up":
        side = "long"
        entry = -0.002  # -0.20% pullback entry
        stop = -0.01    # -1.00% stop
        tp1 = 0.01      # +1.00%
        tp2 = 0.02      # +2.00%
    elif trend == "down":
        side = "short"
        entry = 0.002   # +0.20% bounce entry
        stop = 0.01     # +1.00% invalidation
        tp1 = -0.01     # -1.00%
        tp2 = -0.02     # -2.00%
    else:
        # range play: mean reversion with tighter stops
        side = "long"
        entry = -0.001
        stop = -0.006
        tp1 = 0.006
        tp2 = 0.012
    return TradePlan(side=side, entry=abs(entry), stop_loss=abs(stop), take_profit_1=abs(tp1), take_profit_2=abs(tp2), risk_reward=rr)


@app.post("/chart/analyze", response_model=ChartAnalysisResult)
async def analyze_chart(
    file: UploadFile = File(...),
    timeframe: str = Form(...),
    notes: Optional[str] = Form(None),
):
    # Validate timeframe basic pattern
    tf = timeframe.lower().strip()
    if not tf:
        raise HTTPException(status_code=400, detail="Timeframe is required")

    # Read file to infer size (no heavy image processing to keep it lightweight)
    content = await file.read()
    file_size = len(content)

    ctx = _infer_chart_context(file.filename or "chart.png", file_size, tf, notes)
    trade = _propose_trade(ctx["trend"], tf)

    # Construct key levels as percentage offsets for user to map to price
    key_levels = {
        "entry_offset_pct": trade.entry * 100,
        "stop_offset_pct": trade.stop_loss * 100,
        "tp1_offset_pct": trade.take_profit_1 * 100,
        "tp2_offset_pct": (trade.take_profit_2 or 0) * 100,
    }

    res = ChartAnalysisResult(
        timeframe=tf,
        trend=ctx["trend"],
        momentum=ctx["momentum"],
        confidence=ctx["confidence"],
        key_levels=key_levels,
        trade=trade,
        rationale=[
            "Heuristic derived from timeframe and provided notes",
            "Levels are percentage offsets from current price; map to your chart",
            "Consider liquidity and higher timeframe bias before execution",
        ],
    )

    # persist (best-effort)
    try:
        rec = ChartAnalysisRecord(request=ChartAnalysisRequest(timeframe=tf, notes=notes), result=res)
        create_document("chartanalysisrecord", rec)
    except Exception:
        pass

    return res

# -------- Wallet Tracker (Solscan) --------

SOLSCAN_API = "https://pro-api.solscan.io/v2"
SOLSCAN_KEY = os.getenv("SOLSCAN_API_KEY") or ""
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY") or ""


def _hdrs():
    h = {"accept": "application/json"}
    if SOLSCAN_KEY:
        h["token"] = SOLSCAN_KEY
    return h


def fetch_wallet_txs(address: str, limit: int = 100) -> List[Dict[str, Any]]:
    url = f"{SOLSCAN_API}/account/transactions?address={address}&limit={limit}"
    r = requests.get(url, headers=_hdrs(), timeout=15)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Solscan error: {r.text[:120]}")
    j = r.json() or {}
    return j.get("data") or j.get("tx") or []


def compute_wallet_stats(txs: List[Dict[str, Any]], window_sec: int) -> WalletStats:
    now = int(time.time())
    start = now - window_sec
    pnl = 0.0
    wins = 0
    losses = 0
    trades = 0
    volume = 0.0
    equity_curve: List[Dict[str, Any]] = []

    # Simplified heuristic: treat positive net token flow valued at price as profit; we don't have full fills.
    # Filter by time window
    ftx = [t for t in txs if (t.get("blockTime") or t.get("timestamp") or 0) >= start]
    ftx.sort(key=lambda x: x.get("blockTime") or x.get("timestamp") or 0)

    running_equity = 0.0
    for t in ftx:
        # Attempt to parse instruction summary if available
        change = 0.0
        fee = float(t.get("fee") or 0) / 1e9  # SOL to SOL if fee in lamports
        # If token transfers listed
        transfers = t.get("tokenTransfers") or t.get("tokenBalanceChanges") or []
        sol_change = 0.0
        for tr in transfers:
            amt = tr.get("changeAmount") or tr.get("amount") or 0
            decimals = tr.get("decimals") or 9
            mint = tr.get("mint") or tr.get("tokenAddress")
            # We don't have USD pricing here; best-effort mark-to-fee as cost proxy
            if mint == "So11111111111111111111111111111111111111112":
                sol_change += (float(amt) / (10 ** decimals))
        # treat negative SOL as cost, positive as revenue
        change = -sol_change - fee
        pnl += change
        volume += abs(sol_change)
        trades += 1
        if change > 0:
            wins += 1
        elif change < 0:
            losses += 1
        running_equity += change
        equity_curve.append({"t": t.get("blockTime") or t.get("timestamp") or now, "equity_usd": running_equity})

    hit = (wins / trades) if trades else 0.0
    # Placeholder address/label will be set by caller
    return WalletStats(address="", label=None, window=str(window_sec), pnl_usd=pnl, wins=wins, losses=losses, hit_rate=hit, trades=trades, volume_usd=volume, equity_curve=equity_curve)  # type: ignore


@app.post("/wallets/track")
def track_wallet(req: WalletTrackRequest):
    if not req.address:
        raise HTTPException(status_code=400, detail="address required")

    try:
        # store basic record; refresh happens on-demand
        rec = WalletRecord(address=req.address, label=req.label)
        create_document("walletrecord", rec)
        return {"status": "tracking", "address": req.address, "label": req.label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:120])


@app.get("/wallets/{address}/stats", response_model=WalletStats)
def wallet_stats(address: str, window: str = Query("7d", regex="^(1d|7d|30d)$")):
    # Window map
    win_map = {"1d": 86400, "7d": 7*86400, "30d": 30*86400}
    ws = win_map.get(window, 7*86400)
    txs = fetch_wallet_txs(address, limit=200)
    stats = compute_wallet_stats(txs, ws)
    stats.address = address
    stats.label = None
    stats.window = window
    try:
        create_document("walletstats", stats)
    except Exception:
        pass
    return stats


@app.get("/wallets/leaderboard", response_model=List[LeaderboardEntry])
def wallet_leaderboard(window: str = Query("7d", regex="^(1d|7d|30d)$"), sort: str = Query("pnl", regex="^(pnl|hit_rate|trades)$")):
    # Pull recent cached stats and recompute for a set of tracked wallets
    wallets = get_documents("walletrecord") if True else []
    entries: List[LeaderboardEntry] = []
    for w in wallets:
        addr = w.get("address")
        label = w.get("label")
        try:
            s = wallet_stats(addr, window)  # compute on the fly; could cache later
            entries.append(LeaderboardEntry(address=addr, label=label, pnl_usd=s.pnl_usd, hit_rate=s.hit_rate, trades=s.trades))
        except Exception:
            continue
    if sort == "pnl":
        entries.sort(key=lambda x: x.pnl_usd, reverse=True)
    elif sort == "hit_rate":
        entries.sort(key=lambda x: x.hit_rate, reverse=True)
    else:
        entries.sort(key=lambda x: x.trades, reverse=True)
    return entries[:50]

# -------- MACD Bot --------


def _tf_to_seconds(tf: str) -> int:
    tf = tf.lower().strip()
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400
    return 3600


def fetch_candles_helius(mint: str, timeframe: str, limit: int = 200) -> List[Candle]:
    # Helius Aggregated OHLCV endpoint (if available). Fallback to Birdeye-style or public sources if needed.
    base = f"https://api.helius.xyz/v0/token-metrics/ohlcv?api-key={HELIUS_API_KEY}"
    # Not all environments expose this exact endpoint; we do a best-effort and accept failure gracefully.
    params = {
        "mint": mint,
        "timeframe": timeframe,
        "limit": str(limit)
    }
    try:
        r = requests.get(base, params=params, timeout=15)
        if r.status_code != 200:
            return []
        j = r.json() or {}
        rows = j.get("data") or j.get("candles") or []
        candles: List[Candle] = []
        for row in rows:
            t = int(row.get("t") or row.get("startTime") or 0)
            o = float(row.get("o") or row.get("open") or 0)
            h = float(row.get("h") or row.get("high") or o)
            l = float(row.get("l") or row.get("low") or o)
            c = float(row.get("c") or row.get("close") or o)
            v = float(row.get("v") or row.get("volume") or 0)
            candles.append(Candle(t=t, o=o, h=h, l=l, c=c, v=v))
        return candles
    except Exception:
        return []


def compute_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
    def ema(series: List[float], period: int) -> List[float]:
        if not series:
            return []
        k = 2 / (period + 1)
        out: List[float] = []
        prev = series[0]
        out.append(prev)
        for x in series[1:]:
            prev = x * k + prev * (1 - k)
            out.append(prev)
        return out
    if len(closes) < max(slow, fast) + signal:
        return [], [], []
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast[-len(ema_slow):], ema_slow)]
    sig_line = ema(macd_line, signal)
    # Align lengths
    L = min(len(macd_line), len(sig_line))
    macd_line = macd_line[-L:]
    sig_line = sig_line[-L:]
    hist = [m - s for m, s in zip(macd_line, sig_line)]
    return macd_line, sig_line, hist


def detect_crossovers(candles: List[Candle], macd_line: List[float], sig_line: List[float]) -> List[MACDSignal]:
    signals: List[MACDSignal] = []
    L = min(len(candles), len(macd_line), len(sig_line))
    for i in range(1, L):
        prev_diff = macd_line[i-1] - sig_line[i-1]
        cur_diff = macd_line[i] - sig_line[i]
        if prev_diff <= 0 and cur_diff > 0:
            signals.append(MACDSignal(time=candles[i].t, type="bullish", macd=macd_line[i], signal=sig_line[i], histogram=macd_line[i]-sig_line[i]))
        elif prev_diff >= 0 and cur_diff < 0:
            signals.append(MACDSignal(time=candles[i].t, type="bearish", macd=macd_line[i], signal=sig_line[i], histogram=macd_line[i]-sig_line[i]))
    return signals


def ai_confidence(timeframe: str, volatility: float, recent_signals: int) -> float:
    base = 0.55 if timeframe in ("1h", "4h") else 0.5
    base += min(0.1, volatility / 100)
    base -= min(0.1, max(0, recent_signals - 3) * 0.02)
    return max(0.4, min(0.9, base))


@app.get("/indicator/macd/analyze", response_model=MACDResult)
def macd_analyze(query: str = Query(...), timeframe: str = Query("1h")):
    # Query is a token mint address for Solana
    mint = query.strip()
    candles = fetch_candles_helius(mint, timeframe, limit=300)
    if not candles:
        raise HTTPException(status_code=502, detail="No candles available from provider")
    closes = [c.c for c in candles]
    macd_line, sig_line, hist = compute_macd(closes)
    if not macd_line:
        raise HTTPException(status_code=400, detail="Insufficient data for MACD")
    signals = detect_crossovers(candles[-len(macd_line):], macd_line, sig_line)

    # Volatility proxy: average true range percent (rough)
    rng = [((c.h - c.l) / c.c * 100) if c.c else 0 for c in candles[-50:]]
    vol = sum(rng) / len(rng) if rng else 0
    conf = ai_confidence(timeframe, vol, len([s for s in signals if s.time >= candles[-50].t]))

    # Persist most recent signal best-effort
    try:
        if signals:
            rec = MACDAlertRecord(query=mint, timeframe=timeframe, signal=signals[-1])
            create_document("macdalerts", rec)
    except Exception:
        pass

    return MACDResult(query=mint, timeframe=timeframe, candles=candles[-200:], macd=macd_line[-200:], signal=sig_line[-200:], histogram=hist[-200:], signals=signals[-50:], confidence=conf)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
