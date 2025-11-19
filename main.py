import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional
import requests

from database import create_document
from schemas import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisRecord,
    ChartAnalysisRequest,
    ChartAnalysisResult,
    ChartAnalysisRecord,
    TradePlan,
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
