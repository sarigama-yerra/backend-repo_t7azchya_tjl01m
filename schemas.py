"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Example schemas (replace with your own):

class User(BaseModel):
    """
    Users collection schema
    Collection name: "user" (lowercase of class name)
    """
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    """
    Products collection schema
    Collection name: "product" (lowercase of class name)
    """
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# Solana Meme Coin Analyzer schemas

class AnalysisRequest(BaseModel):
    query: str = Field(..., description="Token address (Solana) or symbol/search term")

class AnalysisResult(BaseModel):
    token_address: Optional[str] = None
    name: Optional[str] = None
    symbol: Optional[str] = None
    chain: Optional[str] = None
    score: float = Field(..., ge=0, le=100)
    verdict: str
    metrics: Dict[str, Any]
    insights: List[str]

class AnalysisRecord(BaseModel):
    query: str
    result: AnalysisResult
    source: str = Field("dexscreener", description="Data source used for analysis")
    tag: str = Field("meme", description="Classification tag")

# Chart AI schemas

class ChartAnalysisRequest(BaseModel):
    timeframe: str = Field(..., description="Timeframe of the chart, e.g., 5m, 15m, 1h, 4h, 1d")
    notes: Optional[str] = Field(None, description="Optional notes or context about the snippet")

class TradePlan(BaseModel):
    side: str = Field(..., description="long or short")
    entry: float = Field(..., gt=0)
    stop_loss: float = Field(..., gt=0)
    take_profit_1: float = Field(..., gt=0)
    take_profit_2: Optional[float] = Field(None, gt=0)
    risk_reward: float = Field(..., gt=0)

class ChartAnalysisResult(BaseModel):
    timeframe: str
    trend: str
    momentum: str
    confidence: float = Field(..., ge=0, le=1)
    key_levels: Dict[str, float]
    trade: TradePlan
    rationale: List[str]

class ChartAnalysisRecord(BaseModel):
    request: ChartAnalysisRequest
    result: ChartAnalysisResult
    tag: str = Field("chart_ai", description="Classification tag for chart AI")
