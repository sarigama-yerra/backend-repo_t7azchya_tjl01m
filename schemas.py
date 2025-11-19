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
