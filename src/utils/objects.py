from typing import Optional, Union, List
from pydantic import BaseModel


class Tag(BaseModel):
    id: str
    label: Optional[str] = None
    slug: Optional[str] = None
    forceShow: Optional[bool] = None  # missing from current events data
    createdAt: Optional[str] = None  # missing from events data
    updatedAt: Optional[str] = None  # missing from current events data


class PolymarketEvent(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    slug: Optional[str] = None  # readable url-style desc
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    active: Optional[bool] = None
    archived: Optional[bool] = None
    closed: Optional[bool] = None
    liquidity: Optional[float] = None
    volume: Optional[float] = None
    markets: Optional[List[str]] = None
    tags: Optional[List[Tag]] = None # e.g. "Politics", "Sports", "Crypto"


class ClobReward(BaseModel):
    id: str  # returned as string in api but really an int?
    conditionId: str
    assetAddress: str
    rewardsAmount: float  # only seen 0 but could be float?
    rewardsDailyRate: int  # only seen ints but could be float?
    startDate: str  # yyyy-mm-dd formatted date string
    endDate: str  # yyyy-mm-dd formatted date string


class Market(BaseModel):
    id: int
    question: Optional[str] = None
    slug: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    active: Optional[bool] = None
    closed: Optional[bool] = None
    archived: Optional[bool] = None
    liquidity: Optional[float] = None
    volume: Optional[float] = None
    outcomePrices: Optional[List[float]] = None
    events: Optional[List[PolymarketEvent]] = None
    clobRewards: Optional[List[ClobReward]] = None
    enableOrderBook: Optional[bool] = None


class SimpleEvent(BaseModel):
    id: int
    ticker: str
    slug: str
    title: str
    description: str
    end: str
    active: bool
    closed: bool
    archived: bool
    restricted: bool
    new: bool
    featured: bool
    restricted: bool
    markets: str


class Article(BaseModel):
    author: Optional[str]
    title: Optional[str]
    description: Optional[str]
    url: Optional[str]
    urlToImage: Optional[str]
    publishedAt: Optional[str]
    content: Optional[str]
