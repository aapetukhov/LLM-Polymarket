import json
from typing import Optional, Union, List
from pydantic import BaseModel


class ClobReward(BaseModel):
    id: str  # returned as string in api but really an int?
    conditionId: str
    assetAddress: str
    rewardsAmount: float  # only seen 0 but could be float?
    rewardsDailyRate: int  # only seen ints but could be float?
    startDate: str  # yyyy-mm-dd formatted date string
    endDate: str  # yyyy-mm-dd formatted date string


class Tag(BaseModel):
    id: str
    label: str
    slug: str


class Market(BaseModel):
    id: str
    question: str
    conditionId: str
    slug: str
    resolutionSource: Optional[str] = None
    endDate: Optional[str] = None
    startDate: Optional[str] = None
    image: Optional[str] = None
    icon: Optional[str] = None
    description: Optional[str] = None
    outcomes: List[str]
    outcomePrices: List[float]
    volume: Optional[float] = None
    active: bool
    closed: bool

    @classmethod
    def parse_market(cls, market: dict):
        if "outcomes" in market:
            market["outcomes"] = json.loads(market["outcomes"]) if isinstance(market["outcomes"], str) else market["outcomes"]
        else:
            market["outcomes"] = []

        if "outcomePrices" in market:
            market["outcomePrices"] = [float(x) for x in json.loads(market["outcomePrices"])] if isinstance(market["outcomePrices"], str) else market["outcomePrices"]
        else:
            market["outcomePrices"] = []
        return cls(**market)


class PolymarketEvent(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    slug: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    active: Optional[bool] = None
    archived: Optional[bool] = None
    closed: Optional[bool] = None
    liquidity: Optional[float] = None
    volume: Optional[float] = None
    markets: Optional[List[Market]] = None
    tags: Optional[List[Tag]] = None
    binary: Optional[bool] = None



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


# TODO: add object for article summarization
# TODO: add object for probability prediction
