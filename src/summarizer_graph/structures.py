from pydantic import BaseModel

class NewsInput(BaseModel):
    question: str
    news_text: str

class ExtractedInfo(BaseModel):
    relevant_summary: str