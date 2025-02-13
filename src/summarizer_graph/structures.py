from pydantic import BaseModel, Field


class NewsInput(BaseModel):
    question: str
    news_text: str


class ExtractedInfo(BaseModel):
    relevant_summary: str


class GraphState:
    def __init__(self, question, text):
        self.question = question
        self.text = text
        self.summary = None
        self.probability = None


class SummaryOutput(BaseModel):
    summary: str = Field(description="Краткое резюме текста, содержащее только релевантную информацию для вопроса")


class PredictionOutput(BaseModel):
    probability: float = Field(description="Оценённая вероятность события в диапазоне [0.0, 1.0]")
