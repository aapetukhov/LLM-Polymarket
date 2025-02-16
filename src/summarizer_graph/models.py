from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field, validator

@dataclass
class GraphState:
    """Состояние графа обработки."""
    question: str
    text: str
    summary: Optional[str] = None
    probability: Optional[float] = None

    def __post_init__(self):
        if not self.question or not self.text:
            raise ValueError("Question and text must not be empty")

class SummaryOutput(BaseModel):
    """Модель для структурированного вывода суммаризации."""
    summary: str = Field(description="Краткое резюме текста, содержащее только релевантную информацию для вопроса")

    @validator('summary')
    def validate_summary(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Summary is too short")
        return v

class PredictionOutput(BaseModel):
    """Модель для структурированного вывода предсказания."""
    probability: float = Field(description="Оценённая вероятность события из вопроса в диапазоне [0,1]")

    @validator('probability')
    def validate_probability(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Probability must be between 0 and 1")
        return round(v, 3)
