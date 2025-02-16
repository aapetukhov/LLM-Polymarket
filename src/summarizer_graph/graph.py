import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from .models import GraphState, SummaryOutput, PredictionOutput
from .config import get_settings

load_dotenv()

settings = get_settings()

def create_llm_clients():
    """Создание клиентов LLM."""
    summarizer = ChatOpenAI(
        model=settings.summarizer_model,
        openai_api_key=settings.openai_api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_summary_tokens
    ).with_structured_output(SummaryOutput)

    predictor = ChatOpenAI(
        model=settings.predictor_model,
        openai_api_key=settings.openai_api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_prediction_tokens
    ).with_structured_output(PredictionOutput)

    return summarizer, predictor

summarizer, predictor = create_llm_clients()

def summarize(state_dict: Dict[str, GraphState]) -> Dict[str, GraphState]:
    """Этап суммаризации текста."""
    state = state_dict["state"]
    try:
        response = summarizer.invoke(
            f"""Вопрос: "{state.question}"
            Текст: "{state.text}"
            
            Суммаризируй текст, оставляя только информацию, помогающую ответить на вопрос."""
        )
        state.summary = response.summary
        return {"state": state}
    except Exception as e:
        raise RuntimeError(f"Summarization failed: {str(e)}")

def predict(state_dict: Dict[str, GraphState]) -> Dict[str, GraphState]:
    """Этап предсказания вероятности."""
    state = state_dict["state"]
    try:
        response = predictor.invoke(
            f"""Вопрос: "{state.question}"
            Суммаризованный текст: "{state.summary}"
            
            Оцени вероятность того, что событие в вопросе произойдет (0.000 - точно нет, 1.000 - точно да)."""
        )
        state.probability = response.probability
        return {"state": state}
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

def create_graph() -> StateGraph:
    """Создание и настройка графа обработки."""
    workflow = StateGraph(GraphState)
    
    workflow.add_node("summarization", summarize)
    workflow.add_node("prediction", predict)
    
    workflow.set_entry_point("summarization")
    workflow.add_edge("summarization", "prediction")
    workflow.add_edge("prediction", END)
    
    return workflow.compile()

def process_text(question: str, text: str) -> GraphState:
    """Обработка текста через граф."""
    graph = create_graph()
    input_state = GraphState(question=question, text=text)
    result = graph.invoke({"state": input_state})
    return result["state"]

input_state = GraphState(
    question="Купит ли Илон Маск ТикТок до 15-го апреля 2025 года?",
    text="""
        WASHINGTON (Reuters) - US President Donald Trump said on Tuesday he was open to billionaire Elon Musk buying social media app TikTok if the Tesla (TSLA.O) CEO wanted to do so.
        The short video app used by 170 million Americans was taken offline temporarily for users shortly before a law that said it must be sold by its Chinese owner ByteDance on national security grounds, or be banned, took effect on Sunday.
        Bloomberg News reported last week that Chinese officials were in preliminary talks about a potential option to sell TikTok's operations in the United States to Musk, though the company has denied that.
        Trump on Monday signed an executive order seeking to delay by 75 days the enforcement of the law that was put in place after U.S. officials warned that under Chinese parent company ByteDance, there was a risk of Americans' data being misused.
    """
)

output = process_text(input_state.question, input_state.text)

print(f"Суммаризация: {output.summary}")
print(f"Вероятность события: {output.probability}")