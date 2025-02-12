import os
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END


load_dotenv()

class GraphState:
    def __init__(self, question, text):
        self.question = question
        self.text = text
        self.summary = None
        self.probability = None

class SummaryOutput(BaseModel):
    summary: str = Field(description="Краткое резюме текста, содержащее только релевантную информацию для вопроса")

class PredictionOutput(BaseModel):
    probability: float = Field(description="Оценённая вероятность события из вопроса в диапазоне [0,1]")


summarizer = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    max_tokens=150
).with_structured_output(SummaryOutput)


predictor = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    max_tokens=50
).with_structured_output(PredictionOutput)


def summarize(state: GraphState):
    response: SummaryOutput = summarizer.invoke(f"""
    Вопрос: "{state.question}"
    Текст: "{state.text}"

    Суммаризируй текст, оставляя только информацию, помогающую ответить на вопрос.
    """)
    state.summary = response.summary
    return {"state": state}


def predict(state):
    response = predictor.invoke(f"""
    Вопрос: "{state.question}"
    Суммаризованный текст: "{state.summary}"

    Оцени вероятность того, что событие в вопросе произойдет (0.000 - точно нет, 1.000 - точно да).
    """)
    state.probability = response.probability
    return {"state": state}


workflow = StateGraph(GraphState)

workflow.add_node("summarization", summarize)
workflow.add_node("prediction", predict)

workflow.set_entry_point("summarization")  
workflow.add_edge("summarization", "prediction")  
workflow.add_edge("prediction", END)  

graph = workflow.compile()



input_state = GraphState(
    question="Купит ли Илон Маск ТикТок до 15-го апреля 2025 года?",
    text="""
        WASHINGTON (Reuters) - US President Donald Trump said on Tuesday he was open to billionaire Elon Musk buying social media app TikTok if the Tesla (TSLA.O) CEO wanted to do so.
        The short video app used by 170 million Americans was taken offline temporarily for users shortly before a law that said it must be sold by its Chinese owner ByteDance on national security grounds, or be banned, took effect on Sunday.
        Bloomberg News reported last week that Chinese officials were in preliminary talks about a potential option to sell TikTok's operations in the United States to Musk, though the company has denied that.
        Trump on Monday signed an executive order seeking to delay by 75 days the enforcement of the law that was put in place after U.S. officials warned that under Chinese parent company ByteDance, there was a risk of Americans' data being misused.
    """
)


output = graph.invoke({"state": input_state})

print(f"Суммаризация: {output['state'].summary}")
print(f"Вероятность события: {output['state'].probability}")