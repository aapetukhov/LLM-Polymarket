import os
from langchain_openai import ChatOpenAI
from retriever_graph import StateGraph, END

os.environ["OPENAI_API_KEY"] = "..."

class GraphState:
    def __init__(self, question, text):
        self.question = question
        self.text = text
        self.summary = None
        self.classification = None

summarizer = ChatOpenAI(model="gpt-4-turbo", temperature=0)
classifier = ChatOpenAI(model="gpt-4-turbo", temperature=0)

def summarize(state):
    prompt = f"""
    You are working as a question-oriented summarizer. Given:
    Question: "{state.question}"
    Text: "{state.text}"
    
    Summarize the text, keeping only information that helps answer the question.
    """
    response = summarizer.invoke(prompt)
    state.summary = response.content
    return state

def classify(state):
    prompt = f"""
    You are working as a classifier. Given summary:
    "{state.summary}"
    
    Determine the category of the text (e.g., "finance", "politics", "sports").
    """
    response = classifier.invoke(prompt)
    state.classification = response.content
    return state

workflow = StateGraph(GraphState)

workflow.add_node("summarization", summarize)
workflow.add_node("classification", classify)

workflow.set_entry_point("summarization")
workflow.add_edge("summarization", "classification")
workflow.add_edge("classification", END)

graph = workflow.compile()

input_state = GraphState(
    question="What investment risks are discussed in the text?",
    text="Investing carries many risks, including market fluctuations, inflation, and liquidity..."
)

output_state = graph.invoke(input_state)

print(f"Summary: {output_state.summary}")
print(f"Classification: {output_state.classification}")
