import os
import json
import requests
from datetime import datetime
from newspaper import Article
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains.llm import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser

from src.polymarket.gamma import GammaMarketClient
from src.gdelt import GDELTRetriever

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

# TODO: DEBUG PIPELINE
def summarize_article(text: str) -> str:
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following article in no more than 50 words:\n{text}\nSummary:"
    )
    chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY")),
        prompt=prompt
    )
    return chain.run({"text": text}).strip()


def get_event_probability(question: str, description: str, summaries: list) -> float:
    combined = "\n".join(summaries)
    
    schema = ResponseSchema(
        name="probability",
        description="Probability as a number between 0.0 and 1.0"
    )
    parser = StructuredOutputParser.from_response_schemas([schema])
    format_instructions = parser.get_format_instructions()
    
    prompt = PromptTemplate(
        input_variables=["question", "description", "summaries", "format_instructions"],
        template=(
            "Event question: {question}\n"
            "Description: {description}\n"
            "News summaries:\n{summaries}\n"
            "Based on the above, provide a probability between 0 and 1 that the event will occur. "
            "Your answer must be valid JSON and follow these instructions:\n"
            "{format_instructions}"
        )
    )

    chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY")),
        prompt=prompt,
        output_parser=parser
    )
    
    result = chain.invoke({
        "question": question,
        "description": description,
        "summaries": combined,
        "format_instructions": format_instructions
    })
    # TODO: debug error with typing in debugger
    return float(result["probability"])


def obtain_predictions(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    
    for event in data:
        question = event["title"]
        description = event["description"]
        articles = event["articles"][:10]

        summaries = [summarize_article(article["text"]) for article in articles]
        probability = get_event_probability(question, description, summaries)

        results.append({
            "id": event["id"],
            "question": question,
            "probability": probability
        })

    return results

if __name__ == "__main__":
    file_path = "/Users/andreypetukhov/Documents/Thesis/LLM-Polymarket/data/gdelt/subsample_100.json"
    predictions = obtain_predictions(file_path)

    with open("/Users/andreypetukhov/Documents/Thesis/LLM-Polymarket/data/predictions/subsample_100.json.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)
