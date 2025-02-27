import os
import json
import requests
from datetime import datetime
from newspaper import Article
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser

from src.polymarket.gamma import GammaMarketClient
from src.gdelt import GDELTRetriever

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def extract_keywords(question: str) -> list:
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Extract keywords from the following binary question:\nQuestion: {question}\nKeywords (comma separated):"
    )
    chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY")),
        prompt=prompt
    )
    keywords_str = chain.run({"question": question})
    return [kw.strip() for kw in keywords_str.split(",") if kw.strip()]

def extract_article_links(gdelt_data: dict) -> list:
    return [article.get("url") for article in gdelt_data.get("articles", []) if article.get("url")]

def parse_article(link: str) -> str:
    try:
        article = Article(link, language="en")
        article.download()
        article.parse()
        return article.text
    except Exception:
        return ""


def summarize_article(text: str) -> str:
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following article in no more than 50 words:\n{text}\nSummary:"
    )
    chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY")),
        prompt=prompt
    )
    return chain.run({"text": text}).strip()


def get_event_probability(summaries: list) -> float:
    combined = "\n".join(summaries)
    prompt = PromptTemplate(
        input_variables=["summaries"],
        template="Based on the following news summaries, provide a probability between 0 and 1 that the event will occur:\n{summaries}\nProbability:"
    )
    chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY")),
        prompt=prompt
    )
    prob_str = chain.run({"summaries": combined}).strip()
    try:
        return float(prob_str)
    except Exception:
        return 0.0

def main():
    gamma = GammaMarketClient()
    events = gamma.get_events(querystring_params={"start_date_min": "2022-01-01", "end_date_max": "2022-12-31"})
    print("NUM_EVENTS:", len(events), '\n')
    if not events:
        print("No binary events found.")
        return
    event = events[0]
    question = event.get("title", "Default binary question")

    keywords = extract_keywords(question)
    if not keywords:
        print("No keywords extracted.")
        return

    gdelt = GDELTRetriever()
    query = " OR ".join(keywords)
    print("QUERY:", query)
    gdelt_data = gdelt.retrieve(query, mode="ArtList", timespan="7d")
    if not gdelt_data:
        print("No news found.")
        return

    links = extract_article_links(gdelt_data)[:10]
    if not links:
        print("No article links extracted.")
        return

    articles = [parse_article(link) for link in links]
    articles = [a for a in articles if a]
    summaries = [summarize_article(text) for text in articles]

    probability = get_event_probability(summaries)
    print(f"Probability for event '{question}': {probability}")

if __name__ == "__main__":
    main()
