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



def parse_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception:
        return ""


def date_for_gdelt(date):
    formats = ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"]
    for fmt in formats:
        try:
            dt = datetime.strptime(date, fmt)
            return dt.strftime("%Y%m%d%H%M%S")
        except ValueError:
            continue
    return None


def extract_keywords(question: str) -> list:
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Extract keywords from the following binary question:\nQuestion: {question}\nKeywords (comma separated):"
    )
    chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY")),
        prompt=prompt,
        output_parser=CommaSeparatedListOutputParser()
    )
    keywords = chain.run({"question": question})
    return keywords


def process_events(json_path, save_path="data/news_results.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    gdelt = GDELTRetriever()
    results = []

    for event in events:
        event_id = event.get("id")
        title = event.get("title", "")
        description = event.get("markets", [{}])[0].get("description", "")
        start_date = date_for_gdelt(event.get("startDate"))
        end_date = date_for_gdelt(event.get("endDate"))

        if not start_date or not end_date:
            print(f"Skipping event {event_id} due to missing dates.")
            continue

        keywords = extract_keywords(title)
        query = " OR ".join(keywords)

        print(f"Processing event {event_id}: {title} | Query: {query}")

        articles_metadata = gdelt.retrieve_articles(query, start_date, end_date)
        articles = [
            {
                "url": article["url"],
                "title": article["title"],
                "date": article["date"],
                "text": parse_article(article["url"])
            }
            for article in articles_metadata
        ]

        results.append({
            "id": event_id,
            "title": title,
            "description": description,
            "start_date": start_date,
            "end_date": end_date,
            "articles": articles
        })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Processing complete. Results saved to {save_path}")


if __name__ == "__main__":
    process_events(
        json_path="data/polymarket/events_2024-01-01_to_2024-01-05.json",
        save_path="data/gdelt/testing.json"
    )