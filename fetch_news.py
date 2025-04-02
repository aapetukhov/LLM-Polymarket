"""
This script fetches news articles related to events from a JSON file, processes them, and saves the results to a new JSON file.
It uses the GDELT API to retrieve articles based on keywords extracted from the event title and description.
The script includes functions to parse articles, compute date ranges, extract keywords, and format queries.
It also includes a command-line interface for specifying input and output file paths.
The script uses the Newspaper3k library to parse articles and the Langchain library for LLM processing.
The script is designed to be run from the command line and takes two arguments: the path to the input JSON file and the path to save the output JSON file.
"""
import os
import re
import json
import requests
import argparse
from tqdm import tqdm
from datetime import datetime, timedelta
from newspaper import Article

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains.llm import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.runnables import RunnableSequence

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


def gdelt_date_string(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def parse_datetime(date: str) -> datetime | None:
    formats = ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date, fmt)
        except ValueError:
            continue
    return None


def compute_pre_days(start: datetime, end: datetime) -> int:
    duration = (end - start).days
    if duration < 2:
        return 1
    elif duration <= 5:
        return 2
    elif duration <= 14:
        return 3
    else:
        return 5


def extract_keywords(title: str, description: str) -> list:
    prompt = PromptTemplate(
        input_variables=["title", "description"],
        template=(
            "Extract optimized keywords from the following event title and description to query the GDELT dataset.\n"
            "Your goal is to find relevant English-language news articles about this event.\n"
            "Follow these rules:\n"
            "- Focus on named entities (e.g., people, organizations, usernames).\n"
            "- Include only terms likely to appear in news coverage.\n"
            "- Include synonyms or aliases if relevant.\n"
            '- Expand abbreviations (e.g., "AI" should be "Artificial Intelligence", "USA" should be "United States").\n'
            '- If a word contains special characters (such as hyphens, dots, or spaces), wrap it in double quotes (e.g., "Ko Wen-je", "Pump.fun", "Taiwan election")\n'
            "- Avoid dates, generic terms, or overly technical terms not used in news headlines (e.g., 'market resolution', 'social media').\n"
            "- Return a comma-separated list of optimized keywords.\n\n"
            "Title: {title}\n"
            "Description: {description}\n\n"
            "Optimized Keywords:"
        )
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
    parser = CommaSeparatedListOutputParser()

    chain: RunnableSequence = prompt | llm | parser
    keywords = chain.invoke({"title": title, "description": description})
    return clean_keywords(keywords)


def clean_keywords(raw_keywords: list[str]) -> list[str]:
    cleaned = []
    for kw in raw_keywords:
        kw = kw.strip().strip("'\"")
        if len(kw) < 3:
            continue
        if re.fullmatch(r"\d{1,2}(st|nd|rd|th)?", kw.lower()):
            continue
        if re.fullmatch(r"\d{4}", kw):
            continue
        cleaned.append(kw)
    return cleaned


def format_query(keywords):
    formatted_keywords = []
    
    for word in keywords:
        # if re.search(r"[-\s&|!().]", word):
        #     formatted_keywords.append(f'"{word}"')
        if not re.fullmatch(r"\w+", word):
            formatted_keywords.append(f'"{word}"')
        else:
            formatted_keywords.append(word)

    return f'({" OR ".join(formatted_keywords)})' if len(formatted_keywords) > 1 else formatted_keywords[0]


def process_events(json_path, save_path="data/news_results.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    gdelt = GDELTRetriever()
    results = []

    for event in tqdm(events, desc="Processing events"):
        try:
            event_id = event.get("id")
            title = event.get("title", "")
            description = event.get("markets", [{}])[0].get("description", "")
            outcome_prices = event.get("markets", [{}])[0].get("outcomePrices", [])

            # computing pre-days to take for the the news coverage
            start_dt = parse_datetime(event.get("startDate"))
            end_dt = parse_datetime(event.get("endDate"))

            if not start_dt or not end_dt:
                print(f"\033[33m{'=' * (len(msg := f'Skipping event {event_id} due to missing dates.') + 4)}\n= {msg} =\n{'=' * (len(msg) + 4)}\033[0m")
                continue

            pre_days = compute_pre_days(start_dt, end_dt)
            start_date = gdelt_date_string(start_dt - timedelta(days=pre_days))
            end_date = gdelt_date_string(end_dt)

            keywords = extract_keywords(title, description)
            query = format_query(keywords)

            print(f"\033[1;30m\nProcessing event {event_id}: {title}\033[0m\n  Query: {query}")

            articles_metadata = gdelt.retrieve(
                query=query,
                mode="ArtList",
                startdatetime=start_date,
                enddatetime=end_date,
                language="eng",
                save_to_file=False
            )

            articles = []
            for article in articles_metadata["articles"]:
                if article.get("language", "English") != "English":
                    # not considering articles in other languages for now
                    continue
                text = parse_article(article["url"])
                if text:
                    # sometimes returns empty text
                    # if text is empty, skip the article
                    articles.append({
                        "url": article["url"],
                        "title": article["title"],
                        "date": article["seendate"],
                        "text": text
                    })

            results.append({
                "id": event_id,
                "title": title,
                "description": description,
                "start_date": start_date,
                "end_date": end_date,
                "outcome_prices": outcome_prices,
                "articles": articles
            })
            print(f"\033[32m{'=' * (len(msg := f'Found {len(articles)} articles for event {event_id}') + 4)}\n= {msg} =\n{'=' * (len(msg) + 4)}\033[0m")

        except Exception as e:
            print(f"\033[91m{'=' * (len(msg := f'Error processing event {event_id}: {e}') + 4)}\n= {msg} =\n{'=' * (len(msg) + 4)}\033[0m")
            continue

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\033[34m{'=' * (len(msg := f'Processing complete. Results saved to {save_path}') + 4)}\n= {msg} =\n{'=' * (len(msg) + 4)}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=str, help="Path to input JSON file")
    parser.add_argument("save_path", type=str, nargs="?", default="data/news_results.json", help="Path to save output JSON")
    args = parser.parse_args()

    process_events(args.json_path, args.save_path)