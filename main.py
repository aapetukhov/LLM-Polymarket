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
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY")),
        prompt=prompt,
        output_parser=parser
    )
    
    result = chain.invoke({
        "question": question,
        "description": description,
        "summaries": combined,
        "format_instructions": format_instructions
    })
    
    return float(result["probability"])


def date_for_gdelt(date: str) -> str:
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d"
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date, fmt)
            return dt.strftime("%Y%m%d%H%M%S")
        except ValueError:
            continue
    
    raise ValueError(f"Unknown date format: {date}")


def main():
    start_date_min = "2024-07-18"
    end_date_max = "2024-12-31"
    gamma = GammaMarketClient()

    if start_date_min and end_date_max:
        local_file_path = f"data/polymarket/events_{start_date_min}_to_{end_date_max}.json"
    else:
        local_file_path = ""

    events = gamma.get_binary_events(
        querystring_params=
        {
            "start_date_min": start_date_min,
            "end_date_max": end_date_max,
            "active": False,
            "archived": True,
            "tag": "politics"
         },
        local_file_path=local_file_path
    )

    message = f"NUM_EVENTS: {len(events)}"
    separator = "-" * len(message)
    print(f"{separator}\n{message}\n{separator}")

    if not events:
        print("No binary events found.")
        return

    # TODO: analyze several events at a time
    event = events[0]
    question = event.title
    description = event.markets[0].description
    
    event_start_date = event.startDate or start_date_min
    event_end_date = event.endDate or end_date_max


    keywords = extract_keywords(question)
    if not keywords:
        print("No keywords extracted.")
        return

    gdelt = GDELTRetriever(save_path=f"data/gdelt/{event.slug}")
    query = " OR ".join(keywords)
    print("QUERY:", query)
    gdelt_data = gdelt.retrieve(
        f"({query})" if len(keywords) > 1 else query,
        mode="ArtList",
        startdatetime=date_for_gdelt(event_start_date),
        enddatetime=date_for_gdelt(event_end_date),
        language="eng"
    )

    if not gdelt_data:
        print("No news found.")
        return get_event_probability([description])

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
