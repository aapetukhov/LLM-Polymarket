"""
This script evaluates events using a language model (LLM) to predict the probability of an event resolving as 'Yes' based on provided news articles.
The model used by default is OpenAI's GPT-4o, and the script is designed to work with events that have a specific structure in a JSON file.
"""
import os
import json
import argparse

from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

schema = {
    "type": "object",
    "properties": {
        "probability_yes": {
            "type": "integer",
            "description": "Estimated probability (0-100) that the event will resolve as 'Yes'."
        },
        "justification": {
            "type": "string",
            "description": "Brief justification based on the provided news context."
        }
    },
    "required": ["probability_yes", "justification"],
    "additionalProperties": False
}


def readable_date(dt_str):
    try:
        return datetime.strptime(dt_str, "%Y%m%d%H%M%S").strftime("%B %d, %Y, %H:%M")
    except Exception:
        return dt_str


def build_event_prompt(event):
    articles_text = "\n\n".join([
        f"[{i+1}] TITLE: {a.get('title', '').strip()}\nDATE: {a.get('date', '').strip()}\nTEXT: {a.get('text', '').strip()}"
        for i, a in enumerate(event["articles"][:5])
    ])
    # TOP-k articles = 5 here !!!
    start = readable_date(event["start_date"])
    end = readable_date(event["end_date"])

    prompt = f"""
You are a geopolitical forecasting assistant.

Your task is to estimate the probability that the following event will resolve as "Yes", based only on the provided news articles.

---

EVENT:
Question: {event["title"].strip()}
Description: {event["description"].strip()}
Date range: {start} to {end}

---

NEWS ARTICLES:
{articles_text}

---

REASONING STEPS:
1. Analyze the evidence in the articles.
2. Determine whether they suggest the event is likely to happen.
3. Estimate the probability (0–100) that the outcome is "Yes".
4. Justify your estimate with 1–2 clear sentences.

---

RESPONSE FORMAT (JSON):
{{
  "probability_yes": <integer from 0 to 100>,
  "justification": "<brief explanation>"
}}
"""
    return prompt.strip()



def query_llm(prompt, event):
    response = client.responses.create(
        model="gpt-4o-2024-08-06",
        input=[{"role": "user", "content": prompt}],
        instructions="You are an expert geopolitical forecaster. Think step-by-step, use only information from the articles, avoid hallucinations. Respond concisely and analytically.",
        text={
            "format": {
                "type": "json_schema",
                "strict": True,
                "schema": schema,
                "name": "event_forecast"
            }
        },
        max_output_tokens=1024,
        temperature=0.0,
        metadata={"event_id": event["id"]},
        user=str(event["id"])
    )

    message_content = response.output_text
    full_response_data = {
        "prompt": prompt,
        "raw_response": response.model_dump(),
        "output_parsed": json.loads(message_content)
    }
    return full_response_data


def evaluate_event(event):
    prompt = build_event_prompt(event)
    response_data = query_llm(prompt, event)
    return {
        "id": event["id"],
        "probability_yes": response_data["output_parsed"]["probability_yes"],
        "justification": response_data["output_parsed"]["justification"],
        "debug": {
            "prompt": response_data["prompt"],
            "raw_response": response_data["raw_response"]
        }
    }


def evaluate_event(event):
    prompt = build_event_prompt(event)
    response_data = query_llm(prompt, event)
    return {
        "id": event["id"],
        "title": event["title"],
        "description": event["description"],
        "start_date": event["start_date"],
        "end_date": event["end_date"],
        "outcome_prices": event["outcome_prices"],
        "probability_yes": response_data["output_parsed"]["probability_yes"],
        "justification": response_data["output_parsed"]["justification"],
        "raw_response": response_data["raw_response"],
    }

def process_events(events):
    results = []
    for event in tqdm(events, desc="Processing events"):
        try:
            prediction = evaluate_event(event)
            results.append(prediction)
        except Exception as e:
            results.append({
                "id": event.get("id", "unknown"),
                "error": str(e)
            })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file with events")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSON file")
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = data if isinstance(data, list) else [data]
    results = process_events(events)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
