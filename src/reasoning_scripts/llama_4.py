import re
import os
import asyncio
import json
import argparse
from tqdm import tqdm
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
import httpx


load_dotenv()
DEBUG = True
TOP_K = 10
K_VALUES = [3, 4]

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = "meta-llama/llama-4-maverick"


SCHEMA = {
    "type": "object",
    "properties": {
        "probability_yes": {"type": "integer", "description": "Estimated probability (0-100) that the event will resolve as 'Yes'."},
        "justification": {"type": "string", "description": "Justification based on the provided news context."}
    },
    "required": ["probability_yes", "justification"],
    "additionalProperties": False
}

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def parse_dt(s, fmt="%Y%m%d%H%M%S"):
    return datetime.strptime(s, fmt)

def parse_article_dt(s, fmt="%Y%m%dT%H%M%SZ"):
    return datetime.strptime(s, fmt)

def readable_date(s, fmt="%Y%m%d%H%M%S"):
    try:
        return parse_dt(s, fmt).strftime("%B %d, %Y, %H:%M")
    except:
        return s

def compute_cutoffs(event, n=4, k_values=K_VALUES):
    start = parse_dt(event["start_date"])
    end = parse_dt(min(event["end_date"], event.get("resolution_time", event["end_date"])))
    delta = end - start - timedelta(seconds=1)
    return {k: start + delta * (k / n) for k in k_values}

def build_prompt(event, cutoff):
    cutoff_str = cutoff.strftime("%B %d, %Y, %H:%M")
    articles = [
        a for a in event["articles"]
        if a.get("score", 0) > -3 and parse_article_dt(a["date"]) <= cutoff
    ][:TOP_K]
    text = "\n\n".join(
        f"[{i+1}] TITLE: {a['title'].strip()}\nDATE: {a['date']}\nTEXT: {a['text'].strip()}"
        for i, a in enumerate(articles)
    )
    start, end = readable_date(event["start_date"]), readable_date(event["end_date"])
    prompt = f"""
You are an expert geopolitical forecaster. Estimate the probability that the event resolves as "Yes".

EVENT:
Question: {event["title"].strip()}
Description: {event["description"].strip()}
Date range: {start} to {end}

NEWS ARTICLES (cutoff {cutoff_str}):
{text}

RESPONSE FORMAT (JSON):
{{
  "justification": "<explanation>",
  "probability_yes": <integer 0-100>
}}
""".strip()
    return prompt, len(articles)

async def query_llm(prompt, event_id, client):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_output_tokens": 1024,
        "logprobs": True,
        "top_logprobs": 5,
        "provider": {"only": ["DeepInfra", "kluster.ai", "NovitaAI"], "require_parameters": False}
    }
    try:
        resp = await client.post(f"{BASE_URL}/chat/completions", json=payload, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        clean = re.sub(r"^```(?:json)?\s*|```$", "", text.strip(), flags=re.IGNORECASE)
        parsed = json.loads(clean)
        return {"parsed": parsed, "usage": data["choices"][0].get("usage")}
    except Exception as e:
        if DEBUG:
            print(f"[Error] Event {event_id}: {e}")
        return {"error": str(e)}

async def process_cutoff(event, name, cutoff, client):
    prompt, num = build_prompt(event, cutoff)
    resp = await query_llm(prompt, event["id"], client)
    entry = {
        "event_id": event["id"],
        "experiment": name,
        "pred_date": cutoff.strftime("%Y%m%d%H%M%S"),
        "num_articles": num
    }
    if "parsed" in resp:
        entry.update(resp["parsed"])
        entry["usage"] = resp.get("usage")
    else:
        entry["error"] = resp["error"]
    return entry

async def evaluate_event(event, client):
    cutoffs = compute_cutoffs(event)
    tasks = [
        process_cutoff(event, name, cutoff, client)
        for name, cutoff in cutoffs.items()
    ]
    return await asyncio.gather(*tasks)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    with open(args.input_path, encoding="utf-8") as f:
        data = json.load(f)
    events = data if isinstance(data, list) else [data]

    async with httpx.AsyncClient() as client:
        tasks = [evaluate_event(e, client) for e in events]
        all_preds = await asyncio.gather(*tasks)

    output = []
    for event, preds in zip(events, all_preds):
        output.append({
            "event_id": event["id"],
            "title": event.get("title"),
            "description": event.get("description"),
            "start_date": event.get("start_date"),
            "end_date": event.get("end_date"),
            "resolution_time": event.get("resolution_time"),
            "predictions": preds
        })

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
