import re
import os
import sys
import json
import argparse
from tqdm import tqdm
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests

load_dotenv()
DEBUG = True
TOP_K = 10
K_VALUES = [3]

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = "deepseek/deepseek-r1"


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


def readable_date(dt_str, fmt="%Y%m%d%H%M%S"):
    try:
        return datetime.strptime(dt_str, fmt).strftime("%B %d, %Y, %H:%M")
    except Exception:
        return dt_str


def parse_dt(dt_str, fmt="%Y%m%d%H%M%S"):
    return datetime.strptime(dt_str, fmt)


def parse_article_dt(dt_str, fmt="%Y%m%dT%H%M%SZ"):
    return datetime.strptime(dt_str, fmt)


def compute_cutoff_dates(event, n=4, k_values=K_VALUES):
    start = parse_dt(event["start_date"])
    end = parse_dt(min(event["end_date"], event.get("resolution_time", event["end_date"])))
    delta = end - start - timedelta(seconds=1)
    return {k: start + delta * (k / n) for k in k_values}


def build_base_rate_prompt(event):
    start = readable_date(event["start_date"])
    end = readable_date(event["end_date"])
    return f"""
You are an expert geopolitical forecaster. First, provide only the base rate probability (outside view) that the event resolves as "Yes".

EVENT:
Question: {event["title"].strip()}
Description: {event["description"].strip()}
Date range: {start} to {end}

RESPONSE FORMAT (JSON):
{{
  "probability_yes": <integer 0-100>
}}
""".strip()


def build_inside_view_prompt(event, cutoff, base_rate):
    cutoff_str = cutoff.strftime("%B %d, %Y, %H:%M")
    filtered = [a for a in event["articles"] if a.get("score", 0) > -3 and parse_article_dt(a["date"]) <= cutoff]
    articles = filtered[:TOP_K]
    articles_text = "\n\n".join([
        f"[{i+1}] TITLE: {a.get('title','').strip()}\nDATE: {a.get('date','').strip()}\nTEXT: {a.get('text','').strip()}"
        for i, a in enumerate(articles)
    ])
    return f"""
You are an expert geopolitical forecaster. Estimate the probability that the event resolves as "Yes". You already have a base rate (outside view) of {base_rate}%.
Now analyze the news articles and update your estimate (inside view) or leave it unchanged.

EVENT:
Question: {event["title"].strip()}

NEWS ARTICLES (cutoff {cutoff_str}):
{articles_text}

RESPONSE FORMAT (JSON):
{{
  "probability_yes": <integer 0-100>,
  "justification": "<explanation>"
}}
""".strip()


def query_llm(prompt, event_id):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        # "response_format": {
        #     "type": "json_schema",
        #     "json_schema": {"schema": SCHEMA, "strict": True},
        #     "required": ["justification", "probability_yes"]
        # },
        "temperature": 0.0,
        "max_output_tokens": 512,
        # "logprobs": True,
        # "top_logprobs": 5,
        "provider": {"only": ["DeepInfra", "inference.net", "Lambda"], "require_parameters": False}
    }

    try:
        resp = requests.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"\033[91m[Event {event_id}] Request failed: {e}\033[0m")
        return {"raw_response": None, "output_parsed": None, "error": f"request_failed: {e}"}

    try:
        data = resp.json()
    except ValueError as e:
        text = resp.text
        print(f"\033[91m[Event {event_id}] Response not valid JSON: {e}\033[0m")
        return {"raw_response": resp.text, "output_parsed": {"parse_error": True, "raw_text": text}, "error": "invalid_json"}

    choice = data.get("choices", [{}])[0]
    text = choice.get("message", {}).get("content", "")
    reasoning = choice.get("message", {}).get("reasoning", "")

    clean = re.sub(r"^```(?:json)?\s*|```$", "", text.strip(), flags=re.IGNORECASE)

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"[Event {event_id}] JSON schema parse error: {e}")
        return {"raw_response": choice, "output_parsed": {"parse_error": True, "raw_text": clean, "reasoning": reasoning}, "error": "schema_parse_error", "reasoning": reasoning}

    return {"raw_response": choice, "output_parsed": parsed, "reasoning": reasoning}


def evaluate_event(event, k_values=K_VALUES):
    results = []
    cutoffs = compute_cutoff_dates(event=event, n=4, k_values=k_values)
    for name, cutoff in cutoffs.items():
        base_prompt = build_base_rate_prompt(event)
        base_resp = query_llm(base_prompt, event["id"])
        base_reasoning = base_resp.get("reasoning", "")
        base_parsed = base_resp.get("output_parsed", {}) or base_resp
        base_rate = base_parsed.get("probability_yes", base_parsed)

        inside_prompt = build_inside_view_prompt(event, cutoff, base_rate)
        inside_resp = query_llm(inside_prompt, event["id"])
        inside_parsed = inside_resp.get("output_parsed", {}) or {}

        entry = {
            "event_id": event["id"],
            "experiment": name,
            "pred_date": cutoff.strftime("%Y%m%d%H%M%S"),
            "base_rate": base_rate,
            "base_reasoning": base_reasoning,
            "probability_yes": inside_parsed.get("probability_yes", inside_parsed),
            "justification": inside_parsed.get("justification", inside_resp["reasoning"]),
            "num_articles": len(filtered := [a for a in event["articles"] if a.get("score",0)>-3 and parse_article_dt(a["date"])<=cutoff][:TOP_K])
        }
        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    events = data if isinstance(data, list) else [data]

    results = []
    try:
        for event in tqdm(events, desc="Processing events"):
            try:
                if isinstance(event, list):
                    event, k_values = event[0], event[1]
                else:
                    # TODO: REMEMBER IT IS ALSO HERE!!!
                    k_values = K_VALUES
                preds = evaluate_event(event, k_values=k_values)
                results.append({
                    "event_id": event.get("id"),
                    "title": event.get("title"),
                    "description": event.get("description"),
                    "start_date": event.get("start_date"),
                    "end_date": event.get("end_date"),
                    "resolution_time": event.get("resolution_time"),
                    "outcome_prices": event.get("outcome_prices"),
                    "predictions": preds
                })
            except Exception as e:
                print(f"\033[91m[Event {event.get('id')}] Evaluation failed:\033[0m \033[93m{type(e).__name__}\033[0m - {str(e)}")
                results.append({"event_id": event.get("id"), "error": str(e)})
            if len(results) % 100 == 0:
                with open(args.output_path, 'w', encoding='utf-8') as out_f:
                    json.dump(results, out_f, ensure_ascii=False, indent=2)

    except KeyboardInterrupt:
        print("Interrupted by user, saving progress and exiting...")
        with open(args.output_path, 'w', encoding='utf-8') as out_f:
            json.dump(results, out_f, ensure_ascii=False, indent=2)
        sys.exit(1)

    with open(args.output_path, 'w', encoding='utf-8') as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

    print(f"Processing complete. Results written to {args.output_path}")


if __name__ == '__main__':
    main()
