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
TOP_K = 15

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = "deepseek/deepseek-chat"


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


def compute_cutoff_dates(event, n=4):
    start = parse_dt(event["start_date"])
    end = parse_dt(min(event["end_date"], event.get("resolution_time", event["end_date"])))
    delta = end - start - timedelta(seconds=1)
    return {k: start + delta * (k / n) for k in [3, 4]}


def build_event_prompt(event, cutoff):
    cutoff_str = cutoff.strftime("%B %d, %Y, %H:%M")
    filtered = [a for a in event["articles"] if a.get("score", 0) > -3 and parse_article_dt(a["date"]) <= cutoff]
    articles = filtered[:TOP_K]
    articles_text = "\n\n".join([
        f"[{i+1}] TITLE: {a.get('title','').strip()}\nDATE: {a.get('date','').strip()}\nTEXT: {a.get('text','').strip()}"
        for i, a in enumerate(articles)
    ])
    start_str = readable_date(event["start_date"])
    end_str = readable_date(event["end_date"])
    prompt = f"""
You are an expert geopolitical forecaster. Estimate the probability that the event resolves as "Yes".

EVENT:
Question: {event["title"].strip()}
Description and resolution conditions: {event["description"].strip()}
Date range: {start_str} to {end_str}

Use the following reasoning framework to structure your forecast:
1. Compare this event to similar past events. Estimate a base rate: how often do such outcomes occur?
2. Identify key facts and evidence in the news that support or challenge possible outcomes.
3. Incorporate recent developments or shifts. What has changed over time?
4. Consider past predictions for similar questions. What went right or wrong and why?
5. Clarify what is actually being asked. Disambiguate if needed.
6. Who has influence over the outcome? What are their incentives and constraints?
7. What legal, institutional, or cultural rules might shape the outcome?
8. Include alternative viewpoints. What might an opposing forecaster argue?
9. Could any unlikely but high-impact events drastically shift the situation?


NEWS ARTICLES:
{articles_text}


Use the following reasoning framework to structure your forecast:
1. Compare this event to similar past events. Estimate a base rate: how often do such outcomes occur?
2. Identify key facts and evidence in the news that support or challenge possible outcomes.
3. Incorporate recent developments or shifts. What has changed over time?
4. Consider past predictions for similar questions. What went right or wrong and why?
5. Clarify what is actually being asked. Disambiguate if needed.
6. Who has influence over the outcome? What are their incentives and constraints?
7. What legal, institutional, or cultural rules might shape the outcome?
8. Include alternative viewpoints. What might an opposing forecaster argue?
9. Could any unlikely but high-impact events drastically shift the situation?


RESPONSE FORMAT (JSON):
{{
  "justification": "<explanation>",
  "probability_yes": <integer 0-100>,
}}
""".strip()
    return prompt, len(articles)


def query_llm(prompt, event_id):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": SCHEMA, "strict": True},
            "required": ["justification", "probability_yes"]
        },
        "temperature": 0.0,
        "max_output_tokens": 1024,
        "logprobs": True,
        "top_logprobs": 5,
        "provider": {"only": ["DeepInfra", "NovitaAI", "Nebius AI Studio"], "require_parameters": False}
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

    clean = re.sub(r"^```(?:json)?\s*|```$", "", text.strip(), flags=re.IGNORECASE)

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"[Event {event_id}] JSON schema parse error: {e}")
        return {"raw_response": choice, "output_parsed": {"parse_error": True, "raw_text": clean}, "error": "schema_parse_error"}

    return {"raw_response": choice, "output_parsed": parsed}



def evaluate_event(event):
    results = []
    cutoffs = compute_cutoff_dates(event)

    for name, cutoff in cutoffs.items():
        prompt, num_articles = build_event_prompt(event, cutoff)
        resp = query_llm(prompt, event["id"])
        parsed = resp.get("output_parsed", {}) or {}
        entry = {
            "event_id": event["id"],
            "experiment": name,
            "pred_date": cutoff.strftime("%Y%m%d%H%M%S"),
            "probability_yes": parsed.get("probability_yes"),
            "justification": parsed.get("justification"),
            "num_articles": num_articles,
        }
        if resp.get("raw_response") and isinstance(resp["raw_response"], dict):
            entry["usage"] = resp["raw_response"].get("usage")
        if resp.get("error"):
            entry["error"] = resp["error"]
            print(
                f"\033[91m[Error] Event {event['id']} | {name}:\033[0m {resp['error']}"
            )
            print()
        else:
            if DEBUG:
                print(
                    f"\033[94m[Event {event['id']} | {name}]\033[0m "
                    f"\033[92m{entry['probability_yes']}% yes\033[0m - "
                    f"\033[93m{entry['justification']}\033[0m"
                )
                print()
        results.append(entry)

    return results


def process_events(events):
    output = []
    for e in tqdm(events, desc="Processing events"):
        try:
            preds = evaluate_event(e)
            output.append({
                "event_id": e["id"],
                "title": e["title"],
                "description": e["description"],
                "start_date": e["start_date"],
                "end_date": e["end_date"],
                "resolution_time": e.get("resolution_time"),
                "predictions": preds
            })
        except Exception as ex:
            if DEBUG:
                print(f"Error on event {e.get('id')}: {ex}")
            output.append({"event_id": e.get("id"), "error": str(ex)})
    return output


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
                preds = evaluate_event(event)
                results.append({
                    "event_id": event.get("id"),
                    "title": event.get("title"),
                    "description": event.get("description"),
                    "start_date": event.get("start_date"),
                    "end_date": event.get("end_date"),
                    "resolution_time": event.get("resolution_time"),
                    "predictions": preds
                })
            except Exception as e:
                print(f"\033[91m[Event {event.get('id')}] Evaluation failed:\033[0m \033[93m{type(e).__name__}\033[0m - {str(e)}")
                results.append({"event_id": event.get("id"), "error": str(e)})
            finally:
                with open(args.output_path, 'w', encoding='utf-8') as out_f:
                    json.dump(results, out_f, ensure_ascii=False, indent=2)

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
