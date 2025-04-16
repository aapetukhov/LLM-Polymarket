# TODO: DEBUG AND LAUNCH (!)

import re, os, json, argparse, random, requests
from tqdm import tqdm
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEBUG = True


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
    end = parse_dt(event["end_date"])
    delta = end - start - timedelta(seconds=1)
    return {k: start + delta * (k / n) for k in [1, 2, 3, 4]}


def build_event_prompt(event, cutoff):
    cutoff_str = cutoff.strftime("%B %d, %Y, %H:%M")
    filtered = [a for a in event["articles"]
                if a.get("score", 0) > -1 and parse_article_dt(a["date"]) <= cutoff]
    articles = filtered[:5]
    articles_text = "\n\n".join([
        f"[{i+1}] TITLE: {a.get('title', '').strip()}\nDATE: {a.get('date', '').strip()}\nTEXT: {a.get('text', '').strip()}"
        for i, a in enumerate(articles)
    ])
    start_str = readable_date(event["start_date"])
    end_str = readable_date(event["end_date"])
    return f"""
You are a geopolitical forecasting assistant.

Estimate the probability that the event resolves as "Yes" using only news articles available until {cutoff_str}.

EVENT:
Question: {event["title"].strip()}
Description: {event["description"].strip()}
Date range: {start_str} to {end_str}

REASONING STEPS:
1. Write down any additional relevant information that is not included above. This should be specific facts that you already know the answer to, rather than information that needs to be looked up.
2. Analyze the evidence in the articles.
3. Determine whether they suggest the event is likely to happen.
4. Estimate the probability (0–100) that the outcome is "Yes".
5. Justify your estimate with 1–2 clear sentences.

NEWS ARTICLES:
{articles_text}

REASONING STEPS:
1. Write down any additional relevant information that is not included above. This should be specific facts that you already know the answer to, rather than information that needs to be looked up.
2. Analyze the evidence in the articles.
3. Determine whether they suggest the event is likely to happen.
4. Estimate the probability (0–100) that the outcome is "Yes".
5. Justify your estimate with 1–2 clear sentences.


RESPONSE FORMAT (JSON):
{{
  "probability_yes": <integer 0-100>,
  "justification": "<brief explanation>"
}}
""".strip()


def query_deepseek(prompt, event):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
        "response_format": "json",
    }

    url = "https://openrouter.ai/api/v1/chat/completions"
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    output = response.json()

    raw_text = output["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"[Warning] Failed to parse JSON for event {event['id']}: {e}")
        raise

    return {
        "prompt": prompt,
        "raw_response": output,
        "output_parsed": parsed
    }



def evaluate_event_experiments(event):
    experiments = []
    cutoffs = compute_cutoff_dates(event)
    for k, cutoff in cutoffs.items():
        prompt = build_event_prompt(event, cutoff)
        response_data = query_deepseek(prompt, event)
        experiments.append({
            "event_id": event["id"],
            "experiment": k,
            "pred_date": cutoff.strftime("%Y%m%d%H%M%S"),
            "title": event["title"],
            "description": event["description"],
            "start_date": event["start_date"],
            "end_date": event["end_date"],
            "outcome_prices": event["outcome_prices"],
            "prediction": {
                "probability_yes": response_data["output_parsed"]["probability_yes"],
                "justification": response_data["output_parsed"]["justification"],
                "usage": response_data["raw_response"]["usage"],
                "meta": response_data["raw_response"]["text"]
            }
        })
    return experiments


def process_events(events):
    results = []
    for event in tqdm(events, desc="Processing events"):
        try:
            experiments = evaluate_event_experiments(event)
            results.append({
                "event_id": event["id"],
                "title": event["title"],
                "description": event["description"],
                "start_date": event["start_date"],
                "end_date": event["end_date"],
                "outcome_prices": event["outcome_prices"],
                "predictions": [
                    {
                        "experiment": exp["experiment"],
                        "pred_date": exp["pred_date"],
                        "probability_yes": exp["prediction"]["probability_yes"],
                        "justification": exp["prediction"]["justification"],
                        "usage": exp["prediction"]["usage"],
                        "meta": exp["prediction"]["meta"]
                    }
                    for exp in experiments
                ]
            })
            if DEBUG:
                for exp in experiments:
                    print(
                        f"\033[94m[Event {exp['event_id']} | Exp {exp['experiment']}]\033[0m "
                        f"\033[92m{exp['prediction']['probability_yes']}% yes\033[0m - "
                        f"\033[93m{exp['prediction']['justification']}\033[0m"
                    )
                    print()
        except Exception as e:
            if DEBUG:
                print(
                    f"\033[91m[Error] Event {event.get('id', 'unknown')}:\033[0m {str(e)}"
                )
                print()
            results.append({"event_id": event.get("id", "unknown"), "error": str(e)})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Input JSON file with events")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSON file")
    args = parser.parse_args()
    
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = data if isinstance(data, list) else [data]
    if len(events) > 400:
        events = random.sample(events, 400)
    results = process_events(events)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
