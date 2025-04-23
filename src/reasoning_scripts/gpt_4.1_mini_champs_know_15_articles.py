import re, os, json, argparse, random
from tqdm import tqdm
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
DEBUG = True
TOP_K = 15
MODEL_NAME = "gpt-4.1-mini-2025-04-14"
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
            "description": "Justification based on the provided news context."
        }
    },
    "required": ["probability_yes", "justification"],
    "additionalProperties": False
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
    end = parse_dt(min(event["end_date"], event["resolution_time"]))
    delta = end - start - timedelta(seconds=1)
    return {k: start + delta * (k / n) for k in [1, 2, 3, 4]}


def build_event_prompt(event, cutoff):
    cutoff_str = cutoff.strftime("%B %d, %Y, %H:%M")
    filtered = [a for a in event["articles"]
                if a.get("score", 0) > -3 and parse_article_dt(a["date"]) <= cutoff]
    articles = filtered[:TOP_K]
    articles_text = "\n\n".join([
        f"[{i+1}] TITLE: {a.get('title', '').strip()}\nDATE: {a.get('date', '').strip()}\nTEXT: {a.get('text', '').strip()}"
        for i, a in enumerate(articles)
    ])
    start_str = readable_date(event["start_date"])
    end_str = readable_date(event["end_date"])
    return f"""
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
  "probability_yes": <integer 0-100>,
  "justification": "<explanation>"
}}
""".strip(), len(articles)


def query_llm(prompt, event):
    response = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": prompt}],
        instructions="You are an expert geopolitical forecaster. Think like a superforecaster. Follow the reasoning framework provided.",
        text={
            "format": {
                "type": "json_schema",
                "strict": True,
                "schema": schema,
                "name": "event_forecast"
            }
        },
        temperature=0.0,
        # reasoning={"effort": "medium"},
        max_output_tokens=1024,
        metadata={"event_id": event["id"]},
        user=str(event["id"])
    )
    
    raw_text = response.output_text
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"[Warning] Failed to parse JSON for event {event['id']}: {e}")
        print("Trying to fix malformed JSON...")
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e2:
            print(f"[Error] Still failed to parse JSON after fix: {e2}")
            raise

    return {
        "prompt": prompt,
        "raw_response": response.model_dump(),
        "output_parsed": parsed
    }


def evaluate_event_experiments(event):
    experiments = []
    cutoffs = compute_cutoff_dates(event)
    for k, cutoff in cutoffs.items():
        prompt, num_articles = build_event_prompt(event, cutoff)
        response_data = query_llm(prompt, event)
        experiments.append({
            "event_id": event["id"],
            "experiment": k,
            "pred_date": cutoff.strftime("%Y%m%d%H%M%S"),
            "title": event["title"],
            "description": event["description"],
            "start_date": event["start_date"],
            "end_date": event["end_date"],
            "resolution_time": event["resolution_time"],
            "outcome_prices": event["outcome_prices"],
            "prediction": {
                "probability_yes": response_data["output_parsed"]["probability_yes"],
                "justification": response_data["output_parsed"]["justification"],
                "usage": response_data["raw_response"]["usage"],
                "meta": response_data["raw_response"]["text"],
                "num_articles": num_articles
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
                        "meta": exp["prediction"]["meta"],
                        "num_articles": exp["prediction"]["num_articles"]
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
