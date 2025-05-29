import asyncio
import aiohttp
import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = "deepseek/deepseek-chat"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROMPT_TEMPLATE = """
You are given a market title and description. Your task is to:

1. Identify the **main geographical location(s)** the market refers to (country, state, city if relevant).
2. Identify **relevant categories** from the following list: [Science, Technology, Research, Healthcare, Biology, Economics, Business, Finance, Crypto, Politics, Geopolitics, Education, Military, War, Sports, Culture, Election, Mentions, Other].

Output ONLY in the following format (JSON, strict):

{{
  "locations": ["<LOCATION_1>", "<LOCATION_2>", ...],
  "categories": ["<CATEGORY_1>", "<CATEGORY_2>", ...]
}}

Market info:
Title: "{title}"
Description: "{description}"
"""

async def fetch_result(session, market, semaphore):
    prompt = PROMPT_TEMPLATE.format(title=market["title"], description=market["description"])
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
        "provider": {"only": ["DeepInfra", "NovitaAI", "Nebius AI Studio"], "require_parameters": False}
    }

    async with semaphore:
        try:
            async with session.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload, timeout=30) as resp:
                if resp.status != 200:
                    logging.error(f"Failed response {resp.status} for market: {market['title']}")
                    return None
                data = await resp.json()
                raw_output = data["choices"][0]["message"]["content"]
                parsed = safe_json_parse(raw_output)
                return {
                    "title": market["title"],
                    "description": market["description"],
                    "locations": parsed.get("locations", []),
                    "categories": parsed.get("categories", []),
                    "raw_response": raw_output
                }
        except Exception as e:
            logging.exception(f"Exception for market: {market['title']}")
            return None

def safe_json_parse(s):
    try:
        start = s.find("{")
        end = s.rfind("}") + 1
        json_part = s[start:end]
        return json.loads(json_part)
    except Exception:
        return {"locations": [], "categories": []}

async def main(markets):
    semaphore = asyncio.Semaphore(5)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_result(session, market, semaphore) for market in markets]
        results = await asyncio.gather(*tasks)

        with open("results.jsonl", "w") as f:
            for r in results:
                if r:
                    f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    with open("/kaggle/input/polymarket_resolved/all_markets.json") as f:
        markets = json.load(f)
    asyncio.run(main(markets))
