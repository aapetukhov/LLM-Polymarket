from datetime import datetime, timedelta
from newspaper import Article
from src.polymarket.gamma import GammaMarketClient

from dotenv import load_dotenv

load_dotenv()


def get_date_ranges(start_date: str, end_date: str, step_days: int = 5, window_size: int = 6):
    date_ranges = []
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    while current_start < end_date_dt:
        current_end = min(current_start + timedelta(days=window_size - 1), end_date_dt)
        date_ranges.append((current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d")))
        current_start += timedelta(days=step_days)

    return date_ranges


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
    min_date = "2024-01-01"
    max_date = "2025-02-25"
    date_ranges = get_date_ranges(min_date, max_date, 5, 6)
    gamma = GammaMarketClient()
    unique_events = set()

    for start, end in date_ranges:
        file_path = f"data/polymarket/events_{start}_to_{end}.json"
        
        events = gamma.get_binary_events(
            querystring_params={
                "limit": 1000,
                "start_date_min": start,
                "start_date_max": end,
                "active": False,
            },
            local_file_path=file_path
        )
        for event in events:
            unique_events.add(event.id)
        print(f"Fetched {len(events)} events for {start} to {end}")
        print(f"Unique events: {len(unique_events)}")


if __name__ == "__main__":
    main()
