import httpx
import json
from typing import List, Optional, Union

from src.polymarket.polymarket import Polymarket
from src.utils.objects import Market, PolymarketEvent, ClobReward, Tag


class GammaMarketClient:
    def __init__(self):
        self.gamma_url = "https://gamma-api.polymarket.com"
        self.gamma_markets_endpoint = self.gamma_url + "/markets"
        self.gamma_events_endpoint = self.gamma_url + "/events"

    def parse_pydantic_market(self, market_object: dict) -> Market:
        try:
            if "clobRewards" in market_object:
                clob_rewards: list[ClobReward] = []
                for clob_rewards_obj in market_object["clobRewards"]:
                    clob_rewards.append(ClobReward(**clob_rewards_obj))
                market_object["clobRewards"] = clob_rewards

            if "events" in market_object:
                events: list[PolymarketEvent] = []
                for market_event_obj in market_object["events"]:
                    events.append(self.parse_nested_event(market_event_obj))
                market_object["events"] = events

            # These two fields below are returned as stringified lists from the api
            if "outcomePrices" in market_object:
                market_object["outcomePrices"] = json.loads(
                    market_object["outcomePrices"]
                )
            if "clobTokenIds" in market_object:
                market_object["clobTokenIds"] = json.loads(
                    market_object["clobTokenIds"]
                )

            return Market(**market_object)
        except Exception as err:
            print(f"[parse_market] Caught exception: {err}")
            print("exception while handling object:", market_object)

    # Event parser for events nested under a markets api response
    def parse_nested_event(self, event_object: dict) -> PolymarketEvent:
        print("[parse_nested_event] called with:", event_object)
        try:
            if "tags" in event_object:
                print("tags here", event_object["tags"])
                tags: list[Tag] = []
                for tag in event_object["tags"]:
                    tags.append(Tag(**tag))
                event_object["tags"] = tags

            return PolymarketEvent(**event_object)
        except Exception as err:
            print(f"[parse_event] Caught exception: {err}")
            print("\n", event_object)

    def parse_pydantic_event(self, event_object: dict) -> PolymarketEvent:
        try:
            if "tags" in event_object:
                print("tags here", event_object["tags"])
                tags: list[Tag] = []
                for tag in event_object["tags"]:
                    tags.append(Tag(**tag))
                event_object["tags"] = tags
            return PolymarketEvent(**event_object)
        except Exception as err:
            print(f"[parse_event] Caught exception: {err}")

    def get_markets(
        self, querystring_params={}, parse_pydantic=False, local_file_path=None
    ) -> List[Market]:
        if parse_pydantic and local_file_path is not None:
            raise Exception(
                'Cannot use "parse_pydantic" and "local_file" params simultaneously.'
            )

        response = httpx.get(self.gamma_markets_endpoint, params=querystring_params)
        if response.status_code == 200:
            data = response.json()
            if local_file_path is not None:
                with open(local_file_path, "w+") as out_file:
                    json.dump(data, out_file)
            elif not parse_pydantic:
                return data
            else:
                markets: list[Market] = []
                for market_object in data:
                    markets.append(self.parse_pydantic_market(market_object))
                return markets
        else:
            print(f"Error response returned from api: HTTP {response.status_code}")
            raise Exception()

    def get_events(
        self, querystring_params={}, parse_pydantic=False, local_file_path=None
    ) -> List[PolymarketEvent]:
        if parse_pydantic and local_file_path is not None:
            raise Exception(
                'Cannot use "parse_pydantic" and "local_file" params simultaneously.'
            )

        response = httpx.get(self.gamma_events_endpoint, params=querystring_params)
        if response.status_code == 200:
            data = response.json()
            if local_file_path is not None:
                with open(local_file_path, "w+") as out_file:
                    json.dump(data, out_file)
            elif not parse_pydantic:
                return data
            else:
                events: list[PolymarketEvent] = []
                for market_event_obj in data:
                    events.append(self.parse_pydantic_event(market_event_obj))
                return events
        else:
            raise Exception()

    def get_binary_events(self, querystring_params=None, local_file_path=None) -> List[PolymarketEvent]:
        """
        Fetches binary events - events with a single market that has two outcomes.

        Args:
            querystring_params (dict, optional): Parameters for filtering events:
            - `id` (int) - ID of a specific event (can be passed multiple times to select multiple events).
            - `slug` (str) - Unique event `slug` (can be passed multiple times).
            - `archived` (bool) - Filter by archived events (`True`/`False`).
            - `active` (bool) - Filter by active events (`True`/`False`).
            - `closed` (bool) - Filter by closed events (`True`/`False`).
            - `liquidity_min` (float) - Minimum liquidity.
            - `liquidity_max` (float) - Maximum liquidity.
            - `volume_min` (float) - Minimum trading volume.
            - `volume_max` (float) - Maximum trading volume.
            - `start_date_min` (str) - Minimum start date (ISO format: `"YYYY-MM-DD"`).
            - `start_date_max` (str) - Maximum start date (ISO format).
            - `end_date_min` (str) - Minimum end date (ISO format).
            - `end_date_max` (str) - Maximum end date (ISO format).
            - `tag` (str) - Filter by tag name.
            - `tag_id` (int) - Filter by tag ID.
            - `related_tags` (bool) - Includes related tags (requires `tag_id`).
            - `tag_slug` (str) - Filter by tag `slug`.
            local_file_path (str, optional): File path to save results as JSON

        Returns:
            List[PolymarketEvent]: List of binary events
        """
        querystring_params = querystring_params or {}
        # not saving to file here
        events = self.get_events(querystring_params=querystring_params, parse_pydantic=True)
        events = [event for event in events if event]  # somehow i see nans there :(

        binary_events = [
            event for event in events
            if event.markets and len(event.markets) == 1 and event.markets[0] and len(event.markets[0].outcomes) == 2
        ]

        # save if file path provided
        if local_file_path:
            with open(local_file_path, "w", encoding="utf-8") as f:
                json.dump([e.model_dump() for e in binary_events], f, ensure_ascii=False, indent=4)

        return binary_events

    def get_all_markets(self, limit=2) -> List[Market]:
        return self.get_markets(querystring_params={"limit": limit})

    def get_all_events(self, limit=2) -> List[PolymarketEvent]:
        return self.get_events(querystring_params={"limit": limit})

    def get_current_markets(self, limit=4) -> List[Market]:
        return self.get_markets(
            querystring_params={
                "active": True,
                "closed": False,
                "archived": False,
                "limit": limit,
            }
        )

    def get_all_current_markets(self, limit=100) -> List[Market]:
        offset = 0
        all_markets = []
        while True:
            params = {
                "active": True,
                "closed": False,
                "archived": False,
                "limit": limit,
                "offset": offset,
            }
            market_batch = self.get_markets(querystring_params=params)
            all_markets.extend(market_batch)

            if len(market_batch) < limit:
                break
            offset += limit

        return all_markets

    def get_current_events(self, limit=4) -> List[PolymarketEvent]:
        return self.get_events(
            querystring_params={
                "active": True,
                "closed": False,
                "archived": False,
                "limit": limit,
            }
        )

    def get_clob_tradable_markets(self, limit=2) -> List[Market]:
        return self.get_markets(
            querystring_params={
                "active": True,
                "closed": False,
                "archived": False,
                "limit": limit,
                "enableOrderBook": True,
            }
        )

    def get_market(self, market_id: int) -> dict:
        url = self.gamma_markets_endpoint + "/" + str(market_id)
        print(url)
        response = httpx.get(url)
        return response.json()
