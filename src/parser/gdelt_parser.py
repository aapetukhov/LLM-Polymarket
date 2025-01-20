import requests
import os
import json
from datetime import datetime


class GDELTRetriever:
    def __init__(self, save_path="./gdelt_results"):
        self.api_url = "https://api.gdeltproject.org/api/v2/doc/doc"  # base API URL
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def build_query(
        self,
        query,
        mode="ArtList",
        timespan=None,
        startdatetime=None,
        enddatetime=None,
        format="JSON",
        sort="HybridRel",
        **kwargs
    ):
        """
        Args:
            - query (str): Search query.
            - mode (str): e.g. 'ArtList', 'TimelineVol', 'TimelineTone', 'ToneChart'. Defaults to 'ArtList'.
            - timespan (str): Timespan for search (e.g., '30d' for 30 days). (no need to pass if startdatetime
            is already passed)
            - startdatetime (str): in 'YYYYMMDDHHMMSS' format.
            - enddatetime (str): in 'YYYYMMDDHHMMSS' format.
            - format (str): 'JSON', 'CSV', etc. Defaults to 'JSON'.
            - sort (str): Sort by ... (e.g. HybridRel - relevance, DateDesc - by date in descending order,
            ToneDesc - by tone in descending order).
            - **kwargs: Additional GDELT API parameters.

        Returns:
            dict: Query parameters.
        """
        params = {
            "query": query,
            "mode": mode,
            "format": format,
            "sort": sort
        }
        if timespan:
            params["timespan"] = timespan
        if startdatetime:
            params["startdatetime"] = startdatetime
        if enddatetime:
            params["enddatetime"] = enddatetime
        params.update(kwargs)
        return params

    def fetch_results(self, params):
        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            if params.get("format", "JSON").upper() == "JSON":
                return response.json()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching: {e}")
            return None

    def save_results(
        self,
        data,
        query,
        format="JSON"
    ):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        query_normalized = query.replace(" ", "_").replace("/", "-")[:50]
        file_name = f"{query_normalized}_{timestamp}.{format.lower()}"
        file_path = os.path.join(self.save_path, file_name)

        try:
            if format.upper() == "JSON":
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(data)
            print(f"Results saved to {file_path}")
            return file_path
        except IOError as e:
            print(f"Error saving file: {e}")
            return None

    def retrieve(
        self,
        query,
        mode="ArtList",
        format="JSON",
        **kwargs,
    ):
        """
        Args:
            - query (str): Search query.
            - mode (str): e.g. 'ArtList', 'TimelineVol', 'TimelineTone', 'ToneChart'. Defaults to 'ArtList'.
            - timespan (str): Timespan for search (e.g., '30d' for 30 days). (no need to pass if startdatetime
            is already passed)
            - startdatetime (str): in 'YYYYMMDDHHMMSS' format.
            - enddatetime (str): in 'YYYYMMDDHHMMSS' format.
            - format (str): 'JSON', 'CSV', etc. Defaults to 'JSON'.
            - sort (str): Sort by ... (e.g. HybridRel - relevance, DateDesc - by date in descending order,
            ToneDesc - by tone in descending order).
            - **kwargs: Additional GDELT API parameters.

        Returns:
            dict or str: Retrieved data.
        """
        params = self.build_query(query, mode=mode, format=format, **kwargs)
        data = self.fetch_results(params)
        if data:
            self.save_results(data, query, format)
        return data
