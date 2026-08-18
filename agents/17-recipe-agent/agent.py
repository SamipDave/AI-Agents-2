import re
import json


def parse_json_response(text: str) -> dict:
    """Parse a possibly fenced JSON response and return the object.

    This strips triple-backtick fences (``` or ```json```) if present, then
    searches for the first JSON object in the cleaned text. To avoid
    pathological performance on very large responses, the regex search is
    limited to the first 100k characters.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    # Limit the search space for performance, while keeping a greedy JSON match
    search_prefix = cleaned[:100_000]
    match = re.search(r"\{.*\}", search_prefix, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)
