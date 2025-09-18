import os
import yaml
from functools import lru_cache
RULES_PATH = os.getenv("RULES_PATH", os.path.join(os.path.dirname(__file__), "rules.yml"))

@lru_cache(maxsize=1)
def load_rules() -> dict:
    try:
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


