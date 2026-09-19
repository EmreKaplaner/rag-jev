"""Export the same schema served at /openapi.json, without requiring an API key."""

import json
from pathlib import Path

from rag_jev.server import create_app

Path("openapi.json").write_text(json.dumps(create_app().openapi(), indent=2) + "\n")
