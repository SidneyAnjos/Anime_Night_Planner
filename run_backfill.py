import os, json, base64, time
from databricks.sdk import WorkspaceClient
w = WorkspaceClient(profile="SidneyAnjos")
sv = w.secrets.get_secret(scope="movie_night_planner", key="tmdb_api_key")
raw = getattr(sv, "value", None)
key = base64.b64decode(raw).decode().strip()
os.environ["TMDB_API_KEY"] = key

import sys; sys.path.insert(0, "app")
import db as db_mod, vectorstore, tools
database = db_mod.Database()
vs = vectorstore.VectorStore()
tools.init(database, vs)

import tmdb_fetch
tmdb_fetch.ENRICHMENT_LIMIT = 10
print("Running backfill with ENRICHMENT_LIMIT=10...")
result = tools.backfill_bronze_from_tmdb()
print("Backfill result:", json.dumps(result, indent=2, default=str))