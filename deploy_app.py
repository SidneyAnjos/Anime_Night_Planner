"""Deploy the Movie Night Planner Databricks App correctly.

WHY THIS EXISTS: the Databricks Apps "Deploy" button and the CLI
(`databricks apps deploy <name>`) create a deployment with NO command and NO env
vars, so the app runs `python agent.py`, exits in ~2s, and reports "app exited
unexpectedly". The app also crashes on first page load (RuntimeError:
SQL_WAREHOUSE_PATH is not set) unless env vars are carried in the deployment.
This script does the full verified deploy:
  1. Sync the local app/ files to the bundle workspace path.
  2. Resolve the TMDB key from the secret scope (the SDK does NOT resolve
     `{{secrets/...}}` refs like the bundle does).
  3. deploy_and_wait with command + env_vars.

Run from the repo root:  .venv/Scripts/python.exe deploy_app.py
(uses the `SidneyAnjos` profile — or set DATABRICKS_CONFIG_PROFILE.)
"""
import base64
import os
import sys
import urllib.request
import urllib.error

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import AppDeployment, AppDeploymentMode, EnvVar
from datetime import timedelta

APP_NAME = "movie-night-planner"
# The 6 env vars the app needs at runtime (mirrors the `env:` block in databricks.yml).
ENV = [
    ("CATALOG", "movie_night_planner"),
    ("SCHEMA", "default"),
    ("SQL_WAREHOUSE_PATH", "/sql/1.0/warehouses/eb31d64c1ca0b603"),
    ("EMBED_ENDPOINT", "databricks-bge-large-en"),
    ("CHAT_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct"),
]


def main():
    w = WorkspaceClient()
    app = w.apps.get(APP_NAME)
    src = app.default_source_code_path
    print(f"App: {APP_NAME} | source: {src}")

    # 1. Sync local app/ -> bundle workspace path (plain source, NOT notebooks).
    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
    files = sorted(os.listdir(local_dir))
    files = [f for f in files if os.path.isfile(os.path.join(local_dir, f))]
    for f in files:
        with open(os.path.join(local_dir, f), "rb") as fh:
            w.workspace.upload(f"{src}/{f}", fh, overwrite=True,
                               format=__import__(
                                   "databricks.sdk.service.workspace",
                                   fromlist=["ImportFormat"]).ImportFormat.AUTO)
    print(f"Synced {len(files)} files from app/")

    # 2. Resolve the TMDB key from the secret scope.
    sv = w.secrets.get_secret(scope="movie_night_planner", key="tmdb_api_key")
    raw = getattr(sv, "value", None)
    tmdb_key = base64.b64decode(raw).decode().strip() if raw else ""
    if not tmdb_key:
        print("WARNING: could not read TMDB_API_KEY from the secret scope")
    env_vars = [EnvVar(name=n, value=v) for n, v in ENV] + [
        EnvVar(name="TMDB_API_KEY", value=tmdb_key)]

    # 3. Deploy with command + env_vars.
    d = w.apps.deploy_and_wait(
        APP_NAME,
        app_deployment=AppDeployment(
            source_code_path=src,
            mode=AppDeploymentMode.SNAPSHOT,
            command=["streamlit", "run", "app.py"],
            env_vars=env_vars,
        ),
        timeout=timedelta(minutes=15),
    )
    print("Deployment:", d.deployment_id, "| state:", d.status.state.value,
          "| msg:", (d.status.message or "")[:80])

    a = w.apps.get(APP_NAME)
    app_state = a.app_status.state.value if a.app_status and a.app_status.state else "?"
    print("app_state:", app_state, "| url:", a.url)

    if app_state == "RUNNING":
        req = urllib.request.Request(a.url, method="GET")
        try:
            urllib.request.urlopen(req, timeout=30)
            print("HTTP GET -> 200")
        except urllib.error.HTTPError as e:
            print("HTTP GET ->", e.code, e.reason)
    else:
        sys.exit(f"App not RUNNING (state={app_state}) — check the deployment logs.")


if __name__ == "__main__":
    main()
