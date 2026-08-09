import os, json, base64, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout, Disposition, Format, StatementState

w = WorkspaceClient(profile="SidneyAnjos")
wid = "/sql/1.0/warehouses/eb31d64c1ca0b603".split("/")[-1]

# Run the fixed DDL for raw_reviews
sql = """CREATE TABLE IF NOT EXISTS `movie_night_planner`.`default`.`raw_reviews` (
    tmdb_id BIGINT,
    review_id STRING,
    source_url STRING,
    fetched_at TIMESTAMP,
    payload STRING
) USING DELTA"""
resp = w.statement_execution.execute_statement(
    warehouse_id=wid, statement=sql, disposition=Disposition.INLINE, format=Format.JSON_ARRAY,
    wait_timeout=ExecuteStatementRequestOnWaitTimeout.SECONDS_30
)
while resp.status and resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
    time.sleep(1); resp = w.statement_execution.get_statement(resp.statement_id)
print("DDL result:", resp.status.state, resp.status.error if resp.status.state != StatementState.SUCCEEDED else "OK")