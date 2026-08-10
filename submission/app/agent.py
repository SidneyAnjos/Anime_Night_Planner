"""The Movie Night Planner agent.

A small tool-calling loop on top of a Mosaic AI foundation model and the tools in
`tools.py`. It READS (semantic search, trending, details, history) and WRITES (watchlist,
ratings, recommendations) — the durable actions that make this a capstone-grade agent.

Why this design (and why NOT text-based ReAct): an earlier version drove the model with a
free-text `Thought:/Action:/Action Input:/Final Answer:` format and parsed it out. Llama-3.3-70b
was not reliable at that format under a low temperature — it often jumped straight to a
`Final Answer` that *claimed* it had added a movie to the watchlist while never actually
emitting a tool call, so no row was written (a silent hallucination). The Databricks Foundation
Model `chat/completions` endpoint supports OpenAI-style **native function-calling** — the model
is forced to emit structured `tool_calls` (or a plain content answer), which makes the agent
deterministic and the writes real.

The chat model is called directly via `WorkspaceClient.api_client.do("POST",
"/serving-endpoints/<name>/invocations", ...)` — the same authenticated transport the app already
uses, reusing the app SP's M2M auth in the app runtime and the local profile in dev. We do NOT
go through `ChatDatabricks` (langchain): it moved between langchain packages across versions and
the loose `langchain>=0.3` pins let pip resolve an incompatible mix (langchain-core 0.3.x
clashing with the langchain-classic/langgraph 1.x that ships in the app's base venv), which
crashed the app at startup. Talking to the endpoint directly removes that whole dependency
surface. `langchain-core` remains only for the `@tool` decorator on the tools themselves.
"""
import json
import os

import tools
from databricks.sdk import WorkspaceClient

CHAT_ENDPOINT = os.environ.get("CHAT_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
MAX_STEPS = 8
# Cap how much of a tool result is fed back to the model (tokens, not a hard safety limit).
OBSERVATION_MAX = 2000

# Map the JSON-schema-ish arg type strings produced by `langchain_core @tool` (.args) onto the
# JSON-schema types the OpenAI tool spec expects.
_ARG_TYPE = {"string": "string", "integer": "integer", "number": "number",
             "boolean": "boolean", "bool": "boolean", "array": "array", "object": "object"}

SYSTEM_PROMPT = """You are the Movie Night Planner agent. You help a group of friends pick what to \
watch tonight and keep track of their watchlist, ratings and recommendations.

Rules:
- NEVER re-recommend a movie the group has already watched, queued, or been recommended. Call
  get_group_history for the group FIRST, and exclude those movie_ids from your suggestions.
- For recommendation requests, always start with search_movies_semantic. Honor any length
  preference (e.g. "short", "under 2 hours" -> max_runtime in minutes) and any genre preference
  by passing genre. Use the movie_id returned by the search for any follow-up writes.
- Use fetch_trending_movies for "what's popular" / "what's trending" style questions.
- When the group agrees on a pick, call add_to_watchlist with the movie_id from the search result
  and the group_id (the group the user is in, default 1), then call log_recommendation to record
  the recommendation with a one-line reason.
- When the group rates a movie, call log_group_rating with a score of 1-10.
- Only report that you added a movie to the watchlist or logged a rating AFTER the tool returns
  success — never claim an action you did not take.
- Answer in a friendly tone. When you give options, give 3-5 with brief reasoning.
"""


def _build_tool_specs():
    """Build the OpenAI-style `tools` list from the langchain @tool decorators in tools.TOOLS.

    Each tool exposes `.name`, `.description`, `.args` ({arg: {"title", "type", "default"?}}).
    We map that onto {"type":"function","function":{name, description, parameters}} so the model
    gets accurate, per-tool schemas (required vs optional args, integer query/runtime args, etc.).
    """
    specs = []
    for t in tools.TOOLS:
        props = {}
        required = []
        for arg, meta in (t.args or {}).items():
            json_type = _ARG_TYPE.get((meta.get("type") or "string"), "string")
            props[arg] = {"type": json_type}
            if "title" in meta and meta["title"]:
                # description is optional in the schema; keep the human title as a lightweight hint
                props[arg]["description"] = meta["title"]
            if "default" not in meta:
                required.append(arg)
        specs.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": (t.description or "").strip(),
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })
    return specs


class _Chat:
    """Thin blocking chat wrapper around the raw Foundation Model endpoint.

    `invoke(messages, tools=None)` returns a `_ChatResult` with `.content` (str) and
    `.tool_calls` (list of {"id", "name", "arguments"}) — never raises on empty tool calls,
    since a no-tool-call response is the final-answer signal. Messages are plain dicts
    ({"role","content"} for system/user/assistant, plus {"role":"tool","tool_call_id",
    "content"} for tool responses) matching the OpenAI chat format.
    """

    def __init__(self, endpoint=CHAT_ENDPOINT, temperature=0.15, max_tokens=800):
        self.endpoint = endpoint
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = WorkspaceClient()  # auth via the app SP (M2M) in app runtime, or local profile

    def invoke(self, messages, tools=None):
        body = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        resp = self._client.api_client.do(
            "POST", f"/serving-endpoints/{self.endpoint}/invocations", body=body,
        )
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        raw_calls = msg.get("tool_calls") or []
        tool_calls = []
        for tc in raw_calls:
            fn = (tc.get("function") or {})
            args_raw = fn.get("arguments")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": tc.get("id"), "name": fn.get("name"), "arguments": args})
        return _ChatResult(content=content, tool_calls=tool_calls)


class _ChatResult:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls


class Agent:
    def __init__(self, llm=None, max_steps=MAX_STEPS):
        # Lower temperature than the old ReAct run: with native tool-calling we want the model
        # deterministic and obedient, not "creative" enough to skip tools.
        self.llm = llm or _Chat(endpoint=CHAT_ENDPOINT, temperature=0.15, max_tokens=800)
        self.max_steps = max_steps
        self.registry = {t.name: t for t in tools.TOOLS}
        self._tool_specs = _build_tool_specs()

    def run(self, user_input, chat_history=None):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in (chat_history or [])[-6:]:
            role = turn.get("role")
            content = str(turn.get("content", ""))
            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": user_input})

        for _ in range(self.max_steps):
            resp = self.llm.invoke(messages, tools=self._tool_specs)

            if resp.tool_calls:
                # Append the assistant turn WITH its tool_calls (the endpoint requires the
                # prior assistant message to carry tool_calls to pair with the tool responses).
                asst = {"role": "assistant", "content": resp.content or ""}
                asst["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"],
                                  "arguments": json.dumps(tc["arguments"], ensure_ascii=False)}}
                    for tc in resp.tool_calls
                ]
                messages.append(asst)

                for tc in resp.tool_calls:
                    result = self._execute(tc["name"], tc["arguments"])
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": result[:OBSERVATION_MAX]})
                continue

            # No tool calls -> the model produced the final answer.
            return resp.content.strip() or "I don't have an answer for that — try rephrasing."

        return "I could not reach a final answer within the step limit — try being more specific."

    def _execute(self, name, args):
        """Run a tool by name with a dict of args, surfacing errors back to the model as text."""
        tool = self.registry.get(name)
        if tool is None:
            return f"Unknown tool '{name}'. Available tools: {', '.join(self.registry)}."
        try:
            return str(tool.invoke(args))
        except Exception as exc:  # noqa: BLE001 - surface tool errors to the model
            return f"Tool '{name}' failed: {exc!r}"
