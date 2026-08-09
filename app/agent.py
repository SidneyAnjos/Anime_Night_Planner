"""The Anime Night Planner agent.

A small, self-contained ReAct loop (Thought / Action / Action Input / Observation /
Final Answer) on top of ChatDatabricks (Mosaic AI foundation chat model) and the tools in
`tools.py`. Kept framework-light on purpose: it needs no tool-calling API from the model, handles
multi-argument tools via JSON action inputs, and is easy to follow in a demo. It both READS
(semantic search, trending, details, history) and WRITES (watchlist, ratings).
"""
import json
import os
import re

from langchain_community.chat_models import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import tools

CHAT_ENDPOINT = os.environ.get("CHAT_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")

SYSTEM_PROMPT = """You are the Anime Night Planner agent. You help a group of friends pick what to
watch tonight and keep track of their watchlist and ratings.

Use the following format:

Thought: you should always think about what to do
Action: the action to take, must be exactly one of: {tool_names}
Action Input: a JSON object with the tool's arguments, e.g. {{"query": "emotional sci-fi", "max_episodes": 12}}
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer for the user

Rules:
- NEVER re-recommend an anime the group has already watched or queued. Call get_group_history for
  the group before recommending, and exclude those titles from your suggestions.
- For recommendation requests, always start with search_anime_semantic. Honor any episode-length
  preference (e.g. "short", "12 episodes") by passing max_episodes, and any genre preference by
  passing genre.
- Use fetch_trending_anime only for live/trending/currently-airing questions.
- When the group agrees on a pick, call add_to_watchlist with the anime_id from the search result
  and the group's id (the group_id the user is in, default 1).
- When the group rates a show, call log_group_rating with a score of 1-10.
- Answer in a friendly tone. When you give options, give 3-5 with brief reasoning.
"""


def _render_tools():
    lines = []
    for t in tools.TOOLS:
        args = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in t.args.items())
        lines.append(f"- {t.name}({args}): {t.description}")
    return "\n".join(lines)


def _parse_action(text):
    """Extract the LAST Action / Action Input block from a model response."""
    action = None
    action_input = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^\s*Action:\s*(.+?)\s*$", lines[i])
        if m:
            action = m.group(1).strip()
            collected = []
            j = i + 1
            while j < len(lines):
                lj = lines[j].strip()
                if re.match(r"^(Thought|Action|Final Answer):", lj):
                    break
                collected.append(lj)
                j += 1
            action_input = " ".join(collected).strip()
            i = j
        else:
            i += 1
    return action, action_input


class Agent:
    def __init__(self, llm=None, max_steps=8):
        self.llm = llm or ChatDatabricks(endpoint=CHAT_ENDPOINT, temperature=0.3, max_tokens=800)
        self.max_steps = max_steps
        self.registry = {t.name: t for t in tools.TOOLS}

    def _system_message(self):
        names = ", ".join(self.registry.keys())
        return SystemMessage(SYSTEM_PROMPT.format(tool_names=names) + "\n\nAvailable tools:\n" + _render_tools())

    def run(self, user_input, chat_history=None):
        messages = [self._system_message()]
        for turn in (chat_history or [])[-6:]:
            role = turn.get("role")
            content = str(turn.get("content", ""))
            if role == "user":
                messages.append(HumanMessage(content))
            elif role == "assistant":
                messages.append(AIMessage(content))
        messages.append(HumanMessage(user_input))

        scratchpad = []
        for _ in range(self.max_steps):
            response = self.llm.invoke(messages + scratchpad)
            text = str(response.content)

            action, action_input = _parse_action(text)
            final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)

            if action and action_input:
                # If a Final Answer appears after this action block, prefer it.
                if final_match and text.rfind(action_input) < final_match.start():
                    return final_match.group(1).strip()
                scratchpad.append(AIMessage(text))
                result = self._execute(action, action_input)
                scratchpad.append(HumanMessage(f"Observation: {result[:2000]}"))
                continue

            if final_match:
                return final_match.group(1).strip()

            scratchpad.append(AIMessage(text))
            scratchpad.append(HumanMessage(
                "You did not pick a tool or give a final answer. Pick one of the available tools, "
                "or output 'Final Answer: <answer>'."
            ))

        return "I could not reach a final answer within the step limit — try being more specific."

    def _execute(self, action, action_input):
        tool = self.registry.get(action)
        if tool is None:
            return f"Unknown tool '{action}'. Available tools: {', '.join(self.registry)}"
        try:
            args = json.loads(action_input)
        except json.JSONDecodeError:
            # Fallback: single-argument tools accept the raw string.
            first_arg = next(iter(tool.args), None)
            args = {first_arg: action_input} if first_arg else {}
        try:
            return str(tool.invoke(args))
        except Exception as exc:  # noqa: BLE001 - surface tool errors to the model
            return f"Tool '{action}' failed: {exc!r}"
