import asyncio
import json
import os
from threading import Thread

from pydantic import BaseModel, Field


class FactorOutput(BaseModel):
    """One semantic factor returned by the discovery agent.

    Example:
        name="AI Infrastructure" describes one residual risk candidate.
    """

    id: str
    name: str
    description: str
    kind: str
    tickers: list[str]
    evidence: list[str]


class FactorListOutput(BaseModel):
    """Discovery output shape.

    Example:
        {"factors": []} means no strong semantic factor was found.
    """

    factors: list[FactorOutput] = Field(default_factory=list)


class MembershipOutput(BaseModel):
    """One binary semantic membership row.

    Example:
        member=1 means the stock belongs to the candidate factor basket.
    """

    ticker: str
    member: int
    reason: str


class MembershipListOutput(BaseModel):
    """Membership output shape.

    Example:
        {"memberships": []} means no stocks were classified as members.
    """

    memberships: list[MembershipOutput] = Field(default_factory=list)


class SemanticLLMClient:
    """Agents SDK JSON client for semantic discovery.

    Example:
        client = SemanticLLMClient(model="gpt-5.4")
        client.complete_json("Return JSON.", {"ticker": "NVDA"})
        returns a Python dict parsed from the model response.
    """

    def __init__(self, model=None, api_key=None, timeout=None):
        self.model = model or os.getenv("OPENFACTOR_SEMANTIC_MODEL", "gpt-5.4")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout or float(os.getenv("OPENFACTOR_SEMANTIC_TIMEOUT", "300"))

    def complete_json(self, instructions, payload):
        """Return one JSON object from system instructions and a JSON payload.

        Example:
            complete_json("Return {'ok': true}.", {}) returns {"ok": True}.
        """
        return run_sync(self.complete_json_async(instructions, payload))

    async def complete_json_async(self, instructions, payload):
        """Return one JSON object using Agents SDK web search.

        Example:
            await complete_json_async("Return JSON.", {}) returns a dict.
        """
        from agents import Agent, Runner, WebSearchTool, set_default_openai_key

        if self.api_key:
            set_default_openai_key(self.api_key)

        output_type = output_schema(instructions)
        agent = Agent(
            name="OpenFactor Semantic Discovery",
            instructions=instructions,
            model=self.model,
            tools=[WebSearchTool(user_location={"type": "approximate", "country": "US"})],
            output_type=output_type,
        )
        result = await asyncio.wait_for(
            Runner.run(
                agent,
                json.dumps(payload, default=str),
                max_turns=8,
            ),
            timeout=self.timeout,
        )
        output = result.final_output_as(output_type, raise_if_incorrect_type=True)
        return output.model_dump(exclude_none=True)


def run_sync(coro):
    """Run an async Agents call from sync OpenFactor code.

    Example:
        run_sync(client.complete_json_async(...)) returns the JSON dict.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result = {}

    def target():
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as error:
            result["error"] = error

    thread = Thread(target=target)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


def output_schema(instructions):
    """Return the strict output schema for one prompt.

    Example:
        membership prompts use MembershipListOutput.
    """
    return MembershipListOutput if "memberships" in str(instructions) else FactorListOutput
