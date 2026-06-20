import json
import os


class SemanticLLMClient:
    """Small JSON client for semantic discovery.

    Example:
        client = SemanticLLMClient(model="gpt-5-mini")
        client.complete_json("Return JSON.", {"ticker": "NVDA"})
        returns a Python dict parsed from the model response.
    """

    def __init__(self, model=None):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model or os.getenv("OPENFACTOR_SEMANTIC_MODEL", "gpt-5-mini")

    def complete_json(self, instructions, payload):
        """Return one JSON object from system instructions and a JSON payload.

        Example:
            complete_json("Return {'ok': true}.", {}) returns {"ok": True}.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content or "{}")
