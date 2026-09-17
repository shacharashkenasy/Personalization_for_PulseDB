"""Provider-neutral JSON agent. No server launchers or provider-specific runtime."""

from __future__ import annotations
import importlib
import json
import os
import time
import urllib.request
from pathlib import Path
from .config import stable_hash
from .validators import _json


class Agent:
    def __init__(self, config, audit_path, seed=1337):
        self.config = config
        self.audit_path = Path(audit_path)
        self.seed = seed
        if config["provider"] not in {"openai_compatible", "custom"}:
            raise ValueError("llm.provider must be openai_compatible or custom")
        if config["provider"] == "custom" and not config.get("custom_callable"):
            raise ValueError("custom provider requires llm.custom_callable")

    def _request(self, prompt, routing):
        cfg = self.config
        request = {
            "model": cfg["model"],
            "messages": [
                {
                    "role": "system",
                    "content": "Return one valid JSON object matching the requested contract.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": cfg[
                "routing_temperature" if routing else "proposal_temperature"
            ],
            "max_tokens": cfg["max_tokens"],
        }
        if cfg["send_seed"]:
            request["seed"] = self.seed
        if cfg["json_mode"]:
            request["response_format"] = {"type": "json_object"}
        if cfg["provider"] == "custom":
            module, function = cfg["custom_callable"].split(":", 1)
            response = getattr(importlib.import_module(module), function)(request, cfg)
            return response if isinstance(response, str) else json.dumps(response)
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(cfg["api_key_env"], "")
        if key:
            headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(
            cfg["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(request).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=cfg["timeout_seconds"]) as handle:
            response = json.load(handle)
        choice = response["choices"][0]
        if choice.get("finish_reason") not in {"stop", None}:
            raise ValueError(f"Incomplete generation: {choice.get('finish_reason')}")
        return choice["message"]["content"]

    def check_ready(self):
        """Fail before expensive Train fitting if the configured endpoint is unusable."""

        def validate(value):
            if value != {"ready": True}:
                raise ValueError('Readiness response must be {"ready": true}')
            return value

        self.call(
            kind="endpoint_readiness",
            prompt='Return exactly {"ready": true}.',
            evidence={},
            validator=validate,
        )

    def call(self, *, kind, prompt, evidence, validator, routing=False):
        grounded = prompt + "\nEVIDENCE_JSON:\n" + json.dumps(evidence, sort_keys=True)
        errors = []
        attempts = self.config["routing_attempts" if routing else "proposal_attempts"]
        for attempt in range(attempts):
            raw = None
            started = time.monotonic()
            try:
                retry = (
                    ("\nCorrect the previous contract error: " + errors[-1])
                    if errors
                    else ""
                )
                raw = self._request(grounded + retry, routing)
                answer = validator(_json(raw))
                self.record(
                    {
                        "kind": kind,
                        "status": "complete",
                        "prompt": grounded,
                        "evidence_hash": stable_hash(evidence),
                        "response": answer,
                        "raw_response": raw,
                        "attempt": attempt + 1,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
                return answer
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                self.record(
                    {
                        "kind": kind,
                        "status": "failed",
                        "prompt": grounded,
                        "attempt": attempt + 1,
                        "error": errors[-1],
                        "raw_response": raw,
                    }
                )
        raise RuntimeError(f"{kind} failed: " + "; ".join(errors))

    def record(self, value):
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "provider": self.config["provider"],
                        "model": self.config["model"],
                        **value,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
