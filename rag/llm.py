"""
Single entry point for every LLM call in CIA.

The rest of the project asks for a task (for example "answer") and never talks to a
provider directly. The routing table config.LLM_TASKS decides which provider handles the
task and which one is the fallback.
"""
import logging
import re
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache

import httpx
from openai import APIConnectionError, APIStatusError, OpenAI

import config

logger = logging.getLogger(__name__)

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    """A provider rejected the request in a way that another provider would not fix."""


class LLMUnavailable(LLMError):
    """No provider configured for the task could produce an answer."""


class EmptyResponse(LLMError):
    """The provider answered, but nothing was left after cleaning the text."""


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    client: OpenAI
    is_local: bool
    extra_body: dict | None = field(default=None)


@dataclass(frozen=True)
class LLMResult:
    text: str
    provider: str
    model: str
    elapsed_ms: float
    used_fallback: bool
    fallback_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def summary(self) -> str:
        """One-line description for logs and debugging."""
        line = f"{self.provider} ({self.model}) in {self.elapsed_ms / 1000:.1f} s"
        if self.used_fallback:
            line += f" [fallback: {self.fallback_reason}]"
        return line


@lru_cache(maxsize=1)
def get_providers() -> dict[str, Provider]:
    """Build one client per provider. Both speak the OpenAI-compatible API."""
    timeout = httpx.Timeout(config.LLM_READ_TIMEOUT, connect=config.LLM_CONNECT_TIMEOUT)
    return {
        "gemini": Provider(
            name="gemini",
            model=config.GEMINI_MODEL,
            client=OpenAI(api_key=config.GEMINI_API_KEY, base_url=config.GEMINI_BASE_URL,
                          timeout=timeout, max_retries=0),
            is_local=False,
        ),
        "qwen": Provider(
            name="qwen",
            model=config.OLLAMA_MODEL,
            client=OpenAI(api_key="ollama", base_url=config.OLLAMA_BASE_URL,
                          timeout=timeout, max_retries=0),
            is_local=True,
            extra_body={"reasoning_effort": "none"},   # ask Qwen3 not to "think" before answering
        ),
    }


def clean_text(text: str) -> str:
    """Remove <think> blocks. An unclosed block means the model was cut off while thinking."""
    text = THINK_BLOCK.sub("", text)
    unclosed = text.lower().find("<think>")
    if unclosed != -1:
        text = text[:unclosed]
    return text.strip()


def _should_fall_back(exc: Exception) -> bool:
    """Fall back only on problems another provider could avoid: outages, limits, empty answers."""
    if isinstance(exc, (APIConnectionError, EmptyResponse)):   # includes timeouts
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def _describe(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__} ({status})" if status else type(exc).__name__


def _call(provider: Provider, messages: list[dict], temperature: float) -> LLMResult:
    started = time.perf_counter()
    response = provider.client.chat.completions.create(
        model=provider.model,
        messages=messages,
        temperature=temperature,
        extra_body=provider.extra_body,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    text = clean_text(response.choices[0].message.content or "")
    if not text:
        raise EmptyResponse(f"{provider.name} returned an empty answer.")

    usage = response.usage
    return LLMResult(
        text=text,
        provider=provider.name,
        model=provider.model,
        elapsed_ms=elapsed_ms,
        used_fallback=False,
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
    )


def generate(
    task: str,
    messages: list[dict],
    *,
    providers: dict[str, Provider] | None = None,
    route: list[str] | None = None,
) -> LLMResult:
    """
    Run a task on the first provider that succeeds, following config.LLM_TASKS.

    messages: OpenAI-style chat messages, e.g. [{"role": "user", "content": "..."}].
    providers: optional override, used by tests to inject broken or fake providers.
    route: optional provider names to try instead of the task's own list, for example ["gemini"]
           in the evaluation, where a silent switch to another model would mix up the results.
    """
    if task not in config.LLM_TASKS:
        raise ValueError(f"Unknown LLM task '{task}'. Known tasks: {list(config.LLM_TASKS)}")
    settings = config.LLM_TASKS[task]
    available = providers or get_providers()
    names = route or settings["providers"]
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"Unknown LLM provider(s) {unknown}. Known providers: {list(available)}")

    failures: list[str] = []
    for name in names:
        provider = available[name]
        try:
            result = _call(provider, messages, settings["temperature"])
        except Exception as exc:
            if not _should_fall_back(exc):
                raise LLMError(f"Provider '{name}' rejected the request: {exc}") from exc
            failures.append(f"{name}: {_describe(exc)}")
            logger.warning("LLM provider '%s' failed for task '%s': %s", name, task, _describe(exc))
            continue

        if failures:
            result = replace(result, used_fallback=True, fallback_reason="; ".join(failures))
        logger.info("Task '%s' answered by %s", task, result.summary())
        return result

    raise LLMUnavailable(f"No provider could handle task '{task}': " + " | ".join(failures))
