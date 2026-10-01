"""
LLM layer check: normal routing, fallback, <think> cleaning and error handling.

Usage (from the project root, with the VPN connected):
    python -m scripts.check_llm

Spends about 2 Gemini requests.
"""
import dataclasses
import logging
import sys

import httpx
from openai import OpenAI

from rag.llm import LLMError, LLMUnavailable, Provider, clean_text, generate, get_providers
from scripts.check_env import run_check

MESSAGES = [
    {"role": "system", "content": "You are a helpful corporate assistant. Always answer in Turkish."},
    {"role": "user", "content": "In one short sentence, say that you are ready to help employees."},
]


def unreachable(provider: Provider) -> Provider:
    """The same provider, pointed at a closed local port to simulate an outage."""
    client = OpenAI(
        api_key="unused",
        base_url="http://127.0.0.1:9/v1",
        timeout=httpx.Timeout(5.0, connect=2.0),
        max_retries=0,
    )
    return dataclasses.replace(provider, client=client)


def check_cleaning() -> str:
    """Offline test of the <think> block cleaner."""
    cases = {
        "<think>internal notes</think>\nFinal answer.": "Final answer.",
        "<THINK>\n\n</THINK>Hello": "Hello",
        "Plain answer": "Plain answer",
        "<think>cut off while thinking": "",
    }
    for raw, expected in cases.items():
        cleaned = clean_text(raw)
        if cleaned != expected:
            raise AssertionError(f"clean_text({raw!r}) returned {cleaned!r}, expected {expected!r}")
    return f"{len(cases)} cases passed"


def check_normal_route() -> str:
    """On a normal day the answer task must be served by Gemini."""
    result = generate("answer", MESSAGES)
    if result.provider != "gemini" or result.used_fallback:
        raise AssertionError(f"Expected Gemini without fallback, got {result.summary()}")
    return f"{result.summary()} -> {result.text!r}"


def check_fallback() -> str:
    """If Gemini is unreachable, the same task must be served by Qwen, without <think> text."""
    providers = dict(get_providers())
    providers["gemini"] = unreachable(providers["gemini"])
    result = generate("answer", MESSAGES, providers=providers)
    if result.provider != "qwen" or not result.used_fallback:
        raise AssertionError(f"Expected Qwen as fallback, got {result.summary()}")
    if "<think>" in result.text.lower():
        raise AssertionError("The Qwen answer still contains a <think> block.")
    return f"{result.summary()} -> {result.text!r}"


def check_no_fallback_on_our_mistake() -> str:
    """A wrong model name is our bug: the layer must stop instead of hiding it behind a fallback."""
    providers = dict(get_providers())
    providers["gemini"] = dataclasses.replace(providers["gemini"], model="model-that-does-not-exist")
    try:
        result = generate("answer", MESSAGES, providers=providers)
    except LLMUnavailable as exc:
        raise AssertionError(f"Expected a direct error, but the layer tried to fall back: {exc}")
    except LLMError as exc:
        return f"stopped without fallback: {str(exc)[:120]}"
    raise AssertionError(f"Expected an error, but got an answer: {result.summary()}")


def check_all_down() -> str:
    """If every provider is unreachable, the caller must get one clear error."""
    providers = {name: unreachable(provider) for name, provider in get_providers().items()}
    try:
        generate("answer", MESSAGES, providers=providers)
    except LLMUnavailable as exc:
        return f"clear error: {exc}"
    raise AssertionError("Expected LLMUnavailable.")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="       (log) %(message)s")
    checks = {
        "Think cleaning": check_cleaning,
        "Normal route": check_normal_route,
        "Fallback to Qwen": check_fallback,
        "No fallback on our mistake": check_no_fallback_on_our_mistake,
        "All providers down": check_all_down,
    }
    results = [run_check(name, check) for name, check in checks.items()]
    print(f"\n{sum(results)}/{len(results)} checks passed.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()