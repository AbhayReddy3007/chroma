"""
llm_client.py — Unified LLM abstraction for steps 7, 8a, 8b, 9
=================================================================
Supports:
  - Gemini models (via google-genai SDK + Google Search grounding)
  - Claude models (via Vertex AI Anthropic endpoint + web_search tool,
                   authenticated using the GCS_SERVICE_ACCOUNT JSON)

Model selection via .env:
  LLM_MODEL=gemini-2.5-flash-preview-05-20     (default)
  LLM_MODEL=gemini-3.1-pro-preview
  LLM_MODEL=gemini-3.1-pro
  LLM_MODEL=claude-sonnet-4-6

API keys / auth:
  GOOGLE_API_KEY or GEMINI_API_KEY   — for Gemini models
  GCS_SERVICE_ACCOUNT                — path to service account JSON (for Claude via Vertex AI)
  VERTEX_AI_REGION                   — Vertex AI region (default: us-east5)
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────
# Model detection
# ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemini-2.5-flash-preview-05-20"
MODEL         = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip()


def is_claude() -> bool:
    return MODEL.lower().startswith("claude")


def is_gemini() -> bool:
    return not is_claude()


def get_model_name() -> str:
    return MODEL


# ─────────────────────────────────────────────────────────────
# Gemini client — lazy
# ─────────────────────────────────────────────────────────────

_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set.\n"
                "Required for Gemini models. Set it in .env."
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# ─────────────────────────────────────────────────────────────
# Claude client — lazy, via Vertex AI using GCS_SERVICE_ACCOUNT
# ─────────────────────────────────────────────────────────────

_claude_client = None

def _get_claude_client():
    """
    Initialise the Anthropic client using the GCS_SERVICE_ACCOUNT
    service account JSON file via Vertex AI.
    """
    global _claude_client
    if _claude_client is None:
        import anthropic

        sa_path = os.getenv("GCS_SERVICE_ACCOUNT")
        if not sa_path:
            raise ValueError(
                "GCS_SERVICE_ACCOUNT is not set in .env.\n"
                "Required for Claude models (Vertex AI authentication).\n"
                "Set it to the path of your service account JSON file."
            )

        if not os.path.isfile(sa_path):
            raise FileNotFoundError(
                f"Service account file not found: {sa_path}\n"
                "Check GCS_SERVICE_ACCOUNT in your .env."
            )

        # Load credentials from service account JSON
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(
            sa_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        project_id = credentials.project_id or os.getenv("BQ_PROJECT_ID", "")
        region     = os.getenv("VERTEX_AI_REGION", "us-east5")

        _claude_client = anthropic.AnthropicVertex(
            project_id = project_id,
            region     = region,
            credentials = credentials,
        )
        print(f"[LLM] Claude client via Vertex AI | project: {project_id} | region: {region}")

    return _claude_client


# ─────────────────────────────────────────────────────────────
# Unified generate function
# ─────────────────────────────────────────────────────────────

async def generate(
    prompt:            str,
    use_web_search:    bool  = False,
    temperature:       float = 0.0,
    max_output_tokens: int   = 65536,
    system:            Optional[str] = None,
) -> str:
    """
    Send a prompt to the configured LLM and return the raw text response.

    Args:
        prompt:            The user prompt text
        use_web_search:    If True, enable web search (Google Search for Gemini,
                           web_search tool for Claude)
        temperature:       Sampling temperature (0.0 = deterministic)
        max_output_tokens: Maximum response tokens
        system:            Optional system prompt (used by Claude; for Gemini
                           it's prepended to the prompt)

    Returns:
        Raw text response from the model.
    """
    if is_gemini():
        return await _generate_gemini(
            prompt, use_web_search, temperature, max_output_tokens, system
        )
    else:
        return await _generate_claude(
            prompt, use_web_search, temperature, max_output_tokens, system
        )


# ─────────────────────────────────────────────────────────────
# Gemini implementation
# ─────────────────────────────────────────────────────────────

async def _generate_gemini(
    prompt:            str,
    use_web_search:    bool,
    temperature:       float,
    max_output_tokens: int,
    system:            Optional[str],
) -> str:
    from google.genai import types

    client = _get_gemini_client()

    # Build config
    config_kwargs = {
        "temperature":      temperature,
        "max_output_tokens": max_output_tokens,
    }
    if use_web_search:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

    config = types.GenerateContentConfig(**config_kwargs)

    # Gemini doesn't have a separate system field in the basic API —
    # prepend system to prompt if provided
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    response = await client.aio.models.generate_content(
        model    = MODEL,
        contents = full_prompt,
        config   = config,
    )

    return (response.text or "").strip()


# ─────────────────────────────────────────────────────────────
# Claude implementation (via Vertex AI — synchronous client)
# ─────────────────────────────────────────────────────────────

async def _generate_claude(
    prompt:            str,
    use_web_search:    bool,
    temperature:       float,
    max_output_tokens: int,
    system:            Optional[str],
) -> str:
    import asyncio

    client = _get_claude_client()

    # Build tools list
    tools = []
    if use_web_search:
        tools.append({
            "type": "web_search_20250305",
            "name": "web_search",
        })

    # Build messages
    messages = [{"role": "user", "content": prompt}]

    # Build request kwargs
    kwargs = {
        "model":      MODEL,
        "max_tokens":  min(max_output_tokens, 16384),
        "messages":    messages,
        "temperature": temperature,
    }
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools

    # AnthropicVertex is synchronous — run in executor to avoid blocking the loop
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.messages.create(**kwargs),
    )

    # Extract text from content blocks
    text_parts = []
    for block in response.content:
        if hasattr(block, "text") and block.text:
            text_parts.append(block.text)

    return "\n".join(text_parts).strip()


# ─────────────────────────────────────────────────────────────
# JSON parsing helper
# ─────────────────────────────────────────────────────────────

def parse_json_response(raw: str) -> Optional[dict | list]:
    """
    Parse a JSON response from the LLM, stripping markdown fences.
    Returns the parsed object, or None if parsing fails.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$",          "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    if not cleaned:
        return None

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt repair — close open structures
        return _try_repair_json(cleaned)


def _try_repair_json(raw: str) -> Optional[dict | list]:
    """Attempt to recover truncated JSON by closing open structures."""
    s = raw.strip()
    quote_count = s.count('"') - s.count('\\"')
    if quote_count % 2 != 0:
        s += '"'
    opens  = s.count('[') - s.count(']')
    closes = s.count('{') - s.count('}')
    s += ']' * max(0, opens) + '}' * max(0, closes)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None
