import json
import urllib.error
import urllib.request

from aiohttp import web
from server import PromptServer

from .lora_dataset.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


routes = PromptServer.instance.routes


def _model_ids(payload):
    if not isinstance(payload, dict):
        return []
    records = payload.get("models") or payload.get("data") or []
    if not isinstance(records, list):
        return []
    values = []
    for record in records:
        if not isinstance(record, dict):
            continue
        value = record.get("name") or record.get("id")
        if value:
            values.append(str(value))
    return sorted(set(values), key=str.casefold)


def _read_json(url, headers):
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


@routes.post("/lora_dataset_caption_suite/models")
async def available_caption_models(request):
    """Return model IDs advertised by an OpenAI-compatible or Ollama API."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    api_url = str(body.get("api_url", "")).strip().rstrip("/")
    api_key = str(body.get("api_key", "")).strip()
    if not api_url:
        return web.json_response(
            {"ok": False, "error": "No api_url provided"}, status=400
        )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        if "nano-gpt.com" in api_url.casefold():
            headers["X-API-Key"] = api_key

    openai_error = None
    try:
        models = _model_ids(_read_json(f"{api_url}/models", headers))
        if models:
            return web.json_response({"ok": True, "models": models})
    except urllib.error.HTTPError as error:
        if error.code != 404:
            openai_error = f"HTTP {error.code}: {error.reason}"
    except Exception as error:
        openai_error = str(error)

    ollama_base = api_url[:-3] if api_url.endswith("/v1") else api_url
    try:
        models = _model_ids(
            _read_json(
                f"{ollama_base}/api/tags",
                {"Content-Type": "application/json"},
            )
        )
        return web.json_response({"ok": True, "models": models})
    except Exception as error:
        detail = str(error)
        if openai_error:
            detail = f"OpenAI-compatible endpoint: {openai_error}; Ollama endpoint: {detail}"
        return web.json_response({"ok": False, "error": detail}, status=502)
