import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


class FakeRoutes:
    def __init__(self):
        self.registered = []

    def post(self, path):
        def decorator(function):
            self.registered.append((path, function))
            return function

        return decorator


def test_comfy_entrypoint_registers_nodes_and_dedicated_model_route(monkeypatch):
    root = Path(__file__).parents[1]
    routes = FakeRoutes()
    prompt_server = SimpleNamespace(instance=SimpleNamespace(routes=routes))
    monkeypatch.setitem(sys.modules, "server", SimpleNamespace(PromptServer=prompt_server))

    package_name = "lora_dataset_caption_suite_entrypoint"
    specification = importlib.util.spec_from_file_location(
        package_name,
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    package = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, package_name, package)
    specification.loader.exec_module(package)

    assert "LoraDatasetBuilder" in package.NODE_CLASS_MAPPINGS
    assert package.WEB_DIRECTORY == "./js"
    assert [path for path, _ in routes.registered] == [
        "/lora_dataset_caption_suite/models"
    ]

    entrypoint = sys.modules[f"{package_name}.nodes"]
    assert entrypoint._model_ids(
        {"data": [{"id": "vision-b"}, {"id": "vision-a"}, {"id": "vision-a"}]}
    ) == ["vision-a", "vision-b"]
