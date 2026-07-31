#!/usr/bin/env python3
"""Persistent process for one Python benchmark adapter."""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Type

from common import BenchmarkAdapter
from common.adapter import clear_cached_inputs


PROTOCOL_PREFIX = "__PYSIGLIB_BENCHMARK_WORKER__"


def load_adapter_module(script_path: Path) -> ModuleType:
    module_name = f"benchmark_adapter_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load adapter module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def find_adapter_class(module: ModuleType) -> Type[BenchmarkAdapter]:
    candidates = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and value is not BenchmarkAdapter
        and issubclass(value, BenchmarkAdapter)
        and value.__module__ == module.__name__
    ]
    if len(candidates) != 1:
        names = [candidate.__name__ for candidate in candidates]
        raise RuntimeError(
            f"Expected one BenchmarkAdapter subclass in {module.__file__}, got {names}"
        )
    return candidates[0]


def send_response(response: dict) -> None:
    print(PROTOCOL_PREFIX + json.dumps(response), flush=True)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python_adapter_worker.py ADAPTER_SCRIPT", file=sys.stderr)
        return 2

    try:
        script_path = Path(sys.argv[1]).resolve()
        adapter_class = find_adapter_class(load_adapter_module(script_path))
    except Exception as error:
        send_response({
            "status": "startup_error",
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        return 1

    send_response({
        "status": "ready",
        "adapter": adapter_class.__name__,
        "worker_scope": getattr(adapter_class, "WORKER_SCOPE", "backend"),
    })
    for line in sys.stdin:
        try:
            config = json.loads(line)
            config["_call_event_callback"] = send_response
            try:
                result = adapter_class(config)._run_benchmark()
            finally:
                if config.get("clear_input_caches_after_task", False):
                    clear_cached_inputs()
            send_response({"status": "result", "result": result})
        except Exception as error:
            send_response({
                "status": "error",
                "error": str(error),
                "traceback": traceback.format_exc(),
            })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
