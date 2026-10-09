"""Small deterministic parsers for YAML, the demo worker pattern and log format."""

import ast
from datetime import datetime
import logging
from pathlib import Path
import re

import yaml
from yaml.nodes import MappingNode, ScalarNode

from core.incident_loader import IncidentFile
from core.schemas import Evidence, SourceReference

logger = logging.getLogger(__name__)


def _source(file: IncidentFile, start: int, end: int | None = None) -> SourceReference:
    end = end or start
    return SourceReference(
        file_path=file.file_path, line_start=start, line_end=end,
        raw_excerpt="\n".join(file.content.splitlines()[start - 1:end]),
    )


def extract_config(file: IncidentFile) -> list[Evidence]:
    loader = yaml.SafeLoader(file.content)
    evidence = []
    active_nodes = set()

    def visit(node: MappingNode, prefix: str = "") -> None:
        if node.tag != "tag:yaml.org,2002:map":
            raise ValueError("Unsupported YAML mapping tag")
        if id(node) in active_nodes:
            raise ValueError("Recursive YAML mappings are unsupported")
        active_nodes.add(id(node))
        seen = set()
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in seen:
                raise ValueError("YAML keys must be unique strings")
            seen.add(key)
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value_node, MappingNode):
                visit(value_node, path)
            elif isinstance(value_node, ScalarNode):
                value = loader.construct_object(value_node)
                if value is not None and not isinstance(value, (str, int, float, bool)):
                    raise ValueError(f"Unsupported YAML scalar at {path}")
                evidence.append(Evidence(
                    id=f"{Path(file.file_path).name}:{path}", category="config_value",
                    statement=f"{path} = {value!r}", value=value,
                    source=_source(file, key_node.start_mark.line + 1,
                                   value_node.end_mark.line + (value_node.end_mark.column > 0)),
                ))
            else:
                raise ValueError(f"Only nested mappings and scalar config values are supported: {path}")
        active_nodes.remove(id(node))

    try:
        root = loader.get_single_node()
        if not isinstance(root, MappingNode):
            raise ValueError("Incident YAML must be a mapping")
        # Validate all tags and resolve aliases through SafeLoader before walking.
        loader.construct_document(root)
        visit(root)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {file.file_path}: {exc}") from exc
    finally:
        loader.dispose()
    return evidence


def extract_code(file: IncidentFile) -> list[Evidence]:
    try:
        tree = ast.parse(file.content)
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python source in {file.file_path}: {exc.msg}") from exc
    evidence = []

    def calls(node: ast.AST):
        # Nested functions are definitions, not proof that the worker calls them.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "load_model":
                yield node
        for child in ast.iter_child_nodes(node):
            yield from calls(child)

    for worker in tree.body:
        if isinstance(worker, (ast.FunctionDef, ast.AsyncFunctionDef)) and worker.name == "worker_main":
            for statement in worker.body:
                for call in calls(statement):
                    evidence.append(Evidence(
                        id=f"{Path(file.file_path).name}:L{call.lineno}:worker_model_load",
                        category="code_behavior", value="worker_model_load",
                        statement="worker_main calls load_model() in its worker lifecycle.",
                        source=_source(file, worker.lineno, worker.end_lineno),
                    ))
    return evidence


def extract_logs(file: IncidentFile) -> list[Evidence]:
    evidence = []
    for line_number, line in enumerate(file.content.splitlines(), 1):
        timestamp_match = re.match(r"^(\d{4}-\d{2}-\d{2}T\S+)\s+", line)
        timestamp = timestamp_match.group(1) if timestamp_match else None
        if timestamp:
            try:
                datetime.fromisoformat(timestamp)
            except ValueError as exc:
                raise ValueError(f"Invalid log timestamp at {file.file_path}:{line_number}") from exc

        def add(kind: str, category: str, statement: str, value, unit=None):
            evidence.append(Evidence(
                id=f"{Path(file.file_path).name}:L{line_number}:{kind}",
                category=category, statement=statement, value=value, unit=unit,
                timestamp=timestamp, source=_source(file, line_number),
            ))

        worker = re.search(r"\b(worker-\d+)\s+(starting|started)\b", line)
        if worker:
            add("worker_start", "log_event", f"Worker startup: {worker.group(1)}", worker.group(1))
        memory = re.search(r"\bgpu_memory_used_gib=(\d+(?:\.\d+)?)\b", line)
        if memory:
            add("memory", "system_metric", "Observed GPU memory usage", float(memory.group(1)), "GiB")
        for field, kind, statement in (
            ("batch_allocation_gib", "batch_allocation", "Observed batch allocation"),
            ("batch_headroom_gib", "batch_headroom", "Observed pre-batch GPU memory headroom"),
        ):
            metric = re.search(rf"\b{field}=(\d+(?:\.\d+)?)\b", line)
            if metric:
                add(kind, "system_metric", statement, float(metric.group(1)), "GiB")
        if re.search(r"out of memory|allocation failure|\bOOM\b", line, re.IGNORECASE):
            add("oom", "log_event", "Observed OOM/allocation failure", "allocation_failure")
        if re.search(r"service (?:startup )?(?:failed|failure)\b", line, re.IGNORECASE):
            add("service_failure", "log_event", "Observed service startup failure", "service_failure")
    return evidence


def extract_evidence(files: list[IncidentFile]) -> list[Evidence]:
    evidence = []
    for file in files:
        suffix = Path(file.file_path).suffix
        if suffix in {".yaml", ".yml"}:
            evidence.extend(extract_config(file))
        elif suffix == ".py":
            evidence.extend(extract_code(file))
        elif suffix in {".log", ".txt"}:
            evidence.extend(extract_logs(file))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    logger.info("Extracted %d observed evidence items", len(evidence))
    return evidence
