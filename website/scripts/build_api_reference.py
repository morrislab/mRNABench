"""Build the website API reference from Python source signatures."""

from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "website"
    / "src"
    / "data"
    / "generated"
    / "api-reference.json"
)

REFERENCE_GROUPS = [
    {
        "title": "Load and configure",
        "description": (
            "Top-level functions used before running an evaluation."
        ),
        "entries": [
            (
                "mrna_bench",
                "load_dataset",
                "mrna_bench/loader/loader.py",
                None,
            ),
            (
                "mrna_bench",
                "load_model",
                "mrna_bench/loader/loader.py",
                None,
            ),
            ("mrna_bench", "update_data_path", "mrna_bench/utils.py", None),
            ("mrna_bench", "get_data_path", "mrna_bench/utils.py", None),
            (
                "mrna_bench",
                "update_model_weights_path",
                "mrna_bench/utils.py",
                None,
            ),
            (
                "mrna_bench",
                "get_model_weights_path",
                "mrna_bench/utils.py",
                None,
            ),
        ],
    },
    {
        "title": "Datasets and splits",
        "description": (
            "Dataset metadata, paired variants, and train-test splits."
        ),
        "entries": [
            (
                "mrna_bench.datasets",
                "DatasetMetadata",
                "mrna_bench/datasets/benchmark_dataset.py",
                "__dataclass__",
            ),
            (
                "mrna_bench.datasets",
                "BenchmarkDataset.get_splits",
                "mrna_bench/datasets/benchmark_dataset.py",
                "get_splits",
            ),
            (
                "mrna_bench.datasets",
                "BenchmarkDataset.get_vep_pairs",
                "mrna_bench/datasets/benchmark_dataset.py",
                "get_vep_pairs",
            ),
        ],
    },
    {
        "title": "Models and embeddings",
        "description": "Model outputs and dataset-wide embedding generation.",
        "entries": [
            (
                "mrna_bench.models",
                "EmbeddingModel.embed",
                "mrna_bench/models/embedding_model.py",
                "embed",
            ),
            (
                "mrna_bench.models",
                "EmbeddingModel.sequence_score",
                "mrna_bench/models/embedding_model.py",
                "sequence_score",
            ),
            (
                "mrna_bench.models",
                "EmbeddingModel.masked_marginal_llr",
                "mrna_bench/models/embedding_model.py",
                "masked_marginal_llr",
            ),
            (
                "mrna_bench.embedder",
                "DatasetEmbedder",
                "mrna_bench/embedder/dataset_embedder.py",
                "__init__",
            ),
            (
                "mrna_bench.embedder",
                "DatasetEmbedder.from_dataframe",
                "mrna_bench/embedder/dataset_embedder.py",
                "from_dataframe",
            ),
            (
                "mrna_bench.embedder",
                "DatasetEmbedder.embed_dataset",
                "mrna_bench/embedder/dataset_embedder.py",
                "embed_dataset",
            ),
            (
                "mrna_bench.embedder",
                "DatasetEmbedder.persist_embeddings",
                "mrna_bench/embedder/dataset_embedder.py",
                "persist_embeddings",
            ),
            (
                "mrna_bench.embedder",
                "DatasetEmbedder.merge_embeddings",
                "mrna_bench/embedder/dataset_embedder.py",
                "merge_embeddings",
            ),
        ],
    },
    {
        "title": "Linear probes",
        "description": (
            "Builder configuration, fitted probes, and seeded runs."
        ),
        "entries": [
            (
                "mrna_bench.linear_probe",
                "LinearProbeBuilder",
                "mrna_bench/linear_probe/linear_probe_builder.py",
                "__init__",
            ),
            (
                "mrna_bench.linear_probe",
                "LinearProbeBuilder.fetch_embedding_by_embedding_instance",
                "mrna_bench/linear_probe/linear_probe_builder.py",
                "fetch_embedding_by_embedding_instance",
            ),
            (
                "mrna_bench.linear_probe",
                "LinearProbeBuilder.build_splitter",
                "mrna_bench/linear_probe/linear_probe_builder.py",
                "build_splitter",
            ),
            (
                "mrna_bench.linear_probe",
                "LinearProbeBuilder.set_target",
                "mrna_bench/linear_probe/linear_probe_builder.py",
                "set_target",
            ),
            (
                "mrna_bench.linear_probe",
                "LinearProbeBuilder.build_evaluator",
                "mrna_bench/linear_probe/linear_probe_builder.py",
                "build_evaluator",
            ),
            (
                "mrna_bench.linear_probe",
                "LinearProbeBuilder.set_regressor",
                "mrna_bench/linear_probe/linear_probe_builder.py",
                "set_regressor",
            ),
            (
                "mrna_bench.linear_probe",
                "LinearProbeBuilder.use_persister",
                "mrna_bench/linear_probe/linear_probe_builder.py",
                "use_persister",
            ),
            (
                "mrna_bench.linear_probe",
                "LinearProbeBuilder.build",
                "mrna_bench/linear_probe/linear_probe_builder.py",
                "build",
            ),
            (
                "mrna_bench.linear_probe",
                "LinearProbe.run_linear_probe",
                "mrna_bench/linear_probe/linear_probe.py",
                "run_linear_probe",
            ),
            (
                "mrna_bench.linear_probe",
                "LinearProbe.linear_probe_multirun",
                "mrna_bench/linear_probe/linear_probe.py",
                "linear_probe_multirun",
            ),
            (
                "mrna_bench.linear_probe",
                "LinearProbe.get_fit_model",
                "mrna_bench/linear_probe/linear_probe.py",
                "get_fit_model",
            ),
        ],
    },
    {
        "title": "Variant effects",
        "description": "Embedding-difference and likelihood-based VEP.",
        "entries": [
            (
                "mrna_bench.zeroshot",
                "ZeroShotVEP.from_embeddings",
                "mrna_bench/zeroshot/vep.py",
                "from_embeddings",
            ),
            (
                "mrna_bench.zeroshot",
                "ZeroShotVEP.from_model",
                "mrna_bench/zeroshot/vep.py",
                "from_model",
            ),
            (
                "mrna_bench.zeroshot",
                "ZeroShotVEP.run",
                "mrna_bench/zeroshot/vep.py",
                "run",
            ),
        ],
    },
    {
        "title": "Fine-tuning",
        "description": (
            "Optional task heads, LoRA adapters, and training loops."
        ),
        "entries": [
            (
                "mrna_bench.fine_tune",
                "TaskHead",
                "mrna_bench/fine_tune/task_heads.py",
                "__init__",
            ),
            (
                "mrna_bench.fine_tune",
                "FineTuneWrapper",
                "mrna_bench/fine_tune/fine_tune_wrapper.py",
                "__init__",
            ),
            (
                "mrna_bench.fine_tune",
                "FineTuneWrapper.apply_lora",
                "mrna_bench/fine_tune/fine_tune_wrapper.py",
                "apply_lora",
            ),
            (
                "mrna_bench.fine_tune",
                "TrainerConfig",
                "mrna_bench/fine_tune/trainer.py",
                "__dataclass__",
            ),
            (
                "mrna_bench.fine_tune",
                "FineTuneTrainer",
                "mrna_bench/fine_tune/trainer.py",
                "__init__",
            ),
            (
                "mrna_bench.fine_tune",
                "FineTuneTrainer.fit",
                "mrna_bench/fine_tune/trainer.py",
                "fit",
            ),
            (
                "mrna_bench.fine_tune",
                "FineTuneTrainer.evaluate",
                "mrna_bench/fine_tune/trainer.py",
                "evaluate",
            ),
            (
                "mrna_bench.fine_tune",
                "create_dataloaders",
                "mrna_bench/fine_tune/dataloader.py",
                None,
            ),
        ],
    },
]

SUMMARY_OVERRIDES = {
    "load_dataset": "Load a registered dataset.",
    "load_model": "Load a registered model and set it to inference mode.",
    "update_data_path": (
        "Set the directory used for processed data, embeddings, and results."
    ),
    "get_data_path": "Return the configured data directory.",
    "update_model_weights_path": (
        "Set the directory used for downloaded model weights."
    ),
    "get_model_weights_path": "Return the configured model-weights directory.",
    "DatasetMetadata": (
        "Metadata used to select targets, splits, and evaluation routes."
    ),
    "DatasetEmbedder": (
        "Generate and save embeddings for a dataset or sequence dataframe."
    ),
    "LinearProbeBuilder": (
        "Configure embeddings, splits, targets, estimators, and persistence."
    ),
    "TaskHead": (
        "Prediction head for regression, classification, or multilabel tasks."
    ),
    "FineTuneWrapper": (
        "Combine an EmbeddingModel with a task head and optional "
        "LoRA adapters."
    ),
    "FineTuneTrainer": "Train and evaluate a FineTuneWrapper.",
    "TrainerConfig": "Options used by FineTuneTrainer.",
}


def _find_class(tree: ast.Module, class_name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise ValueError(f"Class {class_name!r} was not found.")


def _find_function(
    nodes: list[ast.stmt],
    function_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in nodes:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ):
            return node
    raise ValueError(f"Function {function_name!r} was not found.")


def _annotation(node: ast.expr | None) -> str:
    return ast.unparse(node) if node is not None else ""


def _format_signature(
    display_name: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    drop_first: bool,
) -> str:
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults: list[ast.expr | None] = [
        *([None] * (len(positional) - len(node.args.defaults))),
        *node.args.defaults,
    ]
    if drop_first and positional and positional[0].arg in {"self", "cls"}:
        positional = positional[1:]
        defaults = defaults[1:]

    parts: list[str] = []
    positional_only_count = len(node.args.posonlyargs)
    if drop_first and node.args.posonlyargs:
        positional_only_count -= 1

    for index, (argument, default) in enumerate(zip(positional, defaults)):
        rendered = argument.arg
        annotation = _annotation(argument.annotation)
        if annotation:
            rendered += f": {annotation}"
        if default is not None:
            rendered += f" = {ast.unparse(default)}"
        parts.append(rendered)
        if positional_only_count and index + 1 == positional_only_count:
            parts.append("/")

    if node.args.vararg is not None:
        rendered = f"*{node.args.vararg.arg}"
        annotation = _annotation(node.args.vararg.annotation)
        if annotation:
            rendered += f": {annotation}"
        parts.append(rendered)
    elif node.args.kwonlyargs:
        parts.append("*")

    for argument, default in zip(
        node.args.kwonlyargs,
        node.args.kw_defaults,
    ):
        rendered = argument.arg
        annotation = _annotation(argument.annotation)
        if annotation:
            rendered += f": {annotation}"
        if default is not None:
            rendered += f" = {ast.unparse(default)}"
        parts.append(rendered)

    if node.args.kwarg is not None:
        rendered = f"**{node.args.kwarg.arg}"
        annotation = _annotation(node.args.kwarg.annotation)
        if annotation:
            rendered += f": {annotation}"
        parts.append(rendered)

    signature = f"{display_name}({', '.join(parts)})"
    returns = _annotation(node.returns)
    if returns and node.name != "__init__":
        signature += f" -> {returns}"
    return signature


def _format_dataclass_signature(
    class_name: str,
    node: ast.ClassDef,
) -> str:
    if node.bases:
        raise ValueError(
            f"Dataclass inheritance is not supported for {class_name}."
        )

    class_kw_only = False
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        if ast.unparse(decorator.func).split(".")[-1] != "dataclass":
            continue
        options = {
            keyword.arg: keyword.value
            for keyword in decorator.keywords
            if keyword.arg is not None
        }
        if (
            "init" in options
            and isinstance(options["init"], ast.Constant)
            and options["init"].value is False
        ):
            raise ValueError(f"{class_name} has dataclass init disabled.")
        if (
            "kw_only" in options
            and isinstance(options["kw_only"], ast.Constant)
        ):
            class_kw_only = bool(options["kw_only"].value)

    positional_fields = []
    keyword_only_fields = []
    for child in node.body:
        if not isinstance(child, ast.AnnAssign):
            continue
        if not isinstance(child.target, ast.Name):
            continue
        annotation = _annotation(child.annotation)
        if annotation.split(".")[-1].startswith("ClassVar["):
            continue

        default = child.value
        field_kw_only = class_kw_only
        if (
            isinstance(default, ast.Call)
            and ast.unparse(default.func).split(".")[-1] == "field"
        ):
            options = {
                keyword.arg: keyword.value
                for keyword in default.keywords
                if keyword.arg is not None
            }
            if (
                "init" in options
                and isinstance(options["init"], ast.Constant)
                and options["init"].value is False
            ):
                continue
            if (
                "kw_only" in options
                and isinstance(options["kw_only"], ast.Constant)
            ):
                field_kw_only = bool(options["kw_only"].value)
            if "default" in options:
                default = options["default"]
            elif "default_factory" in options:
                factory = ast.unparse(options["default_factory"])
                raise ValueError(
                    f"{class_name}.{child.target.id} uses default_factory "
                    f"{factory}; add an explicit signature formatter."
                )
            else:
                default = None

        rendered = f"{child.target.id}: {annotation}"
        if default is not None:
            rendered += f" = {ast.unparse(default)}"
        if field_kw_only:
            keyword_only_fields.append(rendered)
        else:
            positional_fields.append(rendered)

    fields = positional_fields
    if keyword_only_fields:
        fields = [*fields, "*", *keyword_only_fields]
    return f"{class_name}({', '.join(fields)})"


def _parse_docstring(
    docstring: str | None,
) -> tuple[str, list[dict[str, str]], str]:
    if not docstring:
        return "", [], ""
    lines = inspect.cleandoc(docstring).splitlines()
    section_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line in {"Args:", "Returns:", "Raises:"}
        ),
        len(lines),
    )
    summary_lines = []
    for line in lines[:section_index]:
        if not line.strip() and summary_lines:
            break
        if line.strip():
            summary_lines.append(line.strip())
    summary = " ".join(summary_lines)

    parameters: list[dict[str, str]] = []
    returns = ""
    current_section = ""
    current_parameter: dict[str, str] | None = None
    return_lines: list[str] = []
    for line in lines[section_index:]:
        stripped = line.strip()
        if stripped in {"Args:", "Returns:", "Raises:"}:
            current_section = stripped[:-1].lower()
            current_parameter = None
            continue
        if not stripped:
            continue
        if current_section == "args":
            match = re.match(
                r"^(\*{0,2}[A-Za-z_][A-Za-z0-9_]*)"
                r"(?:\s+\([^)]*\))?:\s*(.*)$",
                stripped,
            )
            if match:
                current_parameter = {
                    "name": match.group(1),
                    "description": match.group(2),
                }
                parameters.append(current_parameter)
            elif current_parameter is not None:
                current_parameter["description"] += f" {stripped}"
        elif current_section == "returns":
            return_lines.append(stripped)
    if return_lines:
        returns = " ".join(return_lines)
    return summary, parameters, returns


def _build_entry(spec: tuple[str, str, str, str | None]) -> dict[str, Any]:
    module_name, display_name, relative_path, member_name = spec
    source_path = REPOSITORY_ROOT / relative_path
    tree = ast.parse(
        source_path.read_text(encoding="utf-8"),
        filename=str(source_path),
    )

    class_name = display_name.split(".", 1)[0] if member_name else None
    if member_name == "__dataclass__":
        assert class_name is not None
        class_node = _find_class(tree, class_name)
        signature = _format_dataclass_signature(class_name, class_node)
        docstring = ast.get_docstring(class_node)
    elif member_name is not None:
        assert class_name is not None
        class_node = _find_class(tree, class_name)
        function_node = _find_function(class_node.body, member_name)
        call_name = (
            class_name
            if member_name == "__init__"
            else display_name
        )
        signature = _format_signature(call_name, function_node, True)
        docstring = ast.get_docstring(function_node)
        class_docstring = (
            ast.get_docstring(class_node)
            if member_name == "__init__"
            else None
        )
    else:
        function_node = _find_function(tree.body, display_name)
        signature = _format_signature(display_name, function_node, False)
        docstring = ast.get_docstring(function_node)

    summary, parameters, returns = _parse_docstring(docstring)
    if member_name == "__init__" and class_docstring:
        class_summary, class_parameters, _ = _parse_docstring(
            class_docstring
        )
        summary = class_summary
        parameter_names = {
            parameter["name"]
            for parameter in parameters
        }
        parameters.extend(
            parameter
            for parameter in class_parameters
            if parameter["name"] not in parameter_names
        )
    summary = SUMMARY_OVERRIDES.get(display_name, summary)
    return {
        "name": display_name,
        "module": module_name,
        "signature": signature,
        "summary": summary,
        "parameters": parameters,
        "returns": returns,
        "sourcePath": relative_path,
    }


def main() -> None:
    """Write the generated API reference JSON."""
    groups = []
    for group in REFERENCE_GROUPS:
        groups.append({
            "title": group["title"],
            "description": group["description"],
            "entries": [
                _build_entry(spec)
                for spec in group["entries"]
            ],
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output:
        json.dump({"groups": groups}, output, indent=2)
        output.write("\n")
    entry_count = sum(len(group["entries"]) for group in groups)
    print(f"Built API reference: {entry_count} callables.")


if __name__ == "__main__":
    main()
