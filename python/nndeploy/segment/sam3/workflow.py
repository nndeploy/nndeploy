"""Explicit workflow registration, resource binding and command-line execution."""

from dataclasses import asdict
import json

import nndeploy.dag
from . import graphs  # register the package's Nodes and Graphs
from .nodes import _InferenceNode
from .runtime import RuntimeOwner
from .types import ModelConfig, SAM3Error


def bind_runtime(graph, owner):
    """Attach one explicit process-local owner to all SAM3 nodes before init."""
    count = 0
    for index in range(graph.get_node_count()):
        node = graph.get_node(index)
        if isinstance(node, _InferenceNode):
            if node.get_initialized():
                raise SAM3Error("already_initialized", "Bind resources before Graph.init")
            config = asdict(owner.config)
            if node.model_config and node.model_config != config:
                raise SAM3Error("config_mismatch", "Workflow model config differs from the supplied owner")
            node.owner, node.model_config = owner, config
            count += 1
        elif isinstance(node, nndeploy.dag.Graph):
            count += bind_runtime(node, owner)
    if not count:
        raise SAM3Error("invalid_workflow", "Workflow contains no SAM3 inference nodes")
    return count


def load_workflow(path, owner):
    graph = nndeploy.dag.Graph("")
    graph.load_file(str(path))
    bind_runtime(graph, owner)
    return graph


def server_factory(workflow_path, owner):
    """Use with the existing Server's graph_factory and thread isolation.

    The application owns the RuntimeOwner and must close it after Server stops.
    A process-isolated Worker requires a separately constructed owner and cannot
    share references created by this factory.
    """
    def build():
        return load_workflow(workflow_path, owner)
    return build


def execute(graph, request):
    """Run an initialized Graph. The caller serializes access to this Graph."""
    graph.get_input(0).set(request)
    status = graph.run()
    if status != nndeploy.base.Status.ok():
        raise RuntimeError(str(status))
    return graph.get_output(0).get_graph_output()


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--export-dir", help="Generate the three JSON workflows without loading weights")
    parser.add_argument("--workflow", help="Load a saved workflow")
    parser.add_argument("--kind", choices=["concept", "interact", "video"], default="concept")
    parser.add_argument("--requests", help="JSONL requests; one owner/context persists across lines")
    parser.add_argument("--output", help="Write JSONL responses")
    args = parser.parse_args()
    config = ModelConfig(args.checkpoint, args.device)
    with RuntimeOwner(config, args.result_dir) as owner:
        choices = {"concept": graphs.SAM3ImageConceptGraph, "interact": graphs.SAM3ImageInteractiveGraph, "video": graphs.SAM3VideoGraph}
        if args.export_dir:
            directory = Path(args.export_dir); directory.mkdir(parents=True, exist_ok=True)
            for kind, cls in choices.items():
                graph = cls("sam3_" + kind, owner=owner)
                graph.save_file(str(directory / f"sam3_{kind}.json"))
            return
        if not args.requests:
            parser.error("--requests or --export-dir is required")
        graph = load_workflow(args.workflow, owner) if args.workflow else choices[args.kind]("sam3_" + args.kind, owner=owner)
        graph.init()
        try:
            import sys
            with open(args.requests, encoding="utf-8") as source:
                target = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
                try:
                    for line in source:
                        target.write(json.dumps(execute(graph, json.loads(line))) + "\n")
                        target.flush()
                finally:
                    if args.output:
                        target.close()
        finally:
            graph.deinit()


if __name__ == "__main__":
    main()
