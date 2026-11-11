from unittest.mock import Mock, patch

import nndeploy.base as base
from nndeploy.dag.base import name_to_io_type
from nndeploy.dag.graph_runner import GraphRunner


def test_pipeline_collects_all_iterations_in_order():
    runner = GraphRunner()
    graph = Mock()
    ok = base.Status(base.StatusCode.Ok)
    graph.init.return_value = graph.run.return_value = graph.deinit.return_value = ok
    graph.get_parallel_type.return_value = base.ParallelType.Pipeline
    graph.get_loop_count.return_value = 3
    output, consumer = Mock(), Mock()
    graph.get_all_output.return_value = [output]
    output.get_graph_output.side_effect = ["first", "second", "third"]
    output.get_consumers.return_value = [consumer]
    consumer.get_name.return_value = "sink"
    consumer.get_io_type.return_value = name_to_io_type["Text"]
    with patch.object(runner, "_build_graph", return_value=(graph, ok)):
        _, results, status, _ = runner.run("{}", "pipeline", "test")
    assert status == base.StatusCode.Ok
    assert results == {"sink": {"Text": ["first", "second", "third"]}}
