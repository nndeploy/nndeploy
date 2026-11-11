"""Regressions for packet replacement, backpressure and Python/C++ ownership."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps
import gc
import os
import subprocess
import sys
import threading
import unittest
import weakref

import numpy as np

from nndeploy import _nndeploy_internal as native
from nndeploy.dag import Edge
from nndeploy.device import Buffer, Tensor
from nndeploy.preprocess import ConvertTo


def isolated(test):
    """Turn native crashes or a blocked GIL into a failure of one test."""
    @wraps(test)
    def run(self):
        if os.environ.get("NNDEPLOY_EDGE_TEST_WORKER") == test.__name__:
            return test(self)
        subprocess.run(
            [sys.executable, __file__, f"{type(self).__name__}.{test.__name__}"],
            env={**os.environ, "NNDEPLOY_EDGE_TEST_WORKER": test.__name__},
            timeout=15, check=True,
        )
    return run


def pipeline():
    edge = native.dag.Edge("output")
    edge.set_parallel_type(native.base.ParallelType.Pipeline)
    edge.set_queue_max_size(1)
    edge.set_queue_overflow_policy(native.base.QueueOverflowPolicy.AllBackpressure)
    edge.mark_graph_output()
    edge.construct()
    return edge


def cast_output(values):
    source, output = Edge("in"), Edge("out")
    node = ConvertTo("cast", [source], [output])
    source.set(Tensor.from_numpy(values))
    assert node.run()
    return source, output, node


class EdgeOwnership(unittest.TestCase):
    def start_blocked(self, pool, operation):
        started = threading.Event()

        def run():
            started.set()
            return operation()

        future = pool.submit(run)
        self.assertTrue(started.wait(2))
        with self.assertRaises(TimeoutError):
            future.result(timeout=0.1)
        return future

    @isolated
    def test_created_handles_survive_packet_replacement(self):
        host = native.device.get_default_host_device()
        for desc in (native.device.BufferDesc(4), Tensor.from_numpy(np.zeros(4)).get_desc()):
            edge = native.dag.Edge("fixed")
            payload = edge.create(host, desc)
            np.asarray(payload)[:] = 42
            self.assertTrue(edge.notify_written(payload))
            edge.set("next")
            np.testing.assert_array_equal(np.asarray(payload), [42] * 4)
            del payload, edge

    @isolated
    def test_tensor_alias_retains_external_buffer(self):
        for path in ("constructor", "create", "modify", "modify_owned"):
            with self.subTest(path=path):
                array = np.arange(6, dtype=np.uint8)
                buffer = Buffer.from_numpy(array)
                ref = weakref.ref(buffer)
                desc = Tensor.from_numpy(array).get_desc()
                if path == "constructor":
                    tensor = Tensor(desc, buffer)
                elif path == "create":
                    tensor = Tensor()
                    tensor.create(desc, buffer)
                else:
                    tensor = Tensor(desc)
                    tensor.just_modify(buffer=buffer, is_external=(path == "modify"))
                alias = Tensor(tensor)
                tensor.clear()
                del buffer, array, tensor
                gc.collect()
                self.assertIsNotNone(ref())
                np.testing.assert_array_equal(np.asarray(alias), np.arange(6))
                del alias
                gc.collect()
                self.assertIsNone(ref())

    @isolated
    def test_native_output_survives_next_frame(self):
        source, output, node = cast_output(np.array([17, 23], dtype=np.uint8))
        frame = output.get(node)
        source.set(Tensor.from_numpy(np.array([1], dtype=np.uint8)))
        self.assertTrue(node.run())
        del node, source, output
        np.testing.assert_array_equal(np.asarray(frame), [17., 23.])

    @isolated
    def test_republishing_native_output_releases_storage(self):
        data = np.arange(8, dtype=np.float32)
        ref = weakref.ref(data)
        source, output, node = cast_output(data)
        output.set(output.get(node))
        source.set("replace")
        del data
        gc.collect()
        self.assertIsNotNone(ref())
        output.set("replace")
        gc.collect()
        self.assertIsNone(ref())

    @isolated
    def test_fixed_edge_concurrent_replacement(self):
        edge = native.dag.Edge("fixed")
        node = native.dag.Node("sender", [], [edge])
        barrier = threading.Barrier(4)

        def produce(index):
            barrier.wait()
            for i in range(500):
                self.assertTrue(node.set_output_data((index, i)))

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(produce, range(4)))
        self.assertEqual(edge.get(None)[1], 499)

    @isolated
    def test_finalizer_can_reenter_edge(self):
        edge = pipeline()
        finalized = threading.Event()

        class Payload:
            def __del__(self):
                edge.request_terminate()
                finalized.set()

        edge.set(Payload())
        current = edge.get_graph_output()
        del current
        self.assertIsNone(edge.get_graph_output())
        self.assertTrue(finalized.is_set())

    @isolated
    def test_terminate_wakes_full_queue_writer(self):
        edge = pipeline()
        edge.set("queued")
        with ThreadPoolExecutor() as pool:
            writer = self.start_blocked(pool, lambda: edge.set("blocked"))
            self.assertTrue(edge.request_terminate())
            with self.assertRaises(RuntimeError):
                writer.result(timeout=2)
        self.assertIsNone(edge.get_graph_output())

    @isolated
    def test_create_backpressure_and_notify_written(self):
        host = native.device.get_default_host_device()
        for desc in (native.device.BufferDesc(2), Tensor.from_numpy(np.zeros(2)).get_desc()):
            edge = pipeline()
            edge.set("first")
            with ThreadPoolExecutor() as pool:
                writer = self.start_blocked(pool, lambda: edge.create(host, desc))
                self.assertEqual(edge.get_graph_output(), "first")
                reader = self.start_blocked(pool, edge.get_graph_output)
                payload = writer.result(timeout=2)
                np.asarray(payload)[:] = [7, 8]
                self.assertTrue(edge.notify_written(payload))
                received = reader.result(timeout=2)
                np.testing.assert_array_equal(np.asarray(received), [7, 8])

    @isolated
    def test_pipeline_two_consumers_and_producers(self):
        edge = pipeline()
        consumer = native.dag.Node("second_consumer", [edge], [])
        edge.increase_consumers([consumer])
        edge.construct()
        writers = 2
        count = 50

        def write(producer):
            for index in range(count):
                edge.set((producer, index))

        def read_graph():
            return [edge.get_graph_output() for _ in range(writers * count)]

        def read_node():
            results = []
            for _ in range(writers * count):
                edge.update(consumer)
                results.append(consumer.get_input_data())
            return results

        with ThreadPoolExecutor(max_workers=4) as pool:
            graph_reader = pool.submit(read_graph)
            node_reader = pool.submit(read_node)
            producers = [pool.submit(write, producer) for producer in range(writers)]
            for producer in producers:
                producer.result(timeout=5)
            graph_values = graph_reader.result(timeout=5)
            self.assertEqual(graph_values, node_reader.result(timeout=5))
        self.assertEqual(set(graph_values), {(p, i) for p in range(writers) for i in range(count)})

    @isolated
    def test_drop_oldest_finalizer_and_cancel_unwritten_packet(self):
        edge = pipeline()
        edge.set_queue_overflow_policy(native.base.QueueOverflowPolicy.DropOldest)
        finalized = threading.Event()

        class Payload:
            def __del__(self):
                edge.request_terminate()
                finalized.set()

        edge.set(Payload())
        edge.set("replacement")
        self.assertTrue(finalized.is_set())
        self.assertIsNone(edge.get_graph_output())

        edge = pipeline()
        payload = edge.create(native.device.get_default_host_device(), native.device.BufferDesc(4))
        with ThreadPoolExecutor() as pool:
            reader = self.start_blocked(pool, edge.get_graph_output)
            edge.request_terminate()
            self.assertIsNone(reader.result(timeout=2))
        del payload

    @isolated
    def test_drop_oldest_waits_for_unwritten_typed_payload(self):
        host = native.device.get_default_host_device()
        for desc in (native.device.BufferDesc(2), Tensor.from_numpy(np.zeros(2)).get_desc()):
            edge = pipeline()
            edge.set_queue_overflow_policy(native.base.QueueOverflowPolicy.DropOldest)
            first = edge.create(host, desc)
            with ThreadPoolExecutor() as pool:
                writer = self.start_blocked(pool, lambda: edge.create(host, desc))
                np.asarray(first)[:] = [3, 4]
                self.assertTrue(edge.notify_written(first))
                second = writer.result(timeout=2)
                np.asarray(second)[:] = [7, 8]
                self.assertTrue(edge.notify_written(second))
                np.testing.assert_array_equal(np.asarray(edge.get_graph_output()), [7, 8])
            np.testing.assert_array_equal(np.asarray(first), [3, 4])

            edge = pipeline()
            edge.set_queue_overflow_policy(native.base.QueueOverflowPolicy.DropOldest)
            first = edge.create(host, desc)
            with ThreadPoolExecutor() as pool:
                writer = self.start_blocked(pool, lambda: edge.create(host, desc))
                edge.request_terminate()
                with self.assertRaises(RuntimeError):
                    writer.result(timeout=2)
                np.asarray(first)[:] = [9, 10]
                self.assertTrue(edge.notify_written(first))
            self.assertIsNone(edge.get_graph_output())

    @isolated
    def test_created_packets_with_two_producers_and_consumers(self):
        edge = pipeline()
        edge.set_queue_max_size(4)
        consumer = native.dag.Node("second_consumer", [edge], [])
        edge.increase_consumers([consumer])
        edge.construct()
        host = native.device.get_default_host_device()
        desc = Tensor.from_numpy(np.zeros(2, dtype=np.int64)).get_desc()
        count = 25

        def produce(producer):
            for index in range(count):
                value = edge.create(host, desc)
                np.asarray(value)[:] = [producer, index]
                self.assertTrue(edge.notify_written(value))

        def consume_graph():
            return [tuple(np.asarray(edge.get_graph_output())) for _ in range(2 * count)]

        def consume_node():
            result = []
            for _ in range(2 * count):
                edge.update(consumer)
                result.append(tuple(np.asarray(consumer.get_input_data())))
            return result

        with ThreadPoolExecutor(max_workers=4) as pool:
            graph = pool.submit(consume_graph)
            node = pool.submit(consume_node)
            writers = [pool.submit(produce, i) for i in range(2)]
            for writer in writers:
                writer.result(timeout=5)
            values = graph.result(timeout=5)
            self.assertEqual(values, node.result(timeout=5))
        self.assertEqual(set(values), {(p, i) for p in range(2) for i in range(count)})

    @isolated
    def test_mixed_types_follow_actual_packet_order(self):
        edge = pipeline()
        first = Tensor.from_numpy(np.array([1, 2], dtype=np.int32))
        second = Buffer.from_numpy(np.array([3], dtype=np.uint8))
        edge.set(first)
        with ThreadPoolExecutor() as pool:
            writer = self.start_blocked(pool, lambda: edge.set(second))
            self.assertIs(edge.get_graph_output(), first)
            self.assertIs(edge.get_graph_output(), second)
            self.assertTrue(writer.result(timeout=2))


if __name__ == "__main__":
    unittest.main()
