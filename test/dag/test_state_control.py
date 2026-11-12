import dataclasses
import gc
import json
import unittest
import weakref

import nndeploy.base
from nndeploy import _nndeploy_internal as native
from nndeploy.dag import Graph, Node, State


@dataclasses.dataclass
class RequestState(State):
    prompt: str = "default"
    step: int = 0
    logs: list = dataclasses.field(default_factory=list)
    payload: object = None
    internal_count: int = dataclasses.field(default=4, init=False)


@dataclasses.dataclass
class ExtendedState(RequestState):
    enabled: bool = True


class StateNode(Node):
    def __init__(self, name, action):
        super().__init__(name)
        self.action = action

    def run(self):
        self.action(self)
        return nndeploy.base.Status.ok()


class StateControlTests(unittest.TestCase):
    def test_defaults_inheritance_and_native_storage(self):
        state = ExtendedState(prompt="cat")
        self.assertEqual(state.prompt, "cat")
        self.assertEqual(state.step, 0)
        self.assertEqual(state.internal_count, 4)
        self.assertTrue(state.enabled)
        state._native_state.set("step", 9)
        self.assertEqual(state.step, 9)  # class default must not shadow storage
        state.step += 1
        self.assertEqual(state._native_state.get("step"), 10)
        self.assertNotIn("step", state.__dict__)
        other = ExtendedState()
        state.logs.append("first")
        self.assertEqual(other.logs, [])

    def test_field_errors_and_integer_range(self):
        state = RequestState()
        with self.assertRaises(AttributeError):
            state.stpe = 4
        with self.assertRaises(AttributeError):
            _ = state.stpe
        with self.assertRaises(AttributeError):
            del state.step
        state.step = 2**50
        self.assertEqual(state.step, 2**50)
        with self.assertRaises((RuntimeError, OverflowError, TypeError)):
            state.step = 2**100
        self.assertEqual(state.step, 2**50)

    def test_reserved_and_frozen_schema_rejected(self):
        @dataclasses.dataclass
        class Reserved(State):
            _native_state: int = 0

        @dataclasses.dataclass(frozen=True)
        class Frozen(State):
            step: int = 0

        with self.assertRaises(TypeError):
            Reserved()
        with self.assertRaises(TypeError):
            Frozen()

    def test_object_identity_and_lifetime(self):
        class Payload:
            pass

        state = RequestState()
        payload = Payload()
        reference = weakref.ref(payload)
        state.payload = payload
        self.assertIs(state.payload, payload)
        del payload
        gc.collect()
        self.assertIsNotNone(reference())
        state.payload = None
        gc.collect()
        self.assertIsNone(reference())

    def test_native_tensor_reference_is_not_copied_or_leaked(self):
        tensor = native.device.Tensor("shared")
        reference = weakref.ref(tensor)
        state = RequestState(payload=tensor)
        self.assertIs(state.payload, tensor)
        del tensor
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertEqual(state.payload.get_name(), "shared")
        state.payload = None
        gc.collect()
        self.assertIsNone(reference())

    def test_cpp_created_state_view_can_update_existing_fields(self):
        storage = native.dag._StateStorage()
        storage.set("step", 10)
        view = State._from_native(storage)
        view.step += 1
        self.assertEqual(storage.get("step"), 11)
        with self.assertRaises(AttributeError):
            view.unknown = 1

    def test_no_state_before_or_after_run_and_request_isolation(self):
        for parallel in (native.base.ParallelType.Sequential, native.base.ParallelType.Task):
            with self.subTest(parallel=parallel):
                expected = []

                def first(node):
                    self.assertIs(node.state, expected[0])
                    self.assertIs(node.get_state(), node.state)
                    node.state.step += 1
                    node.state.logs.append("first")

                def second(node):
                    self.assertEqual(node.state.step, 1)
                    node.state.step += 1
                    node.state.logs.append("second")

                a, b = StateNode("a", first), StateNode("b", second)
                graph = Graph("state")
                graph.set_parallel_type(parallel)
                graph.add_node(b)
                graph.add_node(a)
                self.assertTrue(graph.set_execution_order(["a", "b"]))
                with self.assertRaises(RuntimeError):
                    _ = a.state
                self.assertTrue(graph.init())
                try:
                    for _ in range(2):
                        state = RequestState()
                        expected[:] = [state]
                        self.assertTrue(graph.run_with_state(state))
                        self.assertEqual(state.step, 2)
                        self.assertEqual(state.logs, ["first", "second"])
                        with self.assertRaises(RuntimeError):
                            _ = a.state
                        with self.assertRaises(AttributeError):
                            a.state = state
                finally:
                    graph.deinit()

    def test_python_exception_stops_successor_and_unbinds(self):
        def fail(node):
            node.state.step = 42
            raise ValueError("failure")

        def unexpected(node):
            node.state.logs.append("unexpected")

        graph = Graph("failure")
        a, b = StateNode("a", fail), StateNode("b", unexpected)
        graph.add_node(a)
        graph.add_node(b)
        self.assertTrue(graph.set_execution_order(["a", "b"]))
        self.assertTrue(graph.init())
        state = RequestState()
        try:
            self.assertFalse(graph.run_with_state(state))
            self.assertEqual(state.step, 42)
            self.assertEqual(state.logs, [])
            with self.assertRaises(RuntimeError):
                a.get_state()
        finally:
            graph.deinit()

    def test_graph_run_override_and_nested_state_are_preserved(self):
        class CustomGraph(Graph):
            def run(self):
                self.state.logs.append(self.get_name())
                return super().run()

        child = CustomGraph("child")
        child.add_node(StateNode("leaf", lambda node: node.state.logs.append("leaf")))
        self.assertTrue(child.set_execution_order(["leaf"]))
        parent = CustomGraph("parent")
        parent.add_node(child)
        self.assertTrue(parent.set_execution_order(["child"]))
        self.assertTrue(parent.init())
        state = RequestState()
        try:
            self.assertTrue(parent.run_with_state(state))
            self.assertEqual(state.logs, ["parent", "child", "leaf"])
            with self.assertRaises(RuntimeError):
                _ = child.state
        finally:
            parent.deinit()

    def test_serialized_control_definitions_and_validation(self):
        graph = Graph("serialize")
        graph.add_node(StateNode("a", lambda _: None))
        graph.add_node(StateNode("b", lambda _: None))
        self.assertFalse(graph.set_execution_order(["a", "a"]))
        self.assertFalse(graph.add_control_dependency("a", "missing"))
        self.assertTrue(graph.add_control_dependency("a", "b"))
        self.assertTrue(graph.set_execution_order(["a", "b"]))
        data = json.loads(graph.serialize())
        self.assertEqual(data["execution_order"], ["a", "b"])
        self.assertEqual(data["control_dependencies"], [{"from": "a", "to": "b"}])


if __name__ == "__main__":
    unittest.main()
