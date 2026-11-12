"""Condition/Loop state, hook, nesting and legacy execution regressions."""
from dataclasses import dataclass, field
import unittest

import nndeploy.base as base
from nndeploy.dag import Condition, Graph, Loop, Node, State


@dataclass
class FlowState(State):
    steps: int = 3
    step: int = 0
    branch: int = 0
    total: int = 0
    trace: list[str] = field(default_factory=list)


class Work(Node):
    def __init__(self, name, action):
        super().__init__(name)
        self.action = action

    def run(self):
        self.action(self.state)
        return base.Status.ok()


class Choice(Condition):
    def __init__(self, name):
        super().__init__(name)
        self.select_action = None

    def choose(self):
        if self.select_action:
            return self.select_action()
        return self.state.branch


class Repeat(Loop):
    def __init__(self, name):
        super().__init__(name)
        self.before_action = None
        self.after_action = None
        self.init_calls = []

    def init_start(self):
        self.init_calls.append("start")
        return base.Status.ok()

    def init_end(self):
        self.init_calls.append("end")
        return base.Status.ok()

    def loops(self):
        return self.state.steps

    def iter_before(self):
        self.state.trace.append("before")
        if self.before_action:
            self.before_action()
        return base.Status.ok()

    def iter_after(self):
        self.state.trace.append("after")
        self.state.step += 1
        if self.after_action:
            self.after_action()
        return base.Status.ok()


class StateControlFlowTests(unittest.TestCase):
    def assert_unbound(self, node):
        self.assertFalse(node.is_running())
        with self.assertRaises(RuntimeError):
            node.get_state()

    def test_condition_uses_request_state_and_runs_only_selected_branch(self):
        for parallel in (base.ParallelType.Sequential, base.ParallelType.Task):
            with self.subTest(parallel=parallel):
                choice = Choice("choice")
                choice.set_parallel_type(parallel)
                choice.add_node(Work("zero", lambda s: s.trace.append("zero")))
                choice.add_node(Work("one", lambda s: s.trace.append("one")))
                self.assertTrue(choice.init())
                try:
                    for branch in (1, 0):
                        state = FlowState(branch=branch)
                        self.assertTrue(choice.run_with_state(state))
                        self.assertEqual(state.trace, ["one" if branch else "zero"])
                        self.assert_unbound(choice)
                finally:
                    choice.deinit()

    def test_loop_condition_branch_graph_nesting_and_hooks(self):
        for parallel in (base.ParallelType.Sequential, base.ParallelType.Task):
            with self.subTest(parallel=parallel):
                branch = Graph("branch")
                branch.add_node(Work("first", lambda s: s.trace.append("first")))
                branch.add_node(Work("second", lambda s: s.trace.append("second")))
                self.assertTrue(branch.set_execution_order(["first", "second"]))
                choice = Choice("choice")
                choice.add_node(branch)
                choice.add_node(Work("odd", lambda s: s.trace.append("odd")))
                choice.select_action = lambda: choice.state.step % 2
                loop = Repeat("loop")
                loop.add_node(choice)
                loop.set_parallel_type(parallel)
                self.assertTrue(loop.set_execution_order(["choice"]))
                self.assertTrue(loop.init())
                try:
                    for steps in (3, 0, 1):
                        state = FlowState(steps=steps)
                        self.assertTrue(loop.run_with_state(state))
                        expected = []
                        for index in range(steps):
                            expected += ["before"] + (["odd"] if index % 2 else ["first", "second"]) + ["after"]
                        self.assertEqual(state.trace, expected)
                        self.assertEqual(state.step, steps)
                        self.assert_unbound(loop)
                        self.assert_unbound(choice)
                        self.assert_unbound(branch)
                    self.assertEqual(loop.init_calls, ["start", "end"])
                finally:
                    loop.deinit()

    def test_condition_can_select_a_loop(self):
        loop = Repeat("repeat")
        loop.add_node(Work("body", lambda s: s.trace.append("body")))
        self.assertTrue(loop.set_execution_order(["body"]))
        choice = Choice("choice")
        choice.add_node(loop)
        choice.add_node(Work("other", lambda s: s.trace.append("other")))
        self.assertTrue(choice.init())
        try:
            state = FlowState(steps=2)
            self.assertTrue(choice.run_with_state(state))
            self.assertEqual(state.trace, ["before", "body", "after"] * 2)
            self.assert_unbound(loop)
        finally:
            choice.deinit()

    def test_dynamic_loop_bound_and_recursive_hook_call(self):
        loop = Repeat("loop")
        loop.add_node(Work("body", lambda s: s.trace.append("body")))
        self.assertTrue(loop.set_execution_order(["body"]))

        def before():
            self.assertFalse(loop.run())
            self.assertFalse(loop.run_with_state(FlowState()))

        loop.before_action = before
        loop.after_action = lambda: setattr(loop.state, "steps", 2)
        self.assertTrue(loop.init())
        try:
            state = FlowState(steps=20)
            self.assertTrue(loop.run_with_state(state))
            self.assertEqual(state.step, 2)
            self.assertEqual(state.trace, ["before", "body", "after"] * 2)
        finally:
            loop.deinit()

    def test_invalid_and_throwing_choose_unbind_and_allow_retry(self):
        choice = Choice("choice")
        choice.add_node(Work("branch", lambda s: s.trace.append("branch")))
        self.assertTrue(choice.init())
        try:
            self.assertFalse(choice.run_with_state(FlowState(branch=9)))
            self.assert_unbound(choice)

            def fail():
                choice.state.total = 42
                raise ValueError("choose failed")

            choice.select_action = fail
            state = FlowState()
            self.assertFalse(choice.run_with_state(state))
            self.assertEqual(state.total, 42)
            self.assertEqual(state.trace, [])
            self.assert_unbound(choice)
            choice.select_action = None
            self.assertTrue(choice.run_with_state(FlowState()))
        finally:
            choice.deinit()

    def test_loop_hook_failures_stop_and_keep_partial_state(self):
        for failing_hook in ("before_action", "after_action"):
            with self.subTest(hook=failing_hook):
                loop = Repeat("loop")
                loop.add_node(Work("body", lambda s: s.trace.append("body")))
                loop.set_execution_order(["body"])

                def fail():
                    raise ValueError("hook failed")

                setattr(loop, failing_hook, fail)
                self.assertTrue(loop.init())
                try:
                    state = FlowState()
                    self.assertFalse(loop.run_with_state(state))
                    expected = ["before"] if failing_hook == "before_action" else ["before", "body", "after"]
                    self.assertEqual(state.trace, expected)
                    self.assert_unbound(loop)
                    setattr(loop, failing_hook, None)
                    self.assertTrue(loop.run_with_state(FlowState(steps=1)))
                finally:
                    loop.deinit()

    def test_interruption_stops_iterations(self):
        loop = Repeat("loop")
        loop.add_node(Work("body", lambda s: loop.interrupt()))
        loop.set_execution_order(["body"])
        self.assertTrue(loop.init())
        try:
            state = FlowState()
            self.assertFalse(loop.run_with_state(state))
            self.assertEqual(state.trace, ["before"])
            self.assertEqual(state.step, 0)
            self.assert_unbound(loop)
        finally:
            loop.deinit()

    def test_loop_pipeline_is_still_rejected(self):
        loop = Repeat("loop")
        loop.set_parallel_type(base.ParallelType.Pipeline)
        loop.add_node(Work("body", lambda s: None))
        self.assertTrue(loop.init())
        try:
            self.assertEqual(loop.run_with_state(FlowState()), base.StatusCode.ErrorNotSupport)
            self.assert_unbound(loop)
        finally:
            loop.deinit()


if __name__ == "__main__":
    unittest.main()
