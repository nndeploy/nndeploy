"""Native compute thread settings owned by one isolated process Worker."""
import multiprocessing
import os
import sys


class CpuThreads:
    def __init__(self, count):
        if multiprocessing.parent_process() is None:
            raise RuntimeError('CPU thread settings require an isolated Worker process')
        self.count = count
        self.torch_interop = False
        self.apply()

    def apply(self):
        # Future library loads inherit the settings; already loaded libraries
        # need runtime controls. Never mutate the API process environment.
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = str(self.count)
        from threadpoolctl import threadpool_limits
        self.limits = threadpool_limits(limits=self.count)
        cv2 = sys.modules.get('cv2')
        if cv2 is not None:
            cv2.setNumThreads(self.count)
            if cv2.getNumThreads() > self.count:
                # GCD builds ignore positive thread counts. Sequential mode
                # is a supported upper bound; never silently keep all CPUs.
                cv2.setNumThreads(0)
            if cv2.getNumThreads() > self.count:
                raise RuntimeError('OpenCV could not apply the CPU thread limit')
        numexpr = sys.modules.get('numexpr')
        if numexpr is not None:
            numexpr.set_num_threads(min(self.count, numexpr.get_num_threads()))
        torch = sys.modules.get('torch')
        if torch is not None:
            torch.set_num_threads(self.count)
            if not self.torch_interop:
                if torch.get_num_interop_threads() > self.count:
                    torch.set_num_interop_threads(self.count)
                self.torch_interop = True

    def configure_graph(self, graph):
        from nndeploy import _nndeploy_internal as native
        pending, visited = [graph], {}
        while pending:
            node = pending.pop()
            if id(node) in visited:
                continue
            visited[id(node)] = node
            param = node.get_param()
            if isinstance(param, native.inference.InferenceParam):
                current = param.get_num_thread()
                param.set_num_thread(min(current, self.count) if current > 0 else self.count)
                if hasattr(param, 'inter_op_num_threads_'):
                    current = param.inter_op_num_threads_
                    param.inter_op_num_threads_ = min(current, self.count) if current > 0 else self.count
            if isinstance(node, native.dag.Graph):
                pending.extend(node.get_nodes())
        self.apply()
