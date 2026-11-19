"""Spawned rank workers using ready/ok/error packets and owner-only results.

The algorithm supplies a picklable worker entrypoint and model specification.
This layer owns IPC, timeouts, CUDA IPC acknowledgement and process teardown.
"""
from copy import copy, deepcopy
import os
import tempfile
import time
import torch
import torch.multiprocessing as mp
from torch.utils._pytree import tree_flatten, tree_map


class RankProcessPool:
    def __init__(self, worker, world_size, plan, args, model_spec):
        self.plan = plan
        self.boundary_transport = plan.get('boundary_transport', 'cpu')
        self.processes, self.connections = [], []
        self.closed = False
        self.store = tempfile.TemporaryDirectory(prefix='nndeploy-ranks-')
        started = time.perf_counter()
        context = mp.get_context('spawn')
        try:
            for rank in range(world_size):
                worker_args = copy(args)
                for name in ('distributed_context', 'parallel_context', 'official_parallel_context'):
                    setattr(worker_args, name, None)
                worker_args = deepcopy(worker_args)
                parent, child = context.Pipe()
                process = context.Process(target=worker, args=(rank, plan, worker_args, model_spec, child,
                    f'file://{self.store.name}/store', os.getpid()), name=f'nndeploy-rank-{rank}')
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self.initial_reports = [self._receive(rank, expected='ready') for rank in range(len(self.processes))]
        except BaseException:
            self.close(force=True)
            raise
        self.setup_seconds = time.perf_counter() - started

    def _receive(self, rank=0, *, expected='ok', timeout=300):
        def receive(index):
            try:
                return self.connections[index].recv()
            except (EOFError, OSError) as error:
                raise RuntimeError(f'NCCL DiT rank {index} IPC connection closed') from error
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.connections[rank].poll(.1):
                status, result = receive(rank)
                if status != expected:
                    raise RuntimeError(f'NCCL DiT rank {rank}: {status}: {result}')
                return result
            for index, process in enumerate(self.processes):
                # Ready messages on peer pipes must be retained until consumed.
                if not process.is_alive():
                    detail = receive(index) if self.connections[index].poll() else process.exitcode
                    raise RuntimeError(f'NCCL DiT rank {index} exited: {detail}')
                if expected == 'ok' and index and self.connections[index].poll():
                    raise RuntimeError(f'NCCL DiT rank {index} failed: {receive(index)}')
        raise TimeoutError(f'NCCL DiT rank {rank} response timed out')

    def call(self, command, payload=None):
        if self.closed:
            raise RuntimeError('NCCL DiT process pool is closed')
        try:
            if command == 'predict' and self.boundary_transport == 'cuda_ipc':
                # Inputs can be produced on a caller's non-default stream.
                devices = {v.device for v in tree_flatten(payload)[0]
                           if isinstance(v, torch.Tensor) and v.is_cuda}
                for device in devices:
                    if device != self.plan['devices'][0]:
                        raise ValueError('CUDA IPC boundary requires inputs on the rank-0 device')
                    torch.cuda.synchronize(device)
            for rank, connection in enumerate(self.connections):
                connection.send((command, payload if rank == 0 else None))
            received = self._receive()
            if command == 'predict' and self.boundary_transport == 'cuda_ipc':
                # Never expose imported worker storage to a caller. This also
                # makes held outputs valid after reset, worker failure or close.
                try:
                    result = tree_map(lambda value: value.clone(), received)
                    torch.cuda.synchronize(self.plan['devices'][0])
                finally:
                    # Drop imported handles before any exception closes workers.
                    received = None
                return result
            return received
        except (OSError, EOFError) as error:
            self.close(force=True)
            raise RuntimeError('NCCL DiT worker IPC failed') from error
        except BaseException:
            self.close(force=True)
            raise

    def close(self, force=False):
        if self.closed:
            return
        self.closed = True
        if self.connections and not force:
            try:
                for connection in self.connections:
                    connection.send(('close', None))
            except (OSError, EOFError):
                force = True
        for process in self.processes:
            if force and process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for connection in self.connections:
            connection.close()
        self.store.cleanup()
