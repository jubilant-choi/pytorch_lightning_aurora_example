# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations
from mpi4py import MPI

import logging
import os
import socket
from datetime import timedelta
from typing import Any, Callable
from typing_extensions import override
from contextlib import nullcontext

import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel
import pytorch_lightning as pl
from lightning_fabric.utilities.seed import reset_seed
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies import StrategyRegistry
from pytorch_lightning.utilities.rank_zero import rank_zero_only

__all__ = ["MPIEnvironment", "MPIDDPStrategy"]

default_pg_timeout = timedelta(seconds=1800)
log = logging.getLogger(__name__)

class MPIEnvironment(LightningEnvironment):
    """
    This environment specializes in the use of Intel MPI for distributed
    multiworker instances. The key assumptions for using this environment
    are:

    1. The use of Intel MPI
    2. The launch script utilizes PyTorch Lightning abstractions
    3. The launch script is used via `mpiexec -n -ppn ... python train.py

    The main motivation behind this environment is two-fold: to keep the
    `pl.Trainer` functionality, while maintaining the ability to work with
    NUMA bindings (e.g. via `-map-by numa`) to ensure optimal CPU/memory
    utilization.
    """

    def __init__(self, main_address: str | None = None, main_port: int | None = None, use_only_rank0: bool = False):
        log.debug("MPI env initialized")
        self.main_address = main_address
        self.main_port = main_port
        self.SIZE = MPI.COMM_WORLD.Get_size() if not use_only_rank0 else 1
        self.RANK = MPI.COMM_WORLD.Get_rank() if not use_only_rank0 else 0
        self.LOCAL_RANK = os.environ.get('PALS_LOCAL_RANKID')
        os.environ['RANK'] = str(self.RANK)
        os.environ['WORLD_SIZE'] = str(self.SIZE)
        self.MASTER_ADDR = socket.gethostname() if self.RANK == 0 else None
        self.MASTER_ADDR = MPI.COMM_WORLD.bcast(self.MASTER_ADDR, root=0) if not use_only_rank0 else None
        os.environ['MASTER_ADDR'] = f"{self.MASTER_ADDR}.hsn.cm.aurora.alcf.anl.gov" if self.RANK == 0 else 'localhost'
        os.environ['MASTER_PORT'] = os.environ.get("MASTER_PORT", str(29500))

    def world_size(self) -> int:
        return int(self.SIZE)
        # world_size = os.environ.get("WORLD_SIZE", None)
        # if world_size == None:
        #     world_size = int(os.environ["PALS_LOCAL_SIZE"]) * len(os.environ["PBS_NODEFILE"].split())
        # log.debug(f"MPIEnvironment: WORLD SIZE={int(world_size)}")
        # return int(world_size) # PMI_SIZE | jubin changed.

    def local_rank(self) -> int:
        return int(self.LOCAL_RANK)
        # log.debug(f"MPIEnvironment: LOCAL RANK={int(os.environ.get('PALS_LOCAL_RANKID'))}")
        # return int(os.environ.get("PALS_LOCAL_RANKID")) # MPI_LOCALRANKID | jubin changed.

    def global_rank(self) -> int:
        return int(self.RANK)
        # log.debug(f"MPIEnvironment: GLOBAL RANK={int(os.environ.get('PMIX_RANK'))}")
        # return int(os.environ["PMIX_RANK"]) # PMI_RANK | jubin changed.

    @property
    def main_address(self) -> str:
        return self._main_address

    @main_address.setter
    @override
    def main_address(self, value: str | None) -> str:
        if not value:
            value = os.environ.get("MASTER_ADDR", "127.0.0.1")
        self._main_address = value

        # if not value:
        #     value = os.getenv("HYDRA_BSTRAP_LOCALHOST", None)
        # if not value:
        #     raise ValueError(
        #         "No main address passed, and MPI did not set HYDRA_BSTRAP_LOCALHOST."
        #     )
        # self._main_address = value
        # os.environ["MASTER_ADDR"] = self._main_address

    @property
    def main_port(self) -> int:
        return self._main_port

    @main_port.setter
    def main_port(self, value: int | None):
        self._main_port = (
            int(os.environ["MASTER_PORT"]) if "MASTER_PORT" in os.environ else find_free_network_port()
            )
        # if not value:
        #     value = 30256
        # # check to make sure port and address are accessible
        # self._main_port = value
        # os.environ["MASTER_PORT"] = str(self._main_port)

    @staticmethod
    def _validate_address_port(addr: str, port: int) -> bool:
        obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = obj.connect_ex((addr, port)) == 0
        obj.close()
        return result

    @property
    def creates_processes_externally(self) -> bool:
        """
        Override this because we rely on `mpiexec` or `mpirun` for
        the process spawning.
        """
        return True

    @property
    def local_world_size(self) -> int:
        """Return the number of devices per node."""
        log.debug(f"MPIEnvironment: LOCAL WORLD SIZE={int(os.environ.get('PALS_LOCAL_SIZE'))}")
        return int(os.environ["PALS_LOCAL_SIZE"]) # MPI_LOCALNRANKS | jubin changed.

    @property
    def num_nodes(self) -> int:
        """Return the of numbers, based on ranks per node and global world size."""
        num_nodes = self.world_size() // self.local_world_size
        log.debug(f"MPIEnvironment: NUM NODES={num_nodes}")
        return num_nodes


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.

    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class MPIDDPStrategy(DDPStrategy):
    def __init__(
        self,
        accelerator: pl.accelerators.Accelerator | None = None,
        parallel_devices: list[torch.device] | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: PrecisionPlugin | None = None,
        ddp_comm_state: object | None = None,
        ddp_comm_hook: Callable | None = None,
        ddp_comm_wrapper: Callable | None = None,
        model_averaging_period: int | None = None,
        process_group_backend: str | None = None,
        timeout: timedelta | None = default_pg_timeout,
        cluster_environment: MPIEnvironment | None = None,
        **kwargs: Any,
    ) -> None:
        if not cluster_environment:
            cluster_environment = MPIEnvironment()
        if process_group_backend:
            assert process_group_backend in [
                "xccl",
                "mpi",
            ], f"Unsupported distributed backend! {process_group_backend}"
        super().__init__(
            accelerator,
            parallel_devices,
            cluster_environment,
            checkpoint_io,
            precision_plugin,
            ddp_comm_state,
            ddp_comm_hook,
            ddp_comm_wrapper,
            model_averaging_period,
            process_group_backend,
            timeout,
            **kwargs,
        )

    def setup_distributed(self):
        """Overrides base method so we can perform dummy all_reduce."""
        log.detail(f"{self.__class__.__name__}: setting up distributed...")
        reset_seed()
        rank_zero_only.rank = self.global_rank

        port = self.cluster_environment.main_port
        addr = self.cluster_environment.main_address

        if not dist.is_initialized():
            log.info(f"Initializing distributed: GLOBAL_RANK: {self.cluster_environment.global_rank()}, MEMBER: {self.cluster_environment.global_rank() + 1}/{self.cluster_environment.world_size()}")
            dist.init_process_group(
                backend=self.process_group_backend,
                init_method=f"env://", # f"tcp://{addr}:{port}" | jubin changed.
                world_size=self.cluster_environment.world_size(),
                rank=self.cluster_environment.global_rank(),
                device_id=self.cluster_environment.global_rank(),
            )
        # this is to force initialization of distributed backend
        dummy = torch.ones((5, 2), device=self.root_device)
        dist.broadcast(dummy, src=0)

    def _setup_model(self, model: nn.Module) -> DistributedDataParallel:
        device_ids = self.determine_ddp_device_ids()
        # this enforces an XPU stream, instead of CUDA
        if device_ids is not None and hasattr(torch, "xpu"):
            ctx = torch.xpu.StreamContext(torch.xpu.current_stream())
        else:
            ctx = nullcontext()
        with ctx:
            log.debug(f"return DistributedDataParallel, {type(model)}")
            return DistributedDataParallel(
                module=model, device_ids=device_ids, **self._ddp_kwargs
            )

    def teardown(self):
        """Ensure that distributed processes close gracefully."""
        super().teardown()
        if dist.is_initialized():
            dist.destroy_process_group()


StrategyRegistry.register(
    "ddp_with_mpi",
    MPIDDPStrategy,
    description="Run distributed data parallel with an MPI environment.",
    process_group_backend="mpi",
)

StrategyRegistry.register(
    "ddp_with_xccl",
    MPIDDPStrategy,
    description="Run distributed data parallel with an XCCL environment.",
    process_group_backend="xccl",
)

if hasattr(torch, "xpu"): # remove matsciml's package_registry["ipex"] | jubin changed. 
    StrategyRegistry.register(
        "ddp_with_xpu",
        MPIDDPStrategy,
        description="Run distributed data parallel on Intel XPUs.",
        process_group_backend="xccl",
        accelerator="xpu"
    )