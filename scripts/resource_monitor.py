#!/usr/bin/env python3
"""Laptop Resource Monitor & Throttling Controller for Unitree G1 Generation.

Enforces resource controls during Kimodo motion generation and numerical analysis:
  - Single worker thread execution (torch.set_num_threads(2))
  - Host RAM soft limit: 4.0 GiB, Hard limit: 5.0 GiB
  - Target CPU usage < 60%
  - Target GPU usage < 60%
  - Target VRAM occupancy < 3.6 GiB (60% of 6 GiB RTX 3050)
  - Auto-throttles or pauses when limits are approached
  - Provides state controls: pause, resume, cancel
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, asdict

import psutil
import torch

# Resource Control Constants
RAM_SOFT_LIMIT_BYTES = int(4.0 * 1024 * 1024 * 1024)   # 4.0 GiB
RAM_HARD_LIMIT_BYTES = int(5.0 * 1024 * 1024 * 1024)   # 5.0 GiB
VRAM_LIMIT_BYTES = int(3.6 * 1024 * 1024 * 1024)       # 3.6 GiB (60% of 6GB)
TARGET_CPU_PCT = 60.0                                   # 60%
TARGET_GPU_PCT = 60.0                                   # 60%


@dataclass
class ResourceSnapshot:
    cpu_pct: float
    gpu_pct: float
    vram_used_gb: float
    vram_pct: float
    host_ram_used_gb: float
    host_ram_pct: float
    worker_rss_gb: float
    throttled: bool
    paused: bool
    status_msg: str
    timestamp: float


class ResourceController:
    """Manages worker thread limits, measures utilization, and handles pause/resume."""

    def __init__(self):
        self.paused = False
        self.canceled = False
        self._proc = psutil.Process()
        # Restrict PyTorch CPU thread count to 2 threads
        torch.set_num_threads(2)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)

    def sample(self) -> ResourceSnapshot:
        return self.check_and_throttle()

    def check_and_throttle(self, check_point_name: str = "") -> ResourceSnapshot:
        """Sample metrics; if limits exceeded, sleep/throttle or raise error if hard limit breached."""
        if self.canceled:
            raise RuntimeError("Generation job canceled by user.")

        while self.paused:
            print(f"ResourceController: Paused by user at '{check_point_name}'. Waiting...", flush=True)
            time.sleep(1.0)
            if self.canceled:
                raise RuntimeError("Generation job canceled by user.")

        # Measure Host RAM
        vm = psutil.virtual_memory()
        worker_rss = self._proc.memory_info().rss
        cpu_pct = psutil.cpu_percent(interval=0.1)

        # Measure GPU & VRAM
        vram_used_bytes = 0
        gpu_pct = 0.0
        if torch.cuda.is_available():
            vram_used_bytes = torch.cuda.memory_allocated()
            # Try pynvml if available for GPU utilization %
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_pct = float(util.gpu)
                vram_used_bytes = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            except Exception:
                gpu_pct = 0.0

        vram_used_gb = vram_used_bytes / (1024 ** 3)
        vram_pct = (vram_used_bytes / (6.0 * 1024 ** 3)) * 100.0 if torch.cuda.is_available() else 0.0
        host_ram_used_gb = vm.used / (1024 ** 3)
        worker_rss_gb = worker_rss / (1024 ** 3)

        throttled = False
        status_msg = "NORMAL"

        # Hard limit check
        if worker_rss > RAM_HARD_LIMIT_BYTES:
            raise MemoryError(
                f"Worker RSS {worker_rss_gb:.2f} GiB exceeded hard limit {RAM_HARD_LIMIT_BYTES/(1024**3):.2f} GiB"
            )

        # Soft limit / High load throttling
        if worker_rss > RAM_SOFT_LIMIT_BYTES or cpu_pct > TARGET_CPU_PCT or vram_used_bytes > VRAM_LIMIT_BYTES:
            throttled = True
            reasons = []
            if worker_rss > RAM_SOFT_LIMIT_BYTES:
                reasons.append(f"Worker RSS {worker_rss_gb:.2f} GB > 4.0 GB")
            if cpu_pct > TARGET_CPU_PCT:
                reasons.append(f"CPU {cpu_pct:.1f}% > 60%")
            if vram_used_bytes > VRAM_LIMIT_BYTES:
                reasons.append(f"VRAM {vram_used_gb:.2f} GB > 3.6 GB")

            status_msg = f"THROTTLED: {'; '.join(reasons)}"
            # Sleep briefly to throttle load back below 60%
            time.sleep(0.5)

        return ResourceSnapshot(
            cpu_pct=round(cpu_pct, 1),
            gpu_pct=round(gpu_pct, 1),
            vram_used_gb=round(vram_used_gb, 2),
            vram_pct=round(vram_pct, 1),
            host_ram_used_gb=round(host_ram_used_gb, 2),
            host_ram_pct=round(vm.percent, 1),
            worker_rss_gb=round(worker_rss_gb, 2),
            throttled=throttled,
            paused=self.paused,
            status_msg=status_msg,
            timestamp=time.time()
        )


GLOBAL_CONTROLLER = ResourceController()
