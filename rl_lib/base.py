"""Base trainer class for RL algorithms."""

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import numpy as np
import psutil
import torch
from gymnasium.vector import AsyncVectorEnv
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard.writer import SummaryWriter

from rl_lib.buffers import ReplayBuffer
from rl_lib.config import save_config
from rl_lib.envs import make_env
from rl_lib.utils import get_device


class BaseTrainer(ABC):
    """Base class for RL trainers.

    Handles common training loop logic:
    - Environment setup
    - Logging and checkpointing
    - Performance monitoring
    - Learning rate scheduling
    """

    def __init__(self, config: Dict[str, Any], experiment_name: str = None):
        """Initialize trainer.

        Args:
            config: Configuration dictionary
            experiment_name: Optional experiment name prefix
        """
        self.config = config
        self.experiment_name = experiment_name

        # Load config sections
        self.env_cfg = config.get("env", {})
        self.training_cfg = config.get("training", {})
        self.parallel_cfg = config.get("parallel", {})
        self.logging_cfg = config.get("logging", {})
        self.resume_cfg = config.get("resume", {})

        # Extract hyperparameters
        self.env_name = self.env_cfg.get("name", "ALE/Boxing-v5")
        self.frame_stack = int(self.env_cfg.get("frame_stack", 4))
        self.frameskip = int(self.env_cfg.get("frameskip", 4))
        self.repeat_prob = float(self.env_cfg.get("repeat_action_probability", 0.25))
        self.obs_type = str(self.env_cfg.get("obs_type", "pixel")).lower()
        self.max_episode_steps = int(self.env_cfg.get("max_episode_steps", 0))

        self.num_envs = int(self.parallel_cfg.get("num_envs", 8))
        self.seed_base = int(self.parallel_cfg.get("seed", 42))

        self.total_steps = int(self.training_cfg.get("total_steps", 5_000_000))
        self.buffer_size = int(self.training_cfg.get("buffer_size", 1_000_000))
        self.warmup = int(self.training_cfg.get("warmup_steps", 50_000))
        self.batch_size = int(self.training_cfg.get("batch_size", 256))
        self.updates_per_step = int(self.training_cfg.get("updates_per_step", 4))
        self.gamma = float(self.training_cfg.get("gamma", 0.99))
        self.lr_start = float(self.training_cfg.get("lr", 3e-4))
        self.lr_final = float(self.training_cfg.get("lr_final", self.lr_start))
        self.lr_decay_steps = int(self.training_cfg.get("lr_decay_steps", self.total_steps))
        self.target_update = int(self.training_cfg.get("target_update_freq", 10_000))
        self.target_update_mode = str(self.training_cfg.get("target_update_mode", "hard")).lower()
        self.tau = float(self.training_cfg.get("tau", 0.0))
        self.grad_clip = float(self.training_cfg.get("grad_clip", 10.0))
        self.use_amp = bool(self.training_cfg.get("use_amp", True)) and torch.cuda.is_available()
        self.reward_clip = self.training_cfg.get("reward_clip", None)

        self.log_interval = int(self.logging_cfg.get("log_interval", 10_000))
        self.save_interval = int(self.logging_cfg.get("save_interval", 500_000))

        # Setup device
        self.device = get_device()

        # Setup directories
        self._setup_directories()

        # Resume from checkpoint if specified
        self.resume_ckpt = self.resume_cfg.get("checkpoint") or os.environ.get("RESUME_CKPT")

        # Will be set by subclasses
        self.q = None
        self.q_tgt = None
        self.opt = None
        self.scaler = None

    def _setup_directories(self):
        """Setup logging and checkpoint directories."""
        run_stamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{self.env_name.replace('/', '_')}-{run_stamp}"

        if self.experiment_name:
            prefix = str(self.experiment_name).strip().replace("/", "_").replace("\\", "_")
            if prefix:
                run_name = f"{prefix}-{run_name}"

        log_base = Path(self.logging_cfg.get("log_dir", "logs"))
        save_base = Path(self.logging_cfg.get("save_dir", "checkpoints"))

        self.log_dir = log_base / run_name
        self.save_dir = save_base / run_name

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def setup_envs(self) -> AsyncVectorEnv:
        """Create vectorized environments.

        Returns:
            AsyncVectorEnv: Vectorized environment
        """
        envs = AsyncVectorEnv([
            make_env(
                self.env_name,
                self.seed_base + i,
                self.frameskip,
                self.repeat_prob,
                self.frame_stack,
                self.obs_type,
                max_episode_steps=(self.max_episode_steps if self.max_episode_steps > 0 else None),
            )
            for i in range(self.num_envs)
        ])
        return envs

    @abstractmethod
    def setup_networks(self, obs_shape, n_actions):
        """Setup networks for the algorithm.

        Args:
            obs_shape: Observation shape
            n_actions: Number of actions
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch):
        """Perform one training step.

        Args:
            batch: Training batch from replay buffer

        Returns:
            Tuple of (loss, grad_norm, metrics_dict)
        """
        raise NotImplementedError

    @abstractmethod
    def select_actions(self, obs):
        """Select actions for given observations.

        Args:
            obs: Observations [num_envs, ...]

        Returns:
            Actions as numpy array
        """
        raise NotImplementedError

    def _schedule_helper(self, step_val: int, start: float, end: float, decay_steps: int) -> float:
        """Linear schedule helper.

        Args:
            step_val: Current step
            start: Start value
            end: End value
            decay_steps: Steps to decay from start to end

        Returns:
            Scheduled value
        """
        if decay_steps and decay_steps > 0:
            frac = min(1.0, max(0.0, step_val / float(decay_steps)))
            return start + (end - start) * frac
        return start

    def _update_learning_rate(self, step: int):
        """Update learning rate based on schedule.

        Args:
            step: Current training step
        """
        lr_now = self._schedule_helper(step, self.lr_start, self.lr_final, self.lr_decay_steps)
        for pg in self.opt.param_groups:
            pg["lr"] = lr_now
        return lr_now

    def _update_target_network(self, step: int):
        """Update target network.

        Args:
            step: Current training step
        """
        if self.target_update_mode == "soft" and self.tau > 0.0:
            with torch.no_grad():
                for p_tgt, p in zip(self.q_tgt.parameters(), self.q.parameters()):
                    p_tgt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
        else:
            if step % self.target_update == 0:
                self.q_tgt.load_state_dict(self.q.state_dict())

    def _save_checkpoint(self, step: int):
        """Save training checkpoint.

        Args:
            step: Current training step
        """
        ckpt = {
            "step": step,
            "q_state_dict": self.q.state_dict(),
            "q_tgt_state_dict": self.q_tgt.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "config": self.config,
        }
        if self.scaler is not None:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(ckpt, self.save_dir / f"checkpoint_step{step}.pt")

    def _save_final_model(self):
        """Save final model."""
        torch.save({
            "q_state_dict": self.q.state_dict(),
            "config": self.config,
        }, self.save_dir / "final_q_model.pt")

    def _load_checkpoint(self) -> int:
        """Load checkpoint if resume is specified.

        Returns:
            Step to resume from, or 0 if not resuming
        """
        if self.resume_ckpt is None or not Path(self.resume_ckpt).is_file():
            return 0

        ckpt = torch.load(self.resume_ckpt, map_location=self.device)
        self.q.load_state_dict(ckpt["q_state_dict"])
        self.q_tgt.load_state_dict(ckpt.get("q_tgt_state_dict", ckpt["q_state_dict"]))

        if "optimizer_state_dict" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in ckpt and self.use_amp:
            try:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception:
                pass

        step = int(ckpt.get("step", 0))
        print(f"Resumed from {self.resume_ckpt} at step {step:,} (replay buffer not restored)")
        return step

    def run(self):
        """Run training loop."""
        # Setup
        envs = self.setup_envs()
        obs, _ = envs.reset()

        try:
            n_actions = int(getattr(envs.single_action_space, "n"))
        except Exception:
            raise RuntimeError("Expected Discrete action space with attribute 'n'.")

        # Setup networks
        self.setup_networks(obs.shape[1:], n_actions)

        # Setup optimizer and scaler
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.lr_start)
        # Create a GradScaler; it will be disabled automatically when AMP is not used
        self.scaler = GradScaler(enabled=self.use_amp)

        # Load checkpoint if resuming
        step = self._load_checkpoint()

        # Setup replay buffer and logging
        rb = ReplayBuffer(obs.shape[1:], self.buffer_size)
        writer = SummaryWriter(self.log_dir)

        # Save config for this run
        save_config(self.config, self.save_dir / "config.yaml")

        # Training state
        completed_episodes = 0
        episode_rewards = np.zeros(self.num_envs)
        episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        start_time = time.time()

        last_log_step = step
        last_log_time = start_time

        # Hardware monitoring
        process = psutil.Process(os.getpid())
        try:
            process.cpu_percent(None)
        except Exception:
            pass
        num_cpus = psutil.cpu_count(logical=True) or 1

        # Optional NVML
        nvml = None
        gpu_handle = None
        try:
            import pynvml as nvml
            nvml.nvmlInit()
            gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            nvml = None

        # Performance accumulators
        env_time_accum = 0.0
        train_time_accum = 0.0
        started_training = False

        # Main training loop
        while step < self.total_steps:
            # Update learning rate
            lr_now = self._update_learning_rate(step)

            # Select actions and step environment
            actions = self.select_actions(obs)

            t0 = time.perf_counter()
            nxt, rew, term, trunc, _ = envs.step(actions)
            env_time_accum += time.perf_counter() - t0

            done = term | trunc

            # Optional reward clipping
            if isinstance(self.reward_clip, (list, tuple)) and len(self.reward_clip) == 2:
                rmin, rmax = float(self.reward_clip[0]), float(self.reward_clip[1])
                rew_train = np.clip(rew, rmin, rmax)
            else:
                rew_train = rew

            # Store transitions
            rb.push_batch(obs, actions, rew_train, nxt, done)

            # Log step-level metrics
            try:
                # Log raw (unclipped) step rewards
                writer.add_scalar("train/step_reward_mean", float(np.mean(rew)), step)
                writer.add_scalar("train/step_reward_sum", float(np.sum(rew)), step)

                # If reward clipping is enabled, also log clipped rewards for clarity
                if isinstance(self.reward_clip, (list, tuple)) and len(self.reward_clip) == 2:
                    writer.add_scalar("train/step_reward_mean_clipped", float(np.mean(rew_train)), step)
                    writer.add_scalar("train/step_reward_sum_clipped", float(np.sum(rew_train)), step)

                writer.add_scalar("train/lr", lr_now, step)
            except Exception:
                pass

            # Track episode rewards
            episode_rewards += rew
            episode_lengths += 1
            for i, d in enumerate(done):
                if d:
                    # Log episode metrics against episode count (original behavior)
                    writer.add_scalar("episode/reward", episode_rewards[i], completed_episodes)
                    writer.add_scalar("episode/length", int(episode_lengths[i]), completed_episodes)

                    # Additionally, log the same episode metrics against training step to align with other curves
                    writer.add_scalar("episode_by_step/reward", episode_rewards[i], step)
                    writer.add_scalar("episode_by_step/length", int(episode_lengths[i]), step)

                    completed_episodes += 1
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0

            obs = nxt
            step += self.num_envs

            # Training
            if len(rb) >= self.warmup:
                if not started_training:
                    print(f"Warmup complete at buffer {len(rb):,}. Starting training...")
                    try:
                        writer.add_text("status", f"Training started at step {step}", step)
                    except Exception:
                        pass
                    started_training = True

                t1 = time.perf_counter()
                for _ in range(self.updates_per_step):
                    batch = rb.sample(self.batch_size)
                    loss, grad_norm, metrics = self.train_step(batch)

                    writer.add_scalar("train/loss", loss, step)
                    writer.add_scalar("train/grad_norm", grad_norm, step)

                    # Log algorithm-specific metrics
                    for key, value in metrics.items():
                        writer.add_scalar(f"train/{key}", value, step)

                train_time_accum += time.perf_counter() - t1

            # Update target network
            self._update_target_network(step)

            # Logging
            if step - last_log_step >= self.log_interval:
                elapsed = time.time() - last_log_time
                sps = (step - last_log_step) / max(elapsed, 1e-6)
                buffer_fill = len(rb)

                # CPU/GPU metrics
                cpu_all = process.cpu_percent(None) or 0.0
                cpu_norm = max(0.0, min(100.0, cpu_all / (num_cpus or 1)))
                mem_gb = (process.memory_info().rss) / 1024**3

                if torch.cuda.is_available():
                    gpu_mem_alloc = torch.cuda.memory_allocated(0) / 1024**3
                    gpu_mem_res = torch.cuda.memory_reserved(0) / 1024**3
                    if nvml is not None and gpu_handle is not None:
                        try:
                            util = nvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                            gpu_util = float(util.gpu)
                            writer.add_scalar("perf/gpu_util_percent", gpu_util, step)
                        except Exception:
                            gpu_util = None
                    else:
                        gpu_util = None
                else:
                    gpu_mem_alloc = 0.0
                    gpu_mem_res = 0.0
                    gpu_util = None

                # Env/Train ratio
                total_work = env_time_accum + train_time_accum
                if total_work > 0:
                    env_ratio = env_time_accum / total_work
                    train_ratio = train_time_accum / total_work
                    writer.add_scalar("perf/env_time_ratio", env_ratio, step)
                    writer.add_scalar("perf/train_time_ratio", train_ratio, step)

                # Write perf scalars
                writer.add_scalar("perf/steps_per_sec", sps, step)
                writer.add_scalar("perf/proc_cpu_percent_norm", cpu_norm, step)
                writer.add_scalar("perf/memory_gb", mem_gb, step)
                if torch.cuda.is_available():
                    writer.add_scalar("perf/gpu_memory_allocated_gb", gpu_mem_alloc, step)
                    writer.add_scalar("perf/gpu_memory_reserved_gb", gpu_mem_res, step)

                # Console output
                print(f"Step {step:,} | SPS {sps:.1f} | Buffer {buffer_fill}")
                perf_msg = f"  Performance: {sps:.1f} SPS | CPU: {cpu_norm:.1f}% | RAM: {mem_gb:.2f} GB"
                if torch.cuda.is_available():
                    perf_msg += f" | GPU Mem: {gpu_mem_alloc:.2f}/{gpu_mem_res:.2f} GB"
                    if gpu_util is not None:
                        perf_msg += f" | GPU Util: {gpu_util:.0f}%"
                print(perf_msg)

                # Reset accumulators
                env_time_accum = 0.0
                train_time_accum = 0.0
                last_log_step = step
                last_log_time = time.time()
                try:
                    writer.flush()
                except Exception:
                    pass

            # Checkpoint
            if step % self.save_interval == 0:
                self._save_checkpoint(step)

        # Save final model
        self._save_final_model()
        print("Training complete. Model saved.")

        # Cleanup
        try:
            if nvml is not None:
                nvml.nvmlShutdown()
        except Exception:
            pass
