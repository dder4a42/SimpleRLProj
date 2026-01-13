"""
(do not remove this comment)
A minimal, value-based reinforcement learning baseline for ALE Atari environments (e.g. Pong, Breakout, Boxing) under partial observability and sticky actions.

Highlights
- Purely value-based: no explicit policy/actor network
- Entropy-regularized (soft) backups with Boltzmann acting
- Soft Double Q-learning target to reduce over-estimation
- Standard Atari preprocessing, stacked 84×84 grayscale frames
- Memory-efficient NumPy ring replay buffer
- IQN (Implicit Quantile Networks) support

Observation: stacked grayscale frames (84×84×frame_stack)
Network: Nature CNN (configurable) → Q-values for all actions

Soft Double Q target
Let π_online(a|s') ∝ exp(Q_online(s',a)/α). Then
  V(s') = E_{a'~π_online}[Q_target(s',a')] − α·H(π_online)
  Q(s,a) ← r + γ · V(s')
"""
import os
import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.vector import AsyncVectorEnv
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard.writer import SummaryWriter
import psutil

# Import shared components from rl_lib
from rl_lib.networks import QNet, IQNQNet
from rl_lib.buffers import ReplayBuffer
from rl_lib.envs import AtariPreprocess, make_env
from rl_lib.utils import load_config, get_device


# =========================
# Training step
# =========================
def train_step(q, q_tgt, opt, batch, gamma, alpha, scaler, device, use_amp, grad_clip=10.0, target_clip=100):
    obs, act, rew, nxt, done = batch

    # Pinned-memory path for faster H2D copies
    obs_cpu = torch.from_numpy(obs).pin_memory()
    nxt_cpu = torch.from_numpy(nxt).pin_memory()
    act_cpu = torch.from_numpy(act).pin_memory()
    rew_cpu = torch.from_numpy(rew).pin_memory()
    done_cpu = torch.from_numpy(done).pin_memory()

    obs = obs_cpu.to(device, non_blocking=True)
    nxt = nxt_cpu.to(device, non_blocking=True)
    act = act_cpu.to(device, dtype=torch.long, non_blocking=True)
    rew = rew_cpu.to(device, dtype=torch.float32, non_blocking=True)
    done = done_cpu.to(device, dtype=torch.float32, non_blocking=True)

    opt.zero_grad(set_to_none=True)

    with autocast(enabled=use_amp):
        q_sa = q(obs).gather(1, act[:, None]).squeeze(1)

        with torch.no_grad():
            q_next = q(nxt)
            q_next_t = q_tgt(nxt)

            logits = q_next / max(alpha, 1e-6)
            logits -= logits.max(1, keepdim=True).values
            pi = torch.softmax(logits, dim=1)

            entropy = -(pi * (pi + 1e-8).log()).sum(1)
            v_next = (pi * q_next_t).sum(1) - alpha * entropy

            target = rew + gamma * (1 - done) * v_next
            if target_clip is not None:
                target = target.clamp(-float(target_clip), float(target_clip))

        loss = F.smooth_l1_loss(q_sa, target)

        # Monitoring metrics (detach to float CPU)
        td = q_sa - target
        td_mae = torch.mean(torch.abs(td)).float().item()
        td_mse = torch.mean(td * td).float().item()
        qsa_max = torch.max(q_sa).float().item()
        qsa_min = torch.min(q_sa).float().item()
        tgt_max = torch.max(target).float().item()
        tgt_min = torch.min(target).float().item()
        ent_mean = torch.mean(entropy).float().item()
        ent_min = torch.min(entropy).float().item()
        ent_max = torch.max(entropy).float().item()

    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    grad_norm = nn.utils.clip_grad_norm_(q.parameters(), grad_clip)
    scaler.step(opt)
    scaler.update()

    grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
    metrics = {
        "td_error_mae": td_mae,
        "td_error_mse": td_mse,
        "q_sa_max": qsa_max,
        "q_sa_min": qsa_min,
        "target_max": tgt_max,
        "target_min": tgt_min,
        "entropy_mean": ent_mean,
        "entropy_min": ent_min,
        "entropy_max": ent_max,
    }
    return loss.item(), grad_norm, metrics


# =========================
# IQN Training step (quantile regression with Huber loss)
# =========================
def train_step_iqn(
    q,
    q_tgt,
    opt,
    batch,
    gamma: float,
    scaler: GradScaler,
    device: str,
    use_amp: bool,
    num_quantiles_pred: int = 32,
    num_quantiles_tgt: int = 32,
    kappa: float = 1.0,
    grad_clip: float = 10.0,
):
    obs, act, rew, nxt, done = batch

    # Pinned memory to device
    obs_cpu = torch.from_numpy(obs).pin_memory()
    nxt_cpu = torch.from_numpy(nxt).pin_memory()
    act_cpu = torch.from_numpy(act).pin_memory()
    rew_cpu = torch.from_numpy(rew).pin_memory()
    done_cpu = torch.from_numpy(done).pin_memory()

    obs = obs_cpu.to(device, non_blocking=True)
    nxt = nxt_cpu.to(device, non_blocking=True)
    act = act_cpu.to(device, dtype=torch.long, non_blocking=True)
    rew = rew_cpu.to(device, dtype=torch.float32, non_blocking=True)
    done = done_cpu.to(device, dtype=torch.float32, non_blocking=True)

    opt.zero_grad(set_to_none=True)

    B = obs.shape[0]

    with autocast(enabled=use_amp):
        # Sample taus for prediction and target
        taus_pred = torch.rand(B, int(num_quantiles_pred), device=obs.device)
        taus_tgt = torch.rand(B, int(num_quantiles_tgt), device=obs.device)

        # Predicted quantiles for chosen actions
        q_pred_all = q(obs, taus_pred)  # [B, Np, A]
        q_pred_sa = q_pred_all.gather(2, act[:, None, None].expand(B, int(num_quantiles_pred), 1)).squeeze(2)  # [B, Np]

        # Next-state greedy actions via expected Q of online net (Double IQN)
        q_next_exp = q.expected_q(nxt, num_quantiles=num_quantiles_tgt)  # [B, A]
        a_star = torch.argmax(q_next_exp, dim=1)  # [B]

        # Target quantiles from target net for a_star
        q_tgt_all = q_tgt(nxt, taus_tgt)  # [B, Nt, A]
        q_tgt_star = q_tgt_all.gather(2, a_star[:, None, None].expand(B, int(num_quantiles_tgt), 1)).squeeze(2)  # [B, Nt]
        target = rew[:, None] + gamma * (1.0 - done)[:, None] * q_tgt_star  # [B, Nt]

        # Quantile regression Huber loss
        u = target[:, None, :] - q_pred_sa[:, :, None]  # [B, Np, Nt]
        abs_u = torch.abs(u)
        huber = torch.where(abs_u <= kappa, 0.5 * u * u, kappa * (abs_u - 0.5 * kappa))
        # Weighting by tau - I[u<0]
        tau = taus_pred[:, :, None]  # [B, Np, 1]
        weight = torch.abs(tau - (u.detach() < 0.0).float())  # [B, Np, Nt]
        loss = (weight * huber).mean()

    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    grad_norm = nn.utils.clip_grad_norm_(q.parameters(), grad_clip)
    scaler.step(opt)
    scaler.update()

    grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
    metrics = {
        "iqn_loss": float(loss.detach().item()),
        "grad_norm": grad_norm,
    }
    return float(loss.detach().item()), grad_norm, metrics


# =========================
# Action selection (vector)
# =========================
@torch.no_grad()
def select_actions(q, obs, alpha, device, eval_quantiles: int = 32):
    # Pin and transfer the vectorized observations for faster H2D
    obs_cpu = torch.from_numpy(obs).pin_memory()
    obs = obs_cpu.to(device, non_blocking=True)
    # If IQN, use expected Q over quantiles; else plain Q
    if hasattr(q, "expected_q"):
        q_vals = q.expected_q(obs, num_quantiles=eval_quantiles)
    else:
        q_vals = q(obs)
    logits = q_vals / max(alpha, 1e-6)
    logits -= logits.max(1, keepdim=True).values
    probs = torch.softmax(logits, dim=1)
    return torch.multinomial(probs, 1).squeeze(1).cpu().numpy()


# =========================
# Main
# =========================
def main(exp_name=None):
    config = load_config()

    # Hyperparameters from config
    ENV_NAME = config["env"]["name"]
    FRAME_STACK = int(config["env"].get("frame_stack", 4))
    FRAMESKIP = int(config["env"].get("frameskip", 4))
    REPEAT_PROB = float(config["env"].get("repeat_action_probability", 0.25))

    PARALLEL = config.get("parallel", {})
    NUM_ENVS = int(PARALLEL.get("num_envs", 8))
    SEED_BASE = int(PARALLEL.get("seed", 42))

    TRAINING = config.get("training", {})
    TOTAL_STEPS = int(TRAINING.get("total_steps", 5_000_000))
    BUFFER_SIZE = int(TRAINING.get("buffer_size", 1_000_000))
    WARMUP = int(TRAINING.get("warmup_steps", 50_000))
    BATCH_SIZE = int(TRAINING.get("batch_size", 256))
    UPDATES_PER_STEP = int(TRAINING.get("updates_per_step", 4))
    GAMMA = float(TRAINING.get("gamma", 0.99))
    ALPHA_BASE = float(TRAINING.get("alpha", 0.05))
    ALPHA_START = float(TRAINING.get("alpha_start", ALPHA_BASE))
    ALPHA_FINAL = float(TRAINING.get("alpha_final", ALPHA_BASE))
    ALPHA_DECAY_STEPS = int(TRAINING.get("alpha_decay_steps", 0))
    LR_START = float(TRAINING.get("lr", 3e-4))
    LR_FINAL = float(TRAINING.get("lr_final", LR_START))
    LR_DECAY_STEPS = int(TRAINING.get("lr_decay_steps", TOTAL_STEPS))
    TARGET_UPDATE = int(TRAINING.get("target_update_freq", 10_000))
    TARGET_UPDATE_MODE = str(TRAINING.get("target_update_mode", "hard")).lower()
    TAU = float(TRAINING.get("tau", 0.0))
    GRAD_CLIP = float(TRAINING.get("grad_clip", 10.0))
    USE_AMP = bool(TRAINING.get("use_amp", True)) and torch.cuda.is_available()
    REWARD_CLIP = TRAINING.get("reward_clip", None)
    TARGET_CLIP = TRAINING.get("target_clip", 100)
    TRAINING_METHOD = str(TRAINING.get("method", "standard")).lower()  # "standard" or "iqn"

    IQN_CFG = config.get("iqn", {})
    IQN_NUM_QUANTILES = int(IQN_CFG.get("num_quantiles", 32))
    IQN_EMBED_DIM = int(IQN_CFG.get("embed_dim", 64))
    IQN_KAPPA = float(IQN_CFG.get("kappa", 1.0))
    IQN_EVAL_QUANTILES = int(IQN_CFG.get("eval_quantiles", IQN_NUM_QUANTILES))

    LOGGING = config.get("logging", {})
    RESUME_CFG = config.get("resume", {})
    RESUME_CKPT = RESUME_CFG.get("checkpoint") or os.environ.get("RESUME_CKPT")

    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{ENV_NAME.replace('/', '_')}-{run_stamp}"
    if exp_name:
        prefix = str(exp_name).strip().replace("/", "_").replace("\\", "_")
        if prefix:
            run_name = f"{prefix}-{run_name}"
    log_base = Path(LOGGING.get("log_dir", "logs"))
    save_base = Path(LOGGING.get("save_dir", "checkpoints"))
    log_dir = log_base / run_name
    save_dir = save_base / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_interval = int(LOGGING.get("log_interval", 10_000))  # steps
    save_interval = int(LOGGING.get("save_interval", 500_000))  # steps

    device = get_device()
    envs = AsyncVectorEnv(
        [make_env(ENV_NAME, SEED_BASE + i, FRAMESKIP, REPEAT_PROB, FRAME_STACK) for i in range(NUM_ENVS)]
    )

    obs, _ = envs.reset()
    # Robustly determine number of actions (Discrete)
    try:
        n_actions = int(getattr(envs.single_action_space, "n"))
    except Exception:
        raise RuntimeError("Expected Discrete action space with attribute 'n'.")

    net_cfg = config.get("network", {})
    conv_channels = tuple(net_cfg.get("conv_channels", [32, 64, 64]))
    fc_hidden = int(net_cfg.get("fc_hidden", 512))

    if TRAINING_METHOD == "iqn":
        q = IQNQNet(
            n_actions,
            in_ch=FRAME_STACK,
            conv_channels=conv_channels,
            fc_hidden=fc_hidden,
            quantile_embed_dim=IQN_EMBED_DIM,
        ).to(device)
        q_tgt = IQNQNet(
            n_actions,
            in_ch=FRAME_STACK,
            conv_channels=conv_channels,
            fc_hidden=fc_hidden,
            quantile_embed_dim=IQN_EMBED_DIM,
        ).to(device)
    else:
        q = QNet(n_actions, in_ch=FRAME_STACK, conv_channels=conv_channels, fc_hidden=fc_hidden).to(device)
        q_tgt = QNet(n_actions, in_ch=FRAME_STACK, conv_channels=conv_channels, fc_hidden=fc_hidden).to(device)
    q_tgt.load_state_dict(q.state_dict())

    opt = torch.optim.Adam(q.parameters(), lr=LR_START)
    scaler = GradScaler(enabled=USE_AMP)

    # Optional resume from checkpoint (does not restore replay buffer or env state)
    step = 0
    if RESUME_CKPT is not None and Path(RESUME_CKPT).is_file():
        ckpt = torch.load(RESUME_CKPT, map_location=device)
        q.load_state_dict(ckpt["q_state_dict"])
        q_tgt.load_state_dict(ckpt.get("q_tgt_state_dict", ckpt["q_state_dict"]))
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt and USE_AMP:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception:
                pass
        step = int(ckpt.get("step", 0))
        print(f"Resumed from {RESUME_CKPT} at step {step:,} (replay buffer not restored)")

    rb = ReplayBuffer(obs.shape[1:], BUFFER_SIZE)
    writer = SummaryWriter(log_dir)

    # Persist the effective config for this run
    try:
        import yaml
        with open(save_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)
    except Exception:
        pass

    completed_episodes = 0
    episode_rewards = np.zeros(NUM_ENVS)
    episode_lengths = np.zeros(NUM_ENVS, dtype=np.int32)
    start_time = time.time()

    last_log_step = step
    last_log_time = start_time

    # Hardware monitors
    process = psutil.Process(os.getpid())
    try:
        process.cpu_percent(None)  # prime
    except Exception:
        pass
    num_cpus = psutil.cpu_count(logical=True) or 1

    # Optional NVML for GPU util
    nvml = None
    gpu_handle = None
    try:
        import pynvml as nvml

        nvml.nvmlInit()
        gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        nvml = None

    # Perf accumulators (reset each log window)
    env_time_accum = 0.0
    train_time_accum = 0.0

    started_training = False
    # alpha schedule helper
    def current_alpha(step_val: int) -> float:
        if ALPHA_DECAY_STEPS and ALPHA_DECAY_STEPS > 0:
            frac = min(1.0, max(0.0, step_val / float(ALPHA_DECAY_STEPS)))
            return ALPHA_START + (ALPHA_FINAL - ALPHA_START) * frac
        return ALPHA_START

    # linear learning-rate schedule helper
    def current_lr(step_val: int) -> float:
        if LR_DECAY_STEPS and LR_DECAY_STEPS > 0:
            frac = min(1.0, max(0.0, step_val / float(LR_DECAY_STEPS)))
            return LR_START + (LR_FINAL - LR_START) * frac
        return LR_START

    while step < TOTAL_STEPS:
        alpha_now = current_alpha(step)
        lr_now = current_lr(step)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        actions = select_actions(q, obs, alpha_now, device, eval_quantiles=IQN_EVAL_QUANTILES)
        t0 = time.perf_counter()
        nxt, rew, term, trunc, _ = envs.step(actions)
        env_time_accum += time.perf_counter() - t0
        done = term | trunc
        # Optional training-time reward clipping
        if isinstance(REWARD_CLIP, (list, tuple)) and len(REWARD_CLIP) == 2:
            rmin, rmax = float(REWARD_CLIP[0]), float(REWARD_CLIP[1])
            rew_train = np.clip(rew, rmin, rmax)
        else:
            rew_train = rew

        rb.push_batch(obs, actions, rew_train, nxt, done)
        # Step-level reward monitoring
        try:
            writer.add_scalar("train/step_reward_mean", float(np.mean(rew)), step)
            writer.add_scalar("train/step_reward_sum", float(np.sum(rew)), step)
            if rew_train is not rew:
                writer.add_scalar("train/step_reward_mean_train", float(np.mean(rew_train)), step)
                writer.add_scalar("train/step_reward_sum_train", float(np.sum(rew_train)), step)
            writer.add_scalar("train/alpha", float(alpha_now), step)
            writer.add_scalar("train/lr", float(lr_now), step)
        except Exception:
            pass

        episode_rewards += rew
        episode_lengths += 1
        for i, d in enumerate(done):
            if d:
                writer.add_scalar("episode/reward", episode_rewards[i], completed_episodes)
                writer.add_scalar("episode/length", int(episode_lengths[i]), completed_episodes)
                completed_episodes += 1
                episode_rewards[i] = 0.0
                episode_lengths[i] = 0

        obs = nxt
        step += NUM_ENVS

        if len(rb) >= WARMUP:
            if not started_training:
                print(f"Warmup complete at buffer {len(rb):,}. Starting training...")
                try:
                    writer.add_text("status", f"Training started at step {step}", step)
                except Exception:
                    pass
                started_training = True
            t1 = time.perf_counter()
            for _ in range(UPDATES_PER_STEP):
                batch = rb.sample(BATCH_SIZE)
                if TRAINING_METHOD == "iqn":
                    loss, grad_norm, m = train_step_iqn(
                        q,
                        q_tgt,
                        opt,
                        batch,
                        GAMMA,
                        scaler,
                        device,
                        USE_AMP,
                        num_quantiles_pred=IQN_NUM_QUANTILES,
                        num_quantiles_tgt=IQN_NUM_QUANTILES,
                        kappa=IQN_KAPPA,
                        grad_clip=GRAD_CLIP,
                    )
                    writer.add_scalar("train/loss", loss, step)
                    writer.add_scalar("train/grad_norm", grad_norm, step)
                    writer.add_scalar("train/iqn_loss", m.get("iqn_loss", loss), step)
                else:
                    loss, grad_norm, m = train_step(
                        q,
                        q_tgt,
                        opt,
                        batch,
                        GAMMA,
                        alpha_now,
                        scaler,
                        device,
                        USE_AMP,
                        grad_clip=GRAD_CLIP,
                        target_clip=TARGET_CLIP,
                    )
                    writer.add_scalar("train/loss", loss, step)
                    writer.add_scalar("train/grad_norm", grad_norm, step)
                    # Log extended monitoring metrics
                    writer.add_scalar("train/td_error_mae", m["td_error_mae"], step)
                    writer.add_scalar("train/td_error_mse", m["td_error_mse"], step)
                    writer.add_scalar("train/q_sa_max", m["q_sa_max"], step)
                    writer.add_scalar("train/q_sa_min", m["q_sa_min"], step)
                    writer.add_scalar("train/target_max", m["target_max"], step)
                    writer.add_scalar("train/target_min", m["target_min"], step)
                    writer.add_scalar("train/entropy_mean", m["entropy_mean"], step)
                    writer.add_scalar("train/entropy_min", m["entropy_min"], step)
                    writer.add_scalar("train/entropy_max", m["entropy_max"], step)
            train_time_accum += time.perf_counter() - t1

        # Target network update
        if TARGET_UPDATE_MODE == "soft" and TAU > 0.0:
            with torch.no_grad():
                for p_tgt, p in zip(q_tgt.parameters(), q.parameters()):
                    p_tgt.data.mul_(1.0 - TAU).add_(p.data, alpha=TAU)
        else:
            if step % TARGET_UPDATE == 0:
                q_tgt.load_state_dict(q.state_dict())

        if step - last_log_step >= log_interval:
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

            # Env/Train ratio (in window)
            total_work = env_time_accum + train_time_accum
            if total_work > 0:
                env_ratio = env_time_accum / total_work
                train_ratio = train_time_accum / total_work
                writer.add_scalar("perf/env_time_ratio", env_ratio, step)
                writer.add_scalar("perf/train_time_ratio", train_ratio, step)
            else:
                env_ratio = 0.0
                train_ratio = 0.0

            # Write perf scalars
            writer.add_scalar("perf/steps_per_sec", sps, step)
            writer.add_scalar("perf/proc_cpu_percent_all_cores", cpu_all, step)
            writer.add_scalar("perf/proc_cpu_percent_norm", cpu_norm, step)
            writer.add_scalar("perf/memory_gb", mem_gb, step)
            if torch.cuda.is_available():
                writer.add_scalar("perf/gpu_memory_allocated_gb", gpu_mem_alloc, step)
                writer.add_scalar("perf/gpu_memory_reserved_gb", gpu_mem_res, step)

            # Console
            log_msg = f"Step {step:,} | SPS {sps:.1f} | Buffer {buffer_fill}"
            print(log_msg)
            perf_msg = f"  Performance: {sps:.1f} SPS | CPU: {cpu_norm:.1f}% | RAM: {mem_gb:.2f} GB"
            if torch.cuda.is_available():
                perf_msg += f" | GPU Mem: {gpu_mem_alloc:.2f}/{gpu_mem_res:.2f} GB"
                if gpu_util is not None:
                    perf_msg += f" | GPU Util: {gpu_util:.0f}%"
            perf_msg += f" | Env: {100*env_ratio:.1f}% Train: {100*train_ratio:.1f}%"
            print(perf_msg)

            # Reset window
            env_time_accum = 0.0
            train_time_accum = 0.0
            last_log_step = step
            last_log_time = time.time()

        if step % save_interval == 0:
            ckpt = {
                "step": step,
                "q_state_dict": q.state_dict(),
                "q_tgt_state_dict": q_tgt.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config,
            }
            torch.save(ckpt, save_dir / f"checkpoint_step{step}.pt")

    torch.save({"q_state_dict": q.state_dict(), "config": config}, save_dir / "final_q_model.pt")
    print("Training complete. Model saved.")

    try:
        if nvml is not None:
            nvml.nvmlShutdown()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Value-based RL training for Atari")
    parser.add_argument("--exp_name", type=str, default=None, help="实验名前缀，将加在环境名+时间戳之前")
    args = parser.parse_args()
    main(exp_name=args.exp_name)
