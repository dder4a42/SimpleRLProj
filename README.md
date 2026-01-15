PPO on MuJoCo

```shell
# HalfCheetah (default)
python train.py --algorithm ppo --config configs/ppo_mujoco.yaml --env HalfCheetah-v4

# Other MuJoCo environments
python train.py --algorithm ppo --config configs/ppo_mujoco.yaml --env Hopper-v4
python train.py --algorithm ppo --config configs/ppo_mujoco.yaml --env Walker2d-v4
python train.py --algorithm ppo --config configs/ppo_mujoco.yaml --env Ant-v4
```

SAC on MuJoCo

```bash
# HalfCheetah (default)
python train.py --algorithm sac --config configs/sac_mujoco.yaml --env HalfCheetah-v4

# Other MuJoCo environments
python train.py --algorithm sac --config configs/sac_mujoco.yaml --env Hopper-v4
python train.py --algorithm sac --config configs/sac_mujoco.yaml --env Walker2d-v4
python train.py --algorithm sac --config configs/sac_mujoco.yaml --env Ant-v4
```

Atari value-based: POMDP ablation (pixel vs RAM)

```bash
# 4-way ablation configs (pixel/ram × frame_stack=1/4)
# Runs both DQN and IQN across a few default ALE envs and seeds.
bash scripts/run_pomdp_ablation.sh

# Or specify envs explicitly
bash scripts/run_pomdp_ablation.sh ALE/Pong-v5 ALE/Breakout-v5
```