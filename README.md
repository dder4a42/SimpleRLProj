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