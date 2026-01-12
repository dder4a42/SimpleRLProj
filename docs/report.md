# Report

## Value-based on Atari

### Atari Env

雅达利游戏的默认设置为（Boxing-v5）

+ 观测空间：像素空间
+ 动作空间：18 个按键
+ 随机因素：跳帧 、按键粘滞

这对 RL 建模提出了挑战，首先，这是一个部分可观测环境，当跳帧为 4 时，作出一个决策将获得 4 帧的 RGB 观测，但无法得知确切的 RAM 状态；其次，这是一个随机环境，智能体以 p=0.25 概率重复上一次的行为，此时作出决策无法得到确定性响应。对于前者，我们采取 DQN 的处理方法，取上次决策后获得的 4 帧观测，转为灰度图并下采样为 84x84 尺寸堆叠在一起，试图智能体隐式学习：
$$
o_t \to \hat{s}_t \to \hat{Q}(s_t, a_t)
$$
假设给定 RAM 状态能唯一确定观测，则给定观测时， $Q(o_t, a_t)=E[G_t\mid O_t=o_t]$ 满足
$$
Q(o_t, a_t) = E[Q(s_t, a_t)\mid O_t=o_t]
$$
这是神经网络实际的拟合目标。然而，当我们试图使用 TD 算法逼近最优 Q 函数时，
$$
\theta_{k+1} \gets \theta_k + \alpha \left( r + \gamma \max_{a'} Q_k(o', a') - Q_k(o, a) \right) \nabla_\theta Q_k(o, a)
$$
实际上是在 $p(o'\mid s, o, a)$ 上采样计算期望，得到的是不动点
$$
Q(o, a) = E_{S_t}\left[E_{O_{t+1}}[r + \gamma \max_{a'} Q(o', a')\mid S_t]\large\mid O_t=o, A_t=a\right]
$$
由 Jensen 不等式，信息泄露将导致 Q-learning 中在 max 算符作用后高估 Q 值。

### Implementation 

#### DQN

Q 网络由 3 层 CNN 和 1 层全连接组成，并在输入前进行归一化

```python
class QNet(nn.Module):
    def __init__(self, n_actions, in_ch=4, conv_channels=(32, 64, 64), fc_hidden=512):
        super().__init__()
        c1, c2, c3 = tuple(conv_channels) if len(conv_channels) == 3 else (32, 64, 64)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, c1, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(c1, c2, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(c2, c3, 3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * c3, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, n_actions),
        )

    def forward(self, x):
        x = x.float() / 255.0
        x = self.conv(x)
        return self.fc(x.flatten(1))
```

为获得平滑的更新，对 Q target 网络进行指数平滑更新。为抑制高估 Q 值现象，使用 Double DQN 计算 TD 目标。

#### Boltzmann Policy & soft Q-learning

使用玻尔兹曼策略保留 Q 值分布信息对策略的影响，并使用温度衰减让智能体从前期探索转为后期优先选择高 Q 值动作。

使用 soft Q-learning，即将 max 算符替换为 softmax 算符，缓解 max 带来的系统偏差。

### IQN

使用 IQN 将改善 max 带来的高估问题

### Performance 

训练配置如下，请确保运行设备具有至少 30 GB 内存和 2.5 GB 显存（buffer 占用较多内存）

```yaml
# Environment
env:
  name: "ALE/Boxing-v5"
  frameskip: 4
  repeat_action_probability: 0.25  # sticky actions
  frame_stack: 4

# Network
network:
  conv_channels: [32, 64, 64]
  fc_hidden: 512

# Training (step-based)
training:
  total_steps: 10000000          # total environment steps across all vector envs
  gamma: 0.99
  alpha: 0.05                   # base entropy temperature for soft backups
  alpha_start: 0.05             # alpha schedule: start value
  alpha_final: 0.02             # alpha schedule: final value
  alpha_decay_steps: 4000000     # steps over which alpha decays
  lr: 0.00005
  lr_final: 0.00001             # final learning rate for linear decay
  lr_decay_steps: 10000000      # steps over which lr decays linearly; default to total_steps
  batch_size: 1024
  buffer_size: 600000
  warmup_steps: 50000           # begin updates after this many steps collected
  target_update_freq: 10000     # sync target every N steps (hard update)
  target_update_mode: "soft"    # "hard" or "soft" (Polyak)
  tau: 0.003                    # Polyak factor if soft update is enabled
  grad_clip: 10.0
  updates_per_step: 1           # gradient updates per env step (aggregate)
  use_amp: true                 # enable CUDA AMP (mixed precision)
  reward_clip: null      # optional training-time reward clipping; set to null to disable
  target_clip: null             # optional target clamp magnitude; set to null to disable

# Logging (step-based intervals)
logging:
  log_interval: 10000           # log perf stats every N steps (lower for more frequent console updates)
  save_interval: 500000         # save checkpoint every N steps
  log_dir: "logs"
  save_dir: "checkpoints"

# Parallel envs
parallel:
  num_envs: 16
  async: true                   # hint only; AsyncVectorEnv is used
  seed: 42
  enable_cpu_monitor: true      # reserved flag; metrics are always collected
```

在 Boxing-v5 环境中能达到 80 分以上

![](..\source\Boxing-v5.png)

批次大小和每步进行网络更新次数将影响训练速度。

使用 IQN 后，通常能取得比 DQN 更好的效果

<img src="..\source\IQN-whole.png" style="zoom:50%;" />