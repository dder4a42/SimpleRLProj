# Report

## Value-based on Atari

### Atari Env

Gym 中的雅达利游戏包含 ALE 命名空间下的一系列环境，v5 版本的默认设置为

+ 观测空间：RGB 图像
+ 随机因素：跳帧（frame skip=4） 、按键粘滞（repeat probability=0.25）

其中，跳帧意味着智能体每作出一个决策，环境将运行 4 帧，且仅返回最后一帧观测，模拟人类玩家反应时间。按键粘滞则意味着以一定概率重复上一次决策，模拟真实硬件条件。尽管雅达利环境是一个确定性环境，即完全由内存状态决定，但按键粘滞引入了随机性，0.25 的重复概率加大了学习难度。此时，IQN 建模回报的分布，捕捉这类随机性，在 ALE/Boxing-v5 环境中取得比 DQN 更好的性能。



跳帧问题较为困难。首先，RGB 图像为环境的部分观测，缺少速度信息，分数和生命值也需要从屏幕上读取。另外，游戏内存在大量计数器和冷却时间（硬直时间）逻辑，将影响智能体预判敌人行动。幸运地是，由于所指定的 4 个任务视觉对象较少，不会使用 Flickering 闪烁渲染，造成时序不一致性。

其次，跳帧意味着一个决策将影响多帧，而智能体仅能得到最后一帧，造成信息缺失。DQN 的常见处理方法为，取最近获得的 4 帧观测（游戏运行得很快，4 帧其实肉眼难以看出差异），转为灰度图并下采样为 84x84 尺寸堆叠在一起，相当于从环境最近的 16 帧状态中抽帧，并试图智能体从历史观测序列中学习：
$$
o_t \to \hat{s}_t \to \hat{Q}(s_t, a_t)
$$
当然，神经网络的隐式表达难以解读，不妨考虑观测空间，记观测空间上的价值函数 $Q(o_t, a_t)=E[G_t\mid O_t=o_t]$ ，理想情况下满足
$$
Q(o_t, a_t) = E[Q(s_t, a_t)\mid O_t=o_t]
$$
遗憾的是，此处的历史 RGB 序列并不由当前 RAM 状态完全决定，该公式不成立，通过估计 Q 的分布不能直接解决这个问题。尽管如此，还是能尝试从观测序列中尝试估计当前 RAM 状态。然而，上述得种种因素将导致环境噪声较大。在高方差条件下，TD 算法得收敛条件难以保障，容易出现振荡不收敛得情况。

在实际训练中发现，智能体的学习存在明显的两级分化现象：在早期就能不断提升回报的训练中，智能体往往最终能达到较高水平；若在一段时间内不见起色，则将训练时间延长若干倍后，任务的回报仍然在原水平振荡。可以认为，return 曲线的振荡程度一定程度上反应该智能体的学习潜力。

> 当然，更直接的指标是 grad_norm 等训练过程量。若梯度范数振荡发散，则智能体将进行一次相对失败的学习。

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

使用玻尔兹曼策略保留 Q 值分布信息对策略的影响，即 $\pi(a|s) = \frac{e^{Q(o)/\alpha}}{Z}$ 。为保证数值稳定性，分子分母同除最大 $Q(o)$ 。使用线性温度衰减让智能体从前期探索转为后期优先选择高 Q 值动作。

```python
with torch.no_grad():
    logits = self.q(obs) / alpha
    logits -= logits.max(1, keepdim=True).values
    probs = torch.softmax(logits, dim=1)
    actions = torch.multinomial(probs, 1).squeeze(1)
```



使用 soft Q-learning，即将 max 算符替换为 softmax 算符，缓解 max 带来的系统偏差。

```
entropy = -(pi * (pi + 1e-8).log()).sum(1)
v_next = (pi * q_next_t).sum(1) - alpha * entropy
```



#### IQN

理论上，使用 IQN 将改善 max 带来的 Q 值高估问题。在雅达利环境中，则可以视为对各种不确定性因素进行建模处理的手段，降低 DQN 对真实 Q 值得估计难度。



### Performance 

#### Boxing-v5

拳击任务的敌人攻击性强，喜欢追身殴打，奖励稠密，学习较为顺利。当发现智能体进入瓶颈期或不时掉点，往往降低学习率就能更上一层楼，但需要更长的训练时间。

> 对人类玩家并不友好，被堵在墙角暴打，只能拿到 13 分。

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
  reward_clip: null      		# optional training-time reward clipping; set to null to disable
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

> 由于训练吃 CPU 和内存，训练时长受 OS 调度的影响很大，请尽量避免挂后台刷剧。



DQN 在 Boxing-v5 环境中能达到 80 分以上的表现

![](..\source\Boxing-v5.png)

在可怜的几次超参数实验中，发现批次大小和 updates_per_step 将影响训练速度。（调一调能从 60 上升到 80 分）



使用 IQN 后，能取得比 DQN 更好的效果，在 AEL/Boxing-v5 环境中达到 100 分。

<img src="..\source\IQN.png"  />

#### Pong-v5

乒乓球任务尽管动作空间小，探索简单，但仍然学习困难。打乒乓球时，发现智能体往往从 -21 分上升到 0 分就停滞不前，原因是此时处于相持阶段，接不好球就会丢分，很难尝试主动打怪球的行动。

> 由于 git merge 时的失误，这个权重遗失了

![](D:\sjtu_stuff\RL\Final\source\Pong-v5.png)

首先，敌人动得很频繁，乒乓队存在碰撞反弹，而帧跳跃使得采样频率远小于环境变化速度，导致信息损失较为严重，从帧堆叠中推测的运动和真实运动的差异较大。如下图所示，球拍已经上下走位两次，但观测帧仅获得最后一帧结果。另外，动作粘滞使得在 4 帧中被执行的动作完全不确定，因而 DQN 更趋向于学习一个保守策略。

![](D:\sjtu_stuff\RL\Final\source\Pong-frame.png)

> 在剩下两个环境中，DQN 始终在最低分蠕动，在此不做展示。

### RAM mode

在雅达利游戏中，可以指定将内存状态作为观测空间。除了获得完全信息外，128 维的 RAM 状态还能节约硬件资源。然而，两者的表征学习难度不同，难以设置合适参数量的网络进行对比。经过粗略的实验发现，当使用两层 MLP 即参数量约为 CNN 的 0.25 倍时，就能够实现略高于像素观测的表现。



## Policy-based on MUJOCO

MuJoCo 环境指定了 4 个多关节机器人的行走类任务，动作空间为各关节扭矩，观测空间为机器人位姿及各节点速度，除去接触力未知外，均为完全观测。整体而言，这四个任务不算困难，智能体只需要掌握一种行走模式，就能在环境交互中得到单调上升的分数。

![](D:\sjtu_stuff\RL\Final\source\PPO_MUJOCO_ALL.png)

使用 PPO 算法解决该连续控制问题，网络主干为两层宽度为 64 的全连接层，动作头输出高斯策略的均值，策略头输出价值函数估计，而高斯策略的方差作为独立可学习参数。
