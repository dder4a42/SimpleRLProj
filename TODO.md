# Extension Plan (for future development)


## Phase 1 — Stability & Performance (low risk)

 Distributional Q-learning (QR-DQN or IQN)

Improves robustness under partial observability

Essential for Breakout and Boxing

 Munchausen reward augmentation

Adds entropy-aware shaping without a policy network

 Target network soft updates (τ-update)

## Phase 2 — Partial Observability Handling

 Recurrent value network (CNN → GRU → Q)

Required for Boxing and opponent-driven dynamics

 Frame history ablation (stack vs recurrence)

## Design Philosophy (for contributors)

Prefer soft value backups over hard argmax

Treat Atari as a POMDP, not a fully observed MDP

Avoid deterministic policies under action noise

Extend capacity only when failure modes appear

Keep baselines minimal and interpretable