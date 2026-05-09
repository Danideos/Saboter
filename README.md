# saboter

`saboter` is a dependency-light, single-round simulator for the base game of
Saboteur. It is designed for reinforcement-learning experiments: exact
legal-action generation, hidden-information-safe observations, baseline agents,
and a PettingZoo-style turn API.

```python
from saboter import SaboteurEnv
from saboter.agents import LegalRandomAgent

env = SaboteurEnv(num_players=5)
agent = LegalRandomAgent(seed=0)
env.reset(seed=1)

while not env.is_terminal():
    player = env.agent_selection
    action = agent.act(env, player)
    env.step(action)

print(env.outcome, env.rewards())
```

## Baselines and Evaluation

Available baseline names:

- `random`
- `greedy-miner`
- `greedy-saboteur`
- `role-aware`
- `role-inference`

Run a tournament from the CLI:

```bash
PYTHONPATH=src .venv/bin/python -m saboter.evaluation \
  --games 100 \
  --players 5 \
  --agents role-aware \
  --seed 0 \
  --replay-out replays.json
```

The summary reports miner/saboteur win rates, average game length, action
counts, average reward by agent name and role, and illegal-action attempts.
Replay files include compact public histories plus final roles/rewards for
debugging.

Show one rendered demo game:

```bash
PYTHONPATH=src .venv/bin/python -m saboter.evaluation \
  --games 1 \
  --players 5 \
  --agents role-aware \
  --seed 0 \
  --show-game
```

Write an interactive HTML replay viewer:

```bash
PYTHONPATH=src .venv/bin/python -m saboter.evaluation \
  --games 1 \
  --players 5 \
  --agents role-aware \
  --seed 33 \
  --html-out demo_seed33.html
```

Neural observations, PPO training, league play, and search layers are still
intentionally deferred until the simulator and baselines are well covered.

## Neural Encoders

The simulator exposes dependency-free feature encoders for policy/value models:

```python
from saboter.action_encoding import encode_legal_action_batch
from saboter.observation import encode_observation

features = encode_observation(env, env.agent_selection)
action_batch = encode_legal_action_batch(env, env.agent_selection)

print(features.board_shape)  # (channels, height, width)
print(len(action_batch), "legal actions")
```

The observation encoder returns a fixed board tensor plus hand, player, global,
and recent-history features. The action encoder returns one fixed-width vector
per legal action for action-scoring policies.

## Neural Smoke Test

The first neural milestone is an untrained PyTorch policy that scores only legal
actions and can complete full games. PyTorch is optional:

```bash
.venv/bin/python -m pip install -e ".[neural]"
.venv/bin/python scripts/smoke_test_neural_agent.py --games 1000 --players 5 --seed 0
```

This does not train the model. It only verifies the closed loop from env state
to encoded tensors, legal-action scores, selected actions, and terminal games.

Collect PPO-ready rollouts without learning:

```bash
.venv/bin/python scripts/collect_rollouts_smoke.py --games 100 --players 5 --device cuda
```

Run the first minimal PPO trainer:

```bash
.venv/bin/python scripts/train_ppo.py \
  --iterations 200 \
  --games-per-iter 256 \
  --num-workers 8 \
  --ppo-epochs 4 \
  --batch-size 256 \
  --players 5 \
  --device cuda \
  --save-dir runs/ppo_fast_v1
```

Run the graph action-node PPO baseline:

```bash
.venv/bin/python scripts/train_ppo.py \
  --model graph \
  --iterations 30 \
  --games-per-iter 128 \
  --players 5 \
  --num-workers 8 \
  --worker-torch-threads 1 \
  --device cuda \
  --batch-size 256 \
  --ppo-epochs 3 \
  --hidden-dim 256 \
  --graph-layers 3 \
  --role-belief-coef 0.05 \
  --goal-belief-coef 0.05 \
  --save-dir runs/ppo_graph_v1_probe
```

Profile the training pipeline by stage:

```bash
.venv/bin/python scripts/profile_training_pipeline.py \
  --device cuda \
  --players 5 \
  --rollout-games 32 \
  --worker-counts 1 2 4 8 \
  --worker-torch-threads 1 \
  --batch-sizes 128 256 512
```

## Development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
```

For a longer random-game soak test:

```bash
.venv/bin/python -m pytest tests/test_env.py::test_random_agent_soak_games_complete_without_illegal_actions --runslow
```
