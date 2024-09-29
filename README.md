[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## blackjack rl repo

This started as a simple blackjack game and turned into a learning exercise. I wanted to build the whole ML stack for a reinforcement learning agent myself, neural network math included, and see how far it could get at the table.

Everything below the reporting layer is standard library Python. The network math (forward pass, backprop, optimizers) is written by hand and tested against worked examples and finite difference gradient checks. matplotlib is the only dependency, and it's just for plots.

![agents playing on the board](docs/img/board.gif)

### what's here

- `project/` - the game engine: cards, decks, hands, dealer, players, multi-seat tables, and a casino that runs many tables
- `neural/` - the neural network library: Linear / ReLU / LeakyReLU / Sigmoid / Dropout layers, MSE and softmax cross-entropy losses, SGD / RMSprop / Adam optimizers, and a Trainer with batching, shuffling, and validation splits
- `train/` - the RL layer: a Deep Q-learning agent and a Monte Carlo control agent built on `neural/`, heuristic baseline players, and an actor/learner setup that scales from a single process up to worker processes running dozens of tables
- `review/` - run metrics, CLI reports, matplotlib plots, and a browser board for watching trained agents play

### things I was practicing

- backprop by hand. The layer gradients are derived manually, checked with finite differences, and there's a unit test that reproduces a worked 2-2-1 example exactly
- value based RL two ways: Q-learning with a target network and replay, next to plain Monte Carlo regression. Comparison in [docs/mc-vs-dqn.md](docs/mc-vs-dqn.md)
- multiprocessing. Worker processes simulate tables and stream experience back to a learner, which broadcasts updated weights out to them
- keeping experiments honest: seeded runs, baselines in every experiment, training curves, and artifacts saved per run

## results

200,000 hands, 25 tables, 100 players, 4 worker processes. Full run averages include the exploration phase, so the numbers below are from the last 20% of hands:

| agent | hands | converged avg reward | converged win % |
| --- | --- | --- | --- |
| basic strategy* | 40,000 | +0.036 | 46.5% |
| dqn | 100,000 | +0.020 | 45.9% |
| mc | 50,000 | +0.008 | 45.6% |
| aggressive | 26,000 | -0.247 | 34.5% |
| random | 24,000 | -0.281 | 32.8% |

\* textbook H17 basic strategy, evaluated separately under the same rules.

Both learners end up about 13 points of win rate above the naive baselines and slightly EV positive because of this engine's dealer rule (see house rules). Textbook basic strategy still edges them out. The agents match it on 77% (dqn) and 71% (mc) of states, and the full comparison, the disagreement charts, and when I'd pick each algorithm are in [docs/mc-vs-dqn.md](docs/mc-vs-dqn.md).

![the network deciding](docs/img/board-decision.png)

## how to run:

```shell
pipenv shell
pip install -r requirements.txt

# run the multi-table casino demo with heuristic players
python main.py

# train agents synchronously (single process, reproducible per seed)
python -m train.main --agents "dqn=2,random=1,noob=1" --workers 0 --tables 1 --hands 20000 --seed 7

# train at scale with multiprocessing: 25 tables, 100 players, 4 worker processes
python -m train.main --agents "dqn=50,mc=25,aggressive=13,random=12" --workers 4 --tables 25 --hands 200000 --seed 7

# re-print a run report, render plots, or compare runs (defaults to the latest run)
make report
python -m review.report runs/<timestamp> --plots
python -m review.report runs/<first> runs/<second>

# watch trained agents play on a live board in the browser (defaults to the latest run)
make board
make board AGENTS="dqn=2,mc=1,random=1"
```

Every training run writes `runs/<timestamp>/` with `config.json`, `metrics.jsonl`, `loss.jsonl`, learned weights per agent kind, `report.txt`, and `plots/*.png` (training loss, rolling win rate, reward comparison, result breakdown, bankroll trajectories, and the exploration schedule).

The board is a local page served with the stdlib `http.server`, no extra dependencies. It animates each hand card by card, shows the network's live value estimates for hit / stay / double while the agent decides, and keeps a filterable play history per seat. Add `?auto=1` to the URL to start dealing on load.

## agents

- `dqn` - Deep Q-learning: the network predicts a value per action, trained off an experience replay buffer with a target network and epsilon-greedy exploration
- `mc` - Monte Carlo control: the network regresses observed episode returns for each state/action visited
- `basic` - textbook H17 basic strategy adapted to this action space, the strongest fixed benchmark
- `noob` / `apprehensive` / `aggressive` / `random` - heuristic baselines (always hit, always stay, double when funded, and uniformly random)

Learning agents see their hand total, a soft ace flag, the dealer upcard, and whether double down is available. Rewards are the hand's net profit as a fraction of the bet.

## house rules

- the dealer hits any hand that could still total 16 or less, so soft 17 through soft 20 all get hit. It busts a lot, and the trained agents learn to take advantage of exactly that
- a natural blackjack (two-card 21) pays 3:2, and a drawn 21 is an ordinary win
- dealer bust pays even money, pushes return the stake
- actions are hit, stay, and double down with flat 10 bets. No splits, insurance, or surrender
- training tables re-buy busted players back to their starting bankroll and count the re-buys. The demo casino plays with real bankroll elimination

Synchronous runs (`--workers 0`) are reproducible bit for bit under a given seed. Parallel runs seed each worker deterministically, but process interleaving means exact replays aren't possible.

## development

```shell
make test      # unit tests across all four packages
make lint      # pylint via .pylintrc
make lint-fix  # black + isort, then pylint
```
