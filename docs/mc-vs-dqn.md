# monte carlo control vs deep q-learning

Both agents use the same network from `neural/`: 4→32→32→3, ReLU, Adam at 1e-3, 1,315 parameters. Same observation (hand total, soft ace flag, dealer upcard, whether double is available), same epsilon schedule, same reward (net profit on the hand as a fraction of the bet). The only difference is the target they train toward, so this ends up being a pretty clean comparison of the two ideas.

## the two targets

Monte Carlo waits until the hand is over, then trains every state/action from that hand toward the return that actually happened. There is no estimate inside the target. It's just regression on real outcomes, with no target network or anything else bolted on.

DQN updates after every action using its own guess about the next state: `target = reward + gamma * max Q_target(next state)` over the legal actions. The replay buffer and the periodically synced target network exist to keep that self reference from blowing up, and each update only touches the action that was actually taken.

## results

From the 200,000 hand run (25 tables, 100 players, 4 worker processes, seed 7):

| agent | hands | full run avg reward | converged avg reward* | converged win % | converged bust % |
| --- | --- | --- | --- | --- | --- |
| basic strategy** | 40,000 | +0.036 | +0.036 | 46.5% | 17.8% |
| dqn | 100,000 | -0.096 | **+0.020** | 45.9% | 14.8% |
| mc | 50,000 | -0.090 | **+0.008** | 45.6% | 10.4% |
| aggressive | 26,000 | -0.270 | -0.247 | 34.5% | 37.9% |
| random | 24,000 | -0.289 | -0.281 | 32.8% | 32.4% |

\* last 20% of each agent's hands, after epsilon decayed. The full run average includes the early exploration phase, so it understates the final policy.

\** textbook H17 basic strategy (hit / stay / double only), evaluated separately for 40,000 hands under the same rules and seed. It doesn't learn, so its full run and converged numbers are the same.

They converge to basically the same play, about 13 points of win rate above random. The positive EV is real but it comes from the house rules: the dealer here hits any hand that could still total 16 or less, which includes soft 17 through soft 20. It busts constantly, and both agents learn to stand on stiff hands and wait for that to happen. MC busting less (10.4% vs 14.8%) is the same lesson learned a little more conservatively.

![rolling win rate](img/win_rates.png)

![reward comparison](img/reward_comparison.png)

MC also got there with about half the hands, which you can see in its steeper early curve. With only 1 to 5 decisions per hand, the episode result is nearly as good a signal as a bootstrapped per step target, so MC barely pays anything for being simple. DQN trained smoothly, but its extra machinery mostly bought stability rather than better play.

## what the networks actually learned

Since the observation is only four numbers, the whole policy can be drawn. These charts come from running each trained network over every state (double assumed available) and coloring the argmax action, the same way basic strategy cards are laid out:

![learned policy charts](img/policy_charts.png)

This is my favorite result in the project. The two agents earn almost the same EV with visibly different styles. DQN plays it safe: a thin double region and an early stand line. MC doubles hard 9 through 12 almost everywhere, which is close to what real basic strategy says, and it doubles a wide band of soft hands too. That appetite for doubling is also why MC swings plus and minus 20 so often in the play history. Both agents stand earlier than a standard strategy card would, and that's the dealer rule again: when the dealer hits soft 17 through 20 it busts so often that standing on a stiff hand is worth more here than in a normal game.

The raw weights are less interpretable but worth a look:

![learned weights](img/network_weights.png)

The strongest stripes in the input layer sit on the player-total feature in both networks, so both learned to care most about their own hand. The middle 32x32 layer is mostly small values with a few strong channels, and the output layer shows each action reading its own distinct mix of those channels.

## vs basic strategy

The obvious question is how the learned policies stack up against the prevailing strategy. I encoded textbook multi-deck H17 basic strategy (restricted to hit, stay, and double, since this game has no splits or surrender) as a `basic` agent kind, so it can sit at any table like the other heuristics. Two caveats make this comparison interesting rather than a formality. First, basic strategy was derived for a dealer that stands on 17; this engine's dealer keeps hitting any hand that could still total 16 or less, so soft 17 through soft 20 all get hit and the dealer busts far more than in a real casino. Second, that means the book isn't the true optimum here either. It's just the strongest fixed benchmark available.

Result: over 40,000 hands under the same rules and seed, basic strategy averages **+0.036** per hand with a 46.5% win rate. That beats DQN's converged +0.020 and MC's +0.008. Both agents got most of the way from random (-0.28) to the book, but neither closed the gap.

The disagreement chart shows exactly where the money went:

![learned policy vs basic strategy](img/policy_vs_basic.png)

Outlined cells differ from the book. DQN matches basic on 77% of states and MC on 71%, and the disagreements split into two kinds. Some look like genuine adaptations to this ruleset: both agents stand on 14-16 against strong upcards where basic says hit, which is plausible when the dealer busts this often. Others are just EV left on the table: DQN almost never doubles hard 9-11, which is where basic strategy makes most of its money, and that missing double region accounts for a good share of the gap. MC doubles 9-12 like the book but keeps doubling 13-16 into strong upcards, which is generosity the dealer doesn't deserve even here.

I like this result more than a clean win would have been. The agents rediscovered most of a strategy that took decades of analysis to tabulate, adapted parts of it to a house rule the book doesn't know about, and still left a measurable gap that's visible cell by cell in the chart.

## which one I'd use

MC when hands are short and the reward only shows up at the end, which is exactly blackjack. The targets are unbiased, training is ordinary regression, and there are fewer things to mistune since there's no target network and no bootstrap bias to chase. It's also easier to debug, because a weird target always traces back to a hand that actually happened.

DQN when episodes get long or rewards arrive mid episode. Waiting for the end makes MC targets noisier as hands get longer (one lucky dealer bust rewards every action in the hand equally), while bootstrapping hands out credit per step. The replay buffer also squeezes more out of each experience, which matters when data is expensive instead of simulated for free.

For this game MC is the better deal for the effort. DQN was still the more useful one to build, because the target network and masked target details are where things actually go wrong in practice.
