# the reward scale hides the double down

This one came out of the counting experiment sideways, and it turned out to matter more than what I was actually looking for. It affects every learning agent in this repo, including the numbers in [mc-vs-dqn.md](mc-vs-dqn.md), which were measured before I noticed it.

## the symptom

Agents whose per-unit play looked fine were still losing money. From the 400,000 hand counting run, converged numbers:

| agent | ev/unit | units/hand |
| --- | --- | --- |
| dqn | -0.0037 | **-0.0515** |
| mc | +0.0064 | **-0.0534** |
| mc-count | +0.0024 | **-0.0836** |

`mc` earns a positive return per unit wagered and still bleeds half a unit per twenty hands. Those two columns are measuring different things and the gap between them is the whole story: `ev/unit` averages net profit over the *final* bet, hand by hand, while `units/hand` is net profit in table minimums. They only diverge when bet size correlates with outcome.

## where it goes

Splitting the same run by whether the hand was doubled:

| agent | double % | ev/unit when doubling | ev/unit otherwise |
| --- | --- | --- | --- |
| basic strategy | 10.0% | **+0.171** | +0.019 |
| hand-tuned counter | 11.3% | **+0.204** | +0.010 |
| dqn | 12.7% | **-0.377** | +0.050 |
| dqn-count | 12.5% | -0.357 | +0.055 |
| dqn-ramp | 9.5% | -0.350 | +0.057 |
| mc | 38.7% | -0.155 | +0.108 |
| mc-count | 42.8% | -0.201 | +0.155 |
| mc-ramp | 43.2% | -0.159 | +0.147 |

On hands they don't double, **every learner beats basic strategy per unit** — `mc-count` earns +0.155 against the book's +0.019. Then they double, and it's a bloodbath. Basic strategy doubles a tenth of its hands at +0.171. MC doubles four in ten at -0.155.

The doubling is not a small leak around the edges of a decent policy. It is the entire deficit.

## why

Reward is net profit over the final bet. Work through it:

| hand | net | final bet | reward |
| --- | --- | --- | --- |
| lose flat | -1 unit | 1 unit | **-1** |
| lose doubled | -2 units | 2 units | **-1** |
| win flat | +1 unit | 1 unit | **+1** |
| win doubled | +2 units | 2 units | **+1** |

Doubling costs twice as much when it loses and pays twice as much when it wins, and the training signal is **identical in both cases**. The stake is priced at exactly zero.

So the network was never told that doubling risks anything. Its entire preference for the action comes from the mechanics that remain visible — take exactly one more card, then stand — which is why MC, whose targets are raw episode returns, developed such an appetite for it. It was getting the upside of a free extra card with the downside normalised away.

The metric hid it too, for the same reason. The old summary reported `avg reward`, which normalises the same way the training target does, so a policy bleeding money through doubled hands looked roughly break-even. Reporting profit in table minimums is what made it visible.

That normalisation wasn't a careless choice. Making the target bet-size invariant is genuinely useful — it stops a 50 chip hand shouting over a 10 chip hand about how to play 16, and it's exactly right for hit and stand. It just happens to erase the one action in the game that changes how much is at risk.

## the fix, and what it costs

`--reward-scale initial_bet` divides by the opening bet instead of the final one, so a doubled loss trains on -2 and a doubled win on +2:

```shell
python -m train.main --agents "dqn=2,mc=2,basic=1" --reward-scale initial_bet
```

The default is still `final_bet`, deliberately. Flipping it changes what every previously published number in this repo means, and I'd rather have the old results stay honest about the conditions that produced them than quietly redefine them.

What I have not done is re-run the earlier experiments under the corrected scale. That's the obvious next thing, and the prediction is specific enough to be worth writing down before doing it: the learners should double far less, MC's 43% especially, and the gap between `ev/unit` and `units/hand` should mostly close. If it doesn't, my explanation is wrong.

## the general version

The lesson isn't about blackjack. Normalising a reward to remove a nuisance variable is a standard and usually good move, but the nuisance variable here was *also the stake*, and one action in the space controlled it. Anything that both scales the reward and is under the agent's control cannot be normalised away without deleting a real decision from the problem.

The tell was two metrics that should have agreed and didn't. Worth reporting both when a policy can change its own exposure.
