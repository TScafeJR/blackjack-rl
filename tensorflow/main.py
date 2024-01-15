from tensorforce import Agent, Environment


def main():
       environment = Environment.create(environment='benchmarks/configs/cartpole.json')
    agent = Agent.create(agent='benchmarks/configs/ppo.json', environment=environment)

    # Train for 100 episodes
    for episode in range(100):

        # Episode using act and observe
        states = environment.reset()
        terminal = False
        sum_rewards = 0.0
        num_updates = 0
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            num_updates += agent.observe(terminal=terminal, reward=reward)
            sum_rewards += reward
        print('Episode {}: return={} updates={}'.format(episode, sum_rewards, num_updates))

    # Evaluate for 100 episodes
    sum_rewards = 0.0
    for _ in range(100):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent.act(
                states=states, internals=internals, independent=True, deterministic=True
            )
            states, terminal, reward = environment.execute(actions=actions)
            sum_rewards += reward
    print('Mean evaluation return:', sum_rewards / 100.0)

    # Close agent and environment
    agent.close()
    environment.close()
