############ Imports #############
import gymnasium as gym  #
import matplotlib.pyplot as plt
import pandas as pd

from dqn import Agent

###################################
def plot_episode_returns(returns, final, smoothing_window=20):
    # Plot the episode reward over time
    plt.figure(1)
    returns = pd.Series(returns)
    returns_smooth = returns.rolling(
        smoothing_window, min_periods=smoothing_window
    ).mean()
    plt.clf()
    if final:
        plt.title("Result")
    else:
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Episodic Return")
    plt.plot(ep_returns, label="Raw", c="b", alpha=0.3)
    if len(returns_smooth) >= smoothing_window:
        plt.plot(
            returns_smooth,
            label=f"Smooth (win={smoothing_window})",
            c="k",
            alpha=0.7,
        )
    plt.legend()
    if final:
        plt.show(block=True)
    else:
        plt.pause(0.001)


env = gym.make("LunarLander-v2")
agent = Agent(env=env, gamma=0.99, epsilon=1.0, lr=0.003, batch_size=64, eps_end=0.01)

ep_returns, ep_lens, n_steps = [], [], 0
max_episodes = 500
for ep in range(max_episodes):
    ep_return, ep_len = 0, 0
    done = False
    state, info = env.reset()
    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        ep_len += 1
        n_steps += 1
        agent.store_transition(state, action, reward, next_state, terminated)
        agent.learn()
        if n_steps % agent.targ_update_freq == 0:
            agent.update_target_net()
        state = next_state
        done = terminated or truncated
    ep_returns.append(ep_return)

    print(f"Episode: {ep+1:>4}, Return: {ep_return:>6.0f}, Length: {ep_len:>4}")
    plot_episode_returns(ep_returns, ep == max_episodes - 1, max_episodes // 10)
env.close()

test_env = gym.make("LunarLander-v2", render_mode="human")
done = False
state, info = test_env.reset()
while not done:
    action = agent.choose_action(state)
    next_state, reward, terminated, truncated, info = test_env.step(action)
    state = next_state
    done = terminated or truncated
test_env.close()