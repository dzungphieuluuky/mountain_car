import gym
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters
learning_rate = 0.1
discount = 0.95
epsilon_start = 1
epsilon_end = 0.01
epsilon_decay = (epsilon_start - epsilon_end)
episodes = 100_000

def get_discrete_state(state, env, discrete_os_win_size):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))


def training(episodes):
    env = gym.make("MountainCar-v0", render_mode="human")
    discrete_os_size = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size
    # q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [env.action_space.n]))
    q_table = np.load("q_table.npy")
    eps_rewards = []
    episode_store = []
    for episode in range(episodes):
        episode_store.append(episode)
        episode_reward = 0
        discrete_state = get_discrete_state(np.array(env.reset()[0], dtype=object), env, discrete_os_win_size)
        done = False
        print(f"Episode: {episode}")

        while not done:
            action = np.argmax(q_table[discrete_state])
            new_state, reward, done, truncate, info = env.step(action)
            new_discrete_state = get_discrete_state(new_state, env, discrete_os_win_size)

            if done or truncate:
                break

            episode_reward += reward

            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action, )]
                new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
                q_table[discrete_state + (action, )] = new_q

            elif new_state[0] >= env.goal_position:
                done = True

            discrete_state = new_discrete_state

        eps_rewards.append(episode_reward)
        print(f"Episode reward: {episode_reward}")
    env.close()
    np.save("q_table.npy", q_table)


training(episodes)