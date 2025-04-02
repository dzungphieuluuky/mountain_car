import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from utils import get_discrete_state

# hyperparameters
learning_rate = 0.1
discount = 0.95
epsilon_start = 1
epsilon_end = 0.01
epsilon_decay = (epsilon_start - epsilon_end)
episodes = 100_000
max_reward = -120
results = []

def training():
    env = gym.make("MountainCar-v0")
    _, discrete_os_size = get_discrete_state(np.array(env.reset()[0], dtype=object), env)
    q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [env.action_space.n]))
    eps_rewards = []
    episode_store = []
    episode = 1

    while True:
        episode_store.append(episode)
        episode_reward = 0
        discrete_state, _ = get_discrete_state(np.array(env.reset()[0], dtype=object), env)
        done = False
        print(f"Episode: {episode}")

        while not done:
            action = np.argmax(q_table[discrete_state])
            new_state, reward, done, truncate, info = env.step(action)
            new_discrete_state, _ = get_discrete_state(new_state, env)

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
        episode += 1
        if episode_reward >= max_reward:
            print("Training successfully!")
            break
    env.close()
    np.save("model/q_table.npy", q_table)
    results.append(episode_store)
    results.append(eps_rewards)
    np.save('results.npy', results)

if __name__ == "__main__":
    training()