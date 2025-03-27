import gym
import numpy as np

def get_discrete_state(state, env, discrete_os_win_size):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

def play(q_table):
    env = gym.make("MountainCar-v0", render_mode="human")
    final_reward = 0
    discrete_os_size = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size
    discrete_state = get_discrete_state(np.array(env.reset()[0], dtype=object), env, discrete_os_win_size)
    done = False

    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, truncate, info = env.step(action)
        new_discrete_state = get_discrete_state(new_state, env, discrete_os_win_size)

        if done or truncate:
            break
        
        final_reward += reward
        discrete_state = new_discrete_state
        env.render()
    
    print(f"Final reward: {final_reward}")
    env.close()

q_table = np.load("q_table.npy")
play(q_table)