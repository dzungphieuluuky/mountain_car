import gymnasium as gym
import numpy as np
from utils import get_discrete_state

def play(q_table):
    env = gym.make("MountainCar-v0", render_mode="human")
    final_reward = 0
    done = False
    discrete_state, _ = get_discrete_state(np.array(env.reset()[0], dtype=object), env)

    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, truncate, info = env.step(action)
        new_discrete_state, _ = get_discrete_state(new_state, env)

        if done or truncate:
            break
        
        final_reward += reward
        discrete_state = new_discrete_state
        env.render()
    
    print(f"Final reward: {final_reward}")
    env.close()

if __name__ == "__main__":
    q_table = np.load("model/q_table.npy")
    play(q_table)