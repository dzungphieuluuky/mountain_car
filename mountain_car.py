import gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")
initial = env.reset()
print(f"Initial: {initial}")

learning_rate = 0.1
discount = 0.95
episodes = 25_000
render_every = 2_000

discrete_os_size = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

print(discrete_os_size)
print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [env.action_space.n]))
# print(f"Q-table:\n {q_table}")
print(f"Observation low: {env.observation_space.low}")
print(f"Observation high: {env.observation_space.high}")

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

for episode in range(episodes):
    if (episode + 1) % render_every == 0:
        render = True
        print(f"Episode: {episode + 1}")
    else:
        render = False
    discrete_state = get_discrete_state(np.array(env.reset()[0], dtype=object))
    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, truncate, info = env.step(action)
        print(f"New state: {new_state}")
        new_discrete_state = get_discrete_state(new_state)
        # make new discrete state positive
        print(f"New discrete state: {new_discrete_state}")
        env.render()
        done = done or truncate
        print(reward, new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state


env.close()

