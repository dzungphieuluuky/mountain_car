import matplotlib.pyplot as plt
import numpy as np

def plot_results():
    results = np.load('results.npy')
    episodes = results[0]
    rewards = results[1]
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training progress')
    plt.legend()
    plt.grid()

    plt.savefig('results.png')
    plt.show()

if __name__ == "__main__":
    plot_results()
