# 🏆 RL Agent for MountainCar-v0

## 🎯 Overview
This project implements a **Reinforcement Learning (DRL) agent** using **Q-learning** to solve **MountainCar-v0** from Gymnasium.

## 🚀 Key Feature
- Train using Q-learning.
- Use Q-table to represent q-value of each state-action pairs.
- State discretization required.

## 🎖️ Architecture
- Environment: MountainCar-v0.
- Discretization: discretize states into separate buckets.
- Q-learning: maximize q-value over iteration.
- Training progress: around 2000 episodes with threshold rewards = -120.

## 🌹 Dependencies (Windows)
```bash=
pip install -r requirements.txt
```

## 🐼 References
**sentdex** playlist on **Reinforcement Learning**.

## 🐧 Usage
- Run `train.py` to train the agent.
- Results will automatically be saved in `results.npy.`
- Model file is saved in `model` folder.
- Run `play.py` to watch the agent play.
- Run `plot_results.py` to watch the training progress graph.
