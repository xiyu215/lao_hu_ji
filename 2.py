import numpy as np
import matplotlib.pyplot as plt

np.random.seed(14514)


class Bandit:
    """环境：10 台老虎机，各有固定的真实中奖率。"""

    def __init__(self, arms=10):
        self.arms=arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        self.rates+=np.random.randn(self.arms)*0.1
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    """智能体：用 ε-greedy 选动作，用 1/n 求平均来估计每个臂的价值。"""

    def __init__(self, epsilon, alpha,action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.alpha=alpha

    def update(self, action, reward):
        self.Qs[action] += (reward - self.Qs[action]) *self.alpha

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


steps = 1000
bandit = Bandit()
agent = Agent(epsilon=0.1, alpha=0.8,action_size=10)

total_reward = 0
total_rewards = []
rates = []
for step in range(steps):
    action = agent.get_action()
    reward = bandit.play(action)
    agent.update(action, reward)
    total_reward += reward
    total_rewards.append(total_reward)
    rates.append(total_reward / (step + 1))

print(total_reward)

plt.ylabel("total reward")
plt.xlabel("steps")
plt.plot(total_rewards)
plt.show()

plt.ylabel("rate")
plt.xlabel("steps")
plt.plot(rates)
plt.show()
