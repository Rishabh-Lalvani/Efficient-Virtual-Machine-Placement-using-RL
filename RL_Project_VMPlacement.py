import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium

# Custom Environment for VM Placement
class VMPlacementEnv(gymnasium.Env):
    def __init__(self, num_pms=500, workload_size=100):
        super(VMPlacementEnv, self).__init__()
        self.action_space = gymnasium.spaces.Discrete(num_pms)
        self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(num_pms, 4), dtype=np.float32)
        
        self.state = np.zeros((num_pms, 4))  # CPU, Memory, Disk, Network usage
        self.power_consumption = np.zeros(num_pms)
        self.workload_size = workload_size
        self.total_power_usage = []

    def generate_vm_request(self):
        vm_classes = ['small', 'medium', 'large']
        vm_class = np.random.choice(vm_classes)
        if vm_class == 'small':
            return {'cpu': 0.1, 'memory': 0.1, 'disk': 0.05, 'network': 0.05, 'lifetime': np.random.randint(1, 10)}
        elif vm_class == 'medium':
            return {'cpu': 0.25, 'memory': 0.25, 'disk': 0.2, 'network': 0.1, 'lifetime': np.random.randint(5, 15)}
        else:
            return {'cpu': 0.5, 'memory': 0.5, 'disk': 0.4, 'network': 0.25, 'lifetime': np.random.randint(10, 30)}

    def step(self, action):
        if not self.workload:
            return self.state, 0, True, {}

        vm_request = self.workload.pop(0)
        pm = self.state[action]

        if (pm[0] + vm_request['cpu'] <= 1.0 and
            pm[1] + vm_request['memory'] <= 1.0 and
            pm[2] + vm_request['disk'] <= 1.0 and
            pm[3] + vm_request['network'] <= 1.0):

            self.state[action][0] += vm_request['cpu']
            self.state[action][1] += vm_request['memory']
            self.state[action][2] += vm_request['disk']
            self.state[action][3] += vm_request['network']

            reward = self.calculate_reward()
            self.power_consumption[action] = self.calculate_power_usage(action)
            self.total_power_usage.append(np.sum(self.power_consumption))

            done = len(self.workload) == 0
            return self.state, reward, done, {}
        else:
            return self.state, -10, False, {}

    def calculate_reward(self):
        resource_usage = np.mean(self.state, axis=0)
        efficiency = 1.0 - np.std(self.state)
        return efficiency * 10

    def calculate_power_usage(self, pm_index):
        return self.state[pm_index][0] * 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros((self.action_space.n, 4))
        self.power_consumption = np.zeros(self.action_space.n)
        self.workload = [self.generate_vm_request() for _ in range(self.workload_size)]
        self.total_power_usage = []
        return self.state

    def plot_performance(self, rewards_over_time, power_consumption_over_time):
        plt.figure(figsize=(12, 5))
        
        # Plot total rewards
        plt.subplot(1, 2, 1)
        plt.plot(rewards_over_time, color='b')
        plt.title("Total Reward Over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")

        # Plot power consumption
        plt.subplot(1, 2, 2)
        plt.plot(power_consumption_over_time, color='r')
        plt.title("Total Power Consumption Over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Power Consumption (Watts)")
        
        plt.tight_layout()
        plt.show()

env = VMPlacementEnv()

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# DQN Agent with experience replay
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-Network and target network
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = []
        self.memory_limit = 50000
        self.batch_size = 64
        self.update_target_every = 50  # Frequency of target network updates

    def calculate_reward(self,action):
        efficiency = 1.0 - np.std(self.state)
        energy_penalty = self.calculate_power_usage(action) * 0.01
        return efficiency * 10 - energy_penalty * 0.1

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_dim))  # Random action
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def store_experience(self, experience):
        self.memory.append(experience)
        if len(self.memory) > self.memory_limit:  # Limit replay buffer size
            self.memory.pop(0)

    def sample_experiences(self):
        return random.sample(self.memory, self.batch_size)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of experiences
        batch = self.sample_experiences()
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Compute Q(s, a) and target Q-values
        q_values = self.q_network(states).gather(1, actions).squeeze()
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss calculation
        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        # Copy weights to the target network
        self.target_network.load_state_dict(self.q_network.state_dict())

# Training the agent with tracking and plotting
def train_dqn(agent, env, episodes=100):
    rewards_over_time = []
    power_consumption_over_time = []

    for episode in range(episodes):
        state = env.reset()
        state = state.flatten()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            agent.store_experience((state, action, reward, next_state, done))

            # Train the agent on a batch of experiences
            agent.train()
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward

        # Store metrics after each episode
        rewards_over_time.append(total_reward)
        power_consumption_over_time.append(np.sum(env.power_consumption))

        # Decay epsilon
        agent.update_epsilon()

        # Update the target network every few episodes
        if episode % agent.update_target_every == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon}")

    # Plot performance after training
    env.plot_performance(rewards_over_time, power_consumption_over_time)

# Initialize DQN Agent and train
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]  # Flattened state dimension
action_dim = env.action_space.n
agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
train_dqn(agent, env)