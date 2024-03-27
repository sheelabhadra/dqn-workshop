############ Imports #############
import numpy as np  # math stuff
from copy import deepcopy

import torch as th  # pytorch
import torch.nn as nn  # linear layers
import torch.nn.functional as F  # for activation function
import torch.optim as optim  # optimizer

###################################

# We'll create 2 classes; (1) the deep Q-network and (2) the agent.
# The agent is not the DQN.
# The agent has the DQN (its learning module/brain), memory, chooses actions.


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = th.device(
            "cuda:0" if th.cuda.is_available() else "cpu"
        )  # makes use of GPU if available
        self.to(self.device)  # put the entire network on the device

    # forward propagation
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent:
    def __init__(
        self,
        env,
        gamma,
        epsilon,
        lr,
        batch_size,
        memory_size=1000000,
        eps_end=0.01,
        eps_decay=5e-4,
        targ_update_freq=100,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.lr = lr
        self.targ_update_freq = targ_update_freq

        self.input_dims = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory_ctr = 0

        # Q-network
        self.q_net = DeepQNetwork(
            self.lr,
            n_actions=self.n_actions,
            input_dims=self.input_dims,
            fc1_dims=64,
            fc2_dims=64,
        )
        # Target network
        self.targ_net = deepcopy(self.q_net)

        # Replay memory
        self.states = np.zeros((self.memory_size, self.input_dims), dtype=np.float32)
        self.next_states = np.zeros(
            (self.memory_size, self.input_dims), dtype=np.float32
        )
        self.actions = np.zeros(self.memory_size, dtype=np.int32)
        self.rewards = np.zeros(self.memory_size, dtype=np.float32)
        self.dones = np.zeros(self.memory_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        idx = self.memory_ctr % self.memory_size
        (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        ) = (state, action, reward, next_state, done)
        self.memory_ctr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = th.tensor([observation]).to(self.q_net.device)
            actions = self.q_net.forward(state)
            action = th.argmax(actions).item()
        else:
            action = np.random.choice(self.n_actions)

        return action

    def learn(self):
        if self.memory_ctr < self.batch_size:
            return

        batch = np.random.choice(
            min(self.memory_ctr, self.memory_size), self.batch_size, replace=False
        )

        state_batch = th.tensor(self.states[batch]).to(self.q_net.device)
        reward_batch = th.tensor(self.rewards[batch]).to(self.q_net.device)
        next_state_batch = th.tensor(self.next_states[batch]).to(self.q_net.device)
        dones_batch = th.tensor(self.dones[batch]).to(self.q_net.device)

        action_batch = self.actions[batch]

        q_vals = self.q_net.forward(state_batch)[
            np.arange(self.batch_size, dtype=np.int32), action_batch
        ]
        q_next = self.targ_net.forward(next_state_batch)
        q_next[dones_batch] = 0

        with th.no_grad():
            q_targ = reward_batch + self.gamma * th.max(q_next, dim=1)[0]

        loss = self.q_net.loss(q_vals, q_targ)

        self.q_net.optimizer.zero_grad()
        loss.backward()
        self.q_net.optimizer.step()

        self.epsilon = (
            self.epsilon - self.eps_decay
            if self.epsilon > self.eps_end
            else self.eps_end
        )

    def update_target_net(self):
        self.targ_net = deepcopy(self.q_net)
