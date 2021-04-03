"""
Deep Deterministic Policy Gradient.
- Off policy
- Requires continuous action space
- Deep Q learning for continuous action spaces

https://spinningup.openai.com/en/latest/algorithms/ddpg.html
https://arxiv.org/pdf/1509.02971.pdf
"""
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from nn import NN
from actor_critic import ActorCritic
from ddpg.cartpole import CartPole

np.random.seed(0)
torch.manual_seed(0)


class Actor(nn.Module):
    def __init__(self, layers, activations, alpha, action_space):
        super().__init__()
        self.action_space = action_space
        self.nn = NN(layers, activations)
        self.opt = optim.Adam(self.nn.parameters(), lr=alpha)
        self.target = deepcopy(self.nn)

    def forward(self, x):
        return self.nn(x)

    def act(self, state):
        epsilon = 
        return torch.clip(self.nn(state) + epsilon, min(self.action_space), max(self.action_space))

    def step(self, history, advantages):
        """
        Update policy,
        loss = mean(Q(s, mu(s)))
        """
        self.opt.zero_grad()

        # TODO for s in history
        loss = Q_phi(s, self.nn(s)).mean()

        loss.backward(retain_graph=True)
        self.opt.step()

        # TODO self.target = p * self.target + (1 - p)self.nn


class Critic(nn.Module):
    def __init__(self, layers, activations, alpha, gamma):
        super().__init__()
        self.gamma = gamma
        self.nn = NN(layers, activations)
        self.criterion = nn.MSELoss()
        self.opt = optim.Adam(self.nn.parameters(), lr=alpha)
        self.target = deepcopy(self.nn)

        self.policy_probs

    def forward(self, x):
        return self.nn(x)

    def targets(self, history):
        """
        Compute targets,
        y(r, s', d) = r + self.gamma(1-d)Q_targ(s', mu_targ(s'))
        """
        # TODO

    def step(self, history, values, **kwargs):
        """
        Fit value fn via regression on MSE.
        values = Q(s, a), targets = self.targets.
        """
        self.opt.zero_grad()
        loss = self.criterion(values, self.targets(history))
        loss.backward()
        self.opt.step()

        # TODO self.target = p * self.target + (1 - p)self.nn


if __name__ == '__main__':
    """
    DDPG Learning the cartpole balancing problem.

    Sucess Metric
    -------------
    Longer cartpole trials, >=200.
    """
    N_EPOCH = 50

    env = CartPole()
    N_INPUT = 4
    N_ACTION = env.action_space.n

    policy = Actor(
        [N_INPUT, 400, 300, N_ACTION],
        [nn.ReLu(), nn.ReLu(), nn.Tanh],
        10**-4,
        [-1, 1]
    )
    critic = Critic(
        [N_INPUT, 400, 300, 1],
        [nn.ReLu(), nn.ReLu(), lambda x: x],
        10**-3
    )
    ac = ActorCritic(
        actor,
        critic,
        gamma=.99,
        epoch_steps=4800,
        episode_len=400,
        win_score=200,
        n_to_win=100,
        action_space=action_space,
    )

    for e in range(N_EPOCH):
        ac.play()
        ac.step()
