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
import gym
import torch
import torch.nn as nn
import torch.optim as optim

from nn import NN
from actor_critic import ActorCritic
from ddpg.cartpole import CartPole

np.random.seed(0)
torch.manual_seed(0)


class Actor(nn.Module):
    def __init__(self, layers, activations, alpha, action_space, epsilon_clip):
        super().__init__()
        self.action_space = action_space
        self.epsilon_clip = epsilon_clip
        self.nn = NN(layers, activations)
        self.opt = optim.Adam(self.nn.parameters(), lr=alpha)
        self.target = deepcopy(self.nn)

    def forward(self, x):
        return self.nn(x)

    def step(self, history, policy_probs, advantages):
        """
        Update policy via maximize ppo-clip objective
        https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl
        argmax 1 / NT sum_N sum_T min(pi_theta / pi_theta_k * A^pi_theta_k, g(epsilon, A^pi_theta_k))
        -> w += gradient. Invert loss output
        PPO Clip adds clipped surrogate objective as replacement for
        policy gradient objective. This improves stability by limiting
        change you make to policy at each step.
        Vanilla pg uses log p of action to trace impact of actions,
        ppo clip uses prob of action under current policy / prob under prev policy
        r_t(theta) = pi_theta(a_t, s_t) / pi_theta_old(a_t | s_t)
        Thus L = E[r * advantages]
        r > 1 if action more probable with current policy, else 1 > r > 0.
        to prevent too large steps from too large r, uses clipped surrogate objective
        ie L_clip = E[min(r * advantages, clip(r, 1 - epsilon, 1 + epsilon) * advantages)]
        epsilon = 2.
        """
        self.opt.zero_grad()

        # for _ in range(N_UPDATES):
        # Sample transition batch, B from history

        # Compute targets,
        # y(r, s', d) = r + gamma(1-d)Q_phi_targ(s', mu_theta_targ(s'))

        # Update Q fn by grad descent via MSE Q(s, a) and y(r, s', d) over B.

        # Update policy via gradient ascent using,
        # grad mean_s in B(Q_phi(s, mu_phi(s)))

        # update target networks with
        # phi_target = p phi_target + (1 - p)phi
        # theta_target = p theta_target + (1 - p)theta

        loss.backward(retain_graph=True)
        self.opt.step()
        self.target = deepcopy(self.nn)


class Critic(nn.Module):
    def __init__(self, layers, activations, alpha):
        super().__init__()
        self.nn = NN(layers, activations)
        self.criterion = nn.MSELoss()
        self.opt = optim.Adam(self.nn.parameters(), lr=alpha)
        self.target = deepcopy(self.nn)

    def forward(self, x):
        return self.nn(x)

    def step(self, values, rtg):
        """
        Fit value fn via regression on MSE.
        """
        self.opt.zero_grad()
        loss = self.criterion(values, rtg)
        loss.backward()
        self.opt.step()
        self.target = deepcopy(self.nn)


if __name__ == '__main__':
    """
    DDPG Learning the cartpole balancing problem.

    Sucess Metric
    -------------
    Longer cartpole trials, >=200.
    """
    N_EPOCH = 50

    env = gym.make('CartPole-v0')
    N_INPUT = 4
    N_ACTION = env.action_space.n

    policy = Actor(
        [N_INPUT, 400, 300, N_ACTION],
        [nn.ReLu(), nn.ReLu(), nn.Tanh],
        10**-4
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
