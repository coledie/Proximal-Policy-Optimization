"""
Deep Deterministic Policy Gradient.

https://spinningup.openai.com/en/latest/algorithms/ddpg.html
"""
from copy import deepcopy
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns


class NN(nn.Module):
    def __init__(self, layers, activations):
        super(NN, self).__init__()
        self.activations = activations
        self.layers = nn.ModuleList()
        for i, layer in enumerate(layers[:-1]):
            self.layers.append(nn.Linear(layer, layers[i+1]))

    def forward(self, x):
        x = torch.Tensor(x)
        for i, layer in enumerate(self.layers):
            x = self.activations[i](layer(x))
        return x


def normalize(obs):
    return obs


if __name__ == '__main__':
    """
    DDPG Learning the cartpole balancing problem.

    Sucess Metric
    -------------
    Longer cartpole trials, >=200.
    """
    np.random.seed(0)
    torch.manual_seed(0)

    WIN_SCORE = 200
    N_TO_WIN = 100

    N_EPOCH = 50
    EPOCH_STEPS = 4800
    EPISODE_LEN = 400

    ALPHA = .005
    GAMMA = .95
    EPSILON_CLIP = .2

    ##
    env = gym.make('CartPole-v0')
    N_INPUT = 4
    LATENT_SIZE = 64
    N_ACTION = env.action_space.n
    action_space = [i for i in range(env.action_space.n)]

    policy = NN([N_INPUT, LATENT_SIZE, LATENT_SIZE, N_ACTION], [nn.Tanh(), nn.Tanh(), nn.Softmax(dim=-1)])
    policy_old = deepcopy(policy)
    value = NN([N_INPUT, LATENT_SIZE, LATENT_SIZE, 1], [nn.Tanh(), nn.Tanh(), lambda x: x])

    opt_policy = optim.Adam(policy.parameters(), lr=ALPHA)
    criterion_value = nn.MSELoss()
    opt_value = optim.Adam(value.parameters(), lr=ALPHA)

    lengths, eps = [], []
    for e in range(N_EPOCH):
        opt_policy.zero_grad()
        opt_value.zero_grad()

        # Set target parameters equal to main parameters, theta_targ = theta, phi_targ = phi.
        # TODO

        # Compute trajectories D by running current policy
        n_steps = 0
        e_count = 0
        history, values, policy_probs = [], torch.zeros(EPOCH_STEPS), torch.zeros((EPOCH_STEPS, N_ACTION))
        while n_steps < EPOCH_STEPS:
            e_count += 1
            observation = env.reset()
            state = normalize(observation)
            for l in range(EPISODE_LEN):
                policy_choice = policy(state) + 1e-10
                action = np.random.choice(action_space, p=policy_choice.detach().numpy() / float(policy_choice.sum()))
                observation, reward, done, info = env.step(action)
                state_next = normalize(observation)

                history.append((state, action, reward, state_next))
                values[n_steps] = value(state)
                policy_probs[n_steps] = policy_choice
                state = state_next

                n_steps += 1
                if done or n_steps == EPOCH_STEPS:
                    break

            lengths.append(l+1)

            if np.mean(lengths[-N_TO_WIN:]) >= WIN_SCORE:
                print("Win!")

        eps.append(e_count)
        print(f"{e}: {np.mean(lengths[-e_count:]):.0f}")

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

    """
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    eps_scatter = []
    for i, v in enumerate(eps):
        eps_scatter.extend([i for _ in range(v)])
    sns.violinplot(eps_scatter, lengths, ax=ax1)

    X = np.arange(len(history))
    ax2.plot(X, values.detach().numpy(), label="v")
    ax2.plot(X, rtg, label="rtg")
    ax2.plot(X, advantages.detach().numpy(), label="A")
    ax2.legend()
    plt.show()
    """
