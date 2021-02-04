"""
Proximal policy optimization. PPO-clip version.
- on-policy
- continuous

https://spinningup.openai.com/en/latest/algorithms/ppo.html
https://medium.com/swlh/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a
https://towardsdatascience.com/proximal-policy-optimization-ppo-with-sonic-the-hedgehog-2-and-3-c9c21dbed5e
https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
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
    PPO Learning the cartpole balancing problem.

    Sucess Metric
    -------------
    Bellman resisual approaching 0.
    Longer cartpole trials.
    """
    np.random.seed(0)
    torch.manual_seed(0)

    N_EPOCH = 50
    EPOCH_STEPS = 4800
    EPISODE_LEN = 400

    ALPHA = .05
    GAMMA = .99
    EPSILON_CLIP = .2

    env = gym.make('CartPole-v0')
    action_space = [i for i in range(env.action_space.n)]

    N_INPUT = 4
    N_ACTION = env.action_space.n
    LATENT_SIZE = 64

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

        # 3. Compute trajectories D by running current policy
        n_steps = 0
        e_count = 0
        history, values, policy_probs = [], torch.zeros(EPOCH_STEPS), torch.zeros((EPOCH_STEPS, N_ACTION))
        while n_steps < EPOCH_STEPS:
            e_count += 1
            observation = env.reset()
            state = normalize(observation)
            for l in range(EPISODE_LEN):
                policy_choice = policy(state) + .000000000001
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
        eps.append(e_count)

        print(f"{e}: {np.mean(lengths[-e_count:]):.0f}")

        # 4. Rewards to go, Rhat_t = sum_T(gamma^(i-t) R(s_i))
        rtg = torch.zeros(EPOCH_STEPS)
        split_values = np.cumsum(lengths[-e_count:][::-1])
        for i, (_, _, r,_) in enumerate(history[::-1]):
            if i == 0 or i in split_values:
                rhat = 0
            rhat = r + GAMMA * rhat
            rtg[-i-1] = rhat

        # 5. Advantage estimates Ahat = Q^pi(s, a) - V^pi(s)
        advantages = rtg - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        # 6. Update policy via maximize ppo-clip objective
        # https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl
        # argmax 1 / NT sum_N sum_T min(pi_theta / pi_theta_k * A^pi_theta_k, g(epsilon, A^pi_theta_k))
        # -> w += gradient. Invert loss output
        # PPO Clip adds clippsed surrogate objective as replacement for
        # policy gradient objective. This improves stability by limiting
        # change you make to policy at each step.
        # Vanilla pg uses log p of action to trace impact of actions,
        # ppo clip uses prob of action under current policy / prob under prev policy
        # r_t(theta) = pi_theta(a_t, s_t) / pi_theta_old(a_t | s_t)
        # Thus L = E[r * advantages]
        # r > 1 if action more probable with current policy, else 1 > r > 0.
        # to prevent too large steps from oto large r, uses clipped surrogate objective
        # ie L_clip = E[min(r * advantages, clip(r, 1 - epsilon, 1 + epsilon) * advantages)]
        # epsilon = 2.
        p, p_old = torch.zeros(EPOCH_STEPS), torch.zeros(EPOCH_STEPS)
        for i, (s, a, _, _) in enumerate(history):
            a_idx = action_space.index(a)
            p[i] = policy_probs[i][a_idx]
            p_old[i] = policy_old(s)[a_idx]
        r_theta = p / p_old.detach()
        r_clip = torch.clamp(r_theta, 1 - EPSILON_CLIP, 1 + EPSILON_CLIP)
        loss_policy = -torch.min(r_theta * advantages, r_clip * advantages).mean()

        loss_value = criterion_value(values, rtg)

        loss_policy.backward(retain_graph=True)
        policy_old = deepcopy(policy)
        opt_policy.step()

        # 7. Fit value fn via regression on MSE
        loss_value.backward()
        opt_value.step()

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
