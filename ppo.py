"""
Proximal policy optimization. PPO-clip version.
- on-policy
- continuous

https://spinningup.openai.com/en/latest/algorithms/ppo.html
https://medium.com/swlh/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a
https://towardsdatascience.com/proximal-policy-optimization-ppo-with-sonic-the-hedgehog-2-and-3-c9c21dbed5e
"""
from copy import deepcopy
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt


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

    N_EPOCH = 100
    EPOCH_STEPS = 4800
    EPISODE_LEN = 400

    ALPHA = .005
    GAMMA = .95
    EPSILON_CLIP = .2

    env = gym.make('CartPole-v0')
    action_space = [i for i in range(env.action_space.n)]

    N_INPUT = 4
    N_ACTION = len(action_space)

    policy = NN([N_INPUT, N_ACTION], [F.relu])
    policy_old = deepcopy(policy)
    value = NN([N_INPUT, 1], [torch.tanh, lambda x: x])

    opt_policy = optim.Adam(policy.parameters(), lr=ALPHA)
    criterion_value = nn.MSELoss()
    opt_value = optim.Adam(value.parameters(), lr=ALPHA)

    lengths = []
    for e in range(N_EPOCH):
        opt_policy.zero_grad()
        opt_value.zero_grad()

        # 3. Compute trajectories D by running current policy
        n_steps = 0
        histories, values, policy_probs = [], [], []
        while n_steps < EPOCH_STEPS:
            h, v, p = [], torch.empty(EPISODE_LEN), torch.empty((EPISODE_LEN, len(action_space)))
            observation = env.reset()
            state = normalize(observation)
            for l in range(EPISODE_LEN):
                n_steps += 1
                policy_choice = policy(state) + .000000000001
                action = np.random.choice(action_space, p=policy_choice.detach().numpy() / float(policy_choice.sum()))
                observation, reward, done, info = env.step(action)
                state_next = normalize(observation)
                h.append((state, action, reward, state_next))
                state = state_next
                v[l] = value(state_next)
                p[l] = policy_choice

                if done:
                    break

            lengths.append(len(h))
            v = torch.narrow(v, 0, 0, lengths[-1])
            p = torch.narrow(p, 0, 0, lengths[-1])

            histories.append(h)
            values.append(v)
            policy_probs.append(p)

        print(f"{e}: {lengths[-1]}")

        # 4. Rewards to go, Rhat_t = sum_T(gamma^(i-t) R(s_i))
        rtg = []
        for e, h in enumerate(histories[::-1]):
            rhat = 0
            rtg_ = torch.empty(lengths[-e-1])
            for i, (_, _, r, _) in enumerate(h[::-1]):
                rhat = r + GAMMA * rhat
                rtg_[-i-1] = rhat
            rtg.append(rtg_)
        rtg = rtg[::-1]

        # 5. Advantage estimates Ahat = Q^pi(s, a) - V^pi(s)
        advantages = []
        for i in range(len(histories)):
            h = histories[i]
            r = rtg[i]
            a = r - torch.Tensor([value(s) for s, _, _, _ in h]).detach()
            advantages.append(a)

        a_grouped = []
        for a in advantages:
            a_grouped.extend(a.detach())
        a_mean = np.mean(a_grouped)
        a_std = np.std(a_grouped)

        advantages = [(a - a_mean) / (a_std + 1e-10) for a in advantages]

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
        for l, h in enumerate(histories):
            length = lengths[-len(histories)+l]
            p, p_old = torch.empty(length, requires_grad=True), torch.empty(length, requires_grad=False)
            for i, (s, a, _, _) in enumerate(h):
                with torch.no_grad():
                    a_idx = action_space.index(a)
                    p[i] = policy_probs[l][i][a_idx]
                    p_old[i] = policy_old(s)[a_idx]
            r_theta = p / p_old
            r_clip = torch.clamp(r_theta, 1 - EPSILON_CLIP, 1 + EPSILON_CLIP)
            loss_policy = torch.min(r_theta * advantages[l], r_clip * advantages[l]).mean()

            loss_value = criterion_value(values[l], rtg[l])

            loss_policy.backward(retain_graph=True)
            policy_old = deepcopy(policy)

            # 7. Fit value fn via regression on MSE
            loss_value.backward()

        opt_policy.step()
        opt_value.step()

    V = torch.Tensor([value(s) for s, _, _, _ in histories[-1]]).detach()
    X = np.arange(len(V))
    plt.plot(X, V, label="v")
    plt.plot(X, rtg[-1] / rtg[-1].max(), label="rtg_norm")
    plt.plot(X, advantages[-1], label="A")
    plt.legend()
    plt.show()
