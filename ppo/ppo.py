"""
Proximal policy optimization with a clipped objective function.
- on-policy
- model based
- continuous

Proximal policy optimization takes an agent's behavior policy
represented in a deep neural network and tunes it as a whole
via gradient descent. This is a model(neural network) based,
on-policy reinforcement learning approach to learning to play games.

In this, two separate neural networks are used, a policy network that
determines how the agent will react to the environment and a value
network that approximates the long term value of a given state assuming
the current policy is used.
The value network simply learns to approximate long term, potential
rewards that come after the state by using the current policy.
This is learned via gradient descent on the mean squared error
between its output and the calculated rewards-to-go. Based on the
approximated state values, the policy network's gradients are calculated
with respect to the advantage function between the state's predicted value
and the earned rewards-to-go. The advantage will be positive if the actions
taken in the current episode earned a higher total reward than expected,
otherwise negative.
Therefore if there is a positive change in the networks policy,
its behavior will be reinforced and further solidified.
Though these advantages can be pretty volatile and hard to learn off of.
Thus in standard PPO this factor is modulated by a policy change ratio
that determines how different the current policy is to the old. This
limits the speed of policy changes. In PPO-Clip(implemented here) the
advantages are further stabilized by normalizing and setting bounds
to them, preventing any sudden jumps in network parameters.

On-policy methods(PPO) have pros over off-policy methods(Q-Learning).
They consider the decisions made as a whole instead of separate unrelated
steps. This is crucial in scenarios where it is necessary to take actions
that are less optimal in the short run but will lead to long term gains.
Model based approaches using neural networks also have the benefit that
they can handle continuous state and action spaces. On top of this the
neural models are more capable of understanding the environment they
interact with. This allows them to carry knowledge on how certain actions
will play out over time, giving them a bit of memory and saving much time
unnecessarily retrying the same patterns.

https://spinningup.openai.com/en/latest/algorithms/ppo.html
https://medium.com/swlh/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a
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

        # 3. Compute trajectories D by running current policy
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
        # PPO Clip adds clipped surrogate objective as replacement for
        # policy gradient objective. This improves stability by limiting
        # change you make to policy at each step.
        # Vanilla pg uses log p of action to trace impact of actions,
        # ppo clip uses prob of action under current policy / prob under prev policy
        # r_t(theta) = pi_theta(a_t, s_t) / pi_theta_old(a_t | s_t)
        # Thus L = E[r * advantages]
        # r > 1 if action more probable with current policy, else 1 > r > 0.
        # to prevent too large steps from too large r, uses clipped surrogate objective
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
