"""
Trust Region Policy Optimization
- on-policy
- model based
- continuous

Impose constraint on KL divergence between new and old policy
(the trust region constraint):
maximize_theta L_theta_old(theta)
subject to D_KL^max(theta_old, theta) <= delta

Since this is impractical to solve with so many constraints,
uses average KL div instead,
Dbar_KL(theta_1, theta_2) = E_s~p[D_KL(pi_theta_old(.|s) || pi_theta_2(.|s))]

Thus policy update is,
max_theta(L_theta_old(theta))
subject to Dbar_KL(theta_old, theta) <= delta

https://spinningup.openai.com/en/latest/algorithms/trpo.html
https://arxiv.org/pdf/1502.05477.pdf 
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
    TRPO Learning the cartpole balancing problem.

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
    GAMMA = .01

    ##
    env = gym.make('CartPole-v0')
    N_INPUT = 4
    LATENT_SIZE = 10
    N_ACTION = env.action_space.n
    action_space = [i for i in range(env.action_space.n)]

    policy = NN([N_INPUT, LATENT_SIZE, N_ACTION], [nn.Tanh(), nn.Softmax(dim=-1)])
    policy_old = deepcopy(policy)
    value = NN([N_INPUT, LATENT_SIZE, 1], [nn.Tanh(), lambda x: x])

    opt_policy = optim.Adam(policy.parameters(), lr=ALPHA)
    criterion_value = nn.MSELoss()
    opt_value = optim.Adam(value.parameters(), lr=ALPHA)

    lengths, eps = [], []
    for e in range(N_EPOCH):
        opt_policy.zero_grad()
        opt_value.zero_grad()

        # 4. Compute trajectories D by running current policy
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

        # 5. Rewards to go, Rhat_t = sum_T(gamma^(i-t) R(s_i))
        rtg = torch.zeros(EPOCH_STEPS)
        split_values = np.cumsum(lengths[-e_count:][::-1])
        for i, (_, _, r,_) in enumerate(history[::-1]):
            if i == 0 or i in split_values:
                rhat = 0
            rhat = r + GAMMA * rhat
            rtg[-i-1] = rhat

        # 6. Advantage estimates Ahat = Q^pi(s, a) - V^pi(s)
        advantages = rtg - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        # 7. Estimate policy gradient as,
        # ghat_k = 1/N_traj sum_T in trajs sum_t in T(gradient log(pi_theta(a_t|s_t)) * Ahat_t)

        # 8. Use conjugate gradient algorithm to compute,
        # xhat_k = Hhat^-1_k * ghat_k

        # 9. Update policy by backtracking line search with,
        # theta_k+1 = theta_k + alpha * \
        #       sqrt(xhat_k * 2gamma / (xhat_k.T * Hhat_k * x_hat_k))
        # where j in {0 .. K} is smallest value that improves sample loss
        # and satisfies KL-divergence constraint

        # 10. Fit value fn by regression on MSE,
        # phi_k+1 = argmin_phi 1/N_traj / T * \
        #       sum_t in traj sum t in T(V_phi(s_t) - Rhat_t)^2
        # via sgd
        

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
