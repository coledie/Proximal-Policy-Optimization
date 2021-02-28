"""
Base actor-critic model.
"""
import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, actor, critic, **kwargs):
        super().__init__()

        self.actor = actor
        self.critic = critic
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.epoch = 0
        self.lengths, self.eps = [], []

    def play(self, env):
        """
        Compute actor trajectories by running current policy.
        """
        n_steps, e_count = 0, 0
        self.history, self.values, self.policy_probs = [], torch.zeros(self.epoch_steps), torch.zeros((self.epoch_steps, len(self.action_space)))
        while n_steps < self.epoch_steps:
            e_count += 1
            state = env.reset()
            for l in range(self.episode_len):
                policy_choice = self.actor(state) + 1e-10
                action = np.random.choice(self.action_space, p=policy_choice.detach().numpy() / float(policy_choice.sum()))
                state_next, reward, done, info = env.step(action)

                self.history.append((state, action, reward, state_next))
                self.values[n_steps] = self.critic(state)
                self.policy_probs[n_steps] = policy_choice
                state = state_next

                n_steps += 1
                if done or n_steps == self.epoch_steps:
                    break
            self.lengths.append(l+1)
            if np.mean(self.lengths[-self.n_to_win:]) >= self.win_score:
                print("Win!")
        self.eps.append(e_count)
        print(f"{self.epoch}: {np.mean(self.lengths[-e_count:]):.0f}")
        self.epoch += 1

    def rtg(self):
        """
        Calculate rewards to go,
        Rhat_t = sum_T(gamma^(i-t) R(s_i))
        """
        rtg = torch.zeros(self.epoch_steps)
        split_values = np.cumsum(self.lengths[-self.eps[-1]:][::-1])
        for i, (_, _, r,_) in enumerate(self.history[::-1]):
            if i == 0 or i in split_values:
                rhat = 0
            rhat = r + self.gamma * rhat
            rtg[-i-1] = rhat
        return rtg

    def advantages(self, rtg=None):
        """
        Calculate general advantage estimates,
        Ahat = Q^pi(s, a) - V^pi(s)
        """
        rtg = self.rtg() if rtg is None else rtg
        advantages = rtg - self.values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        return advantages

    def step(self):
        """
        Update policy and value networks.
        """
        rtg = self.rtg()
        advantages = self.advantages(rtg)

        self.actor.step(self.history, self.policy_probs, advantages)
        self.critic.step(self.values, rtg)
