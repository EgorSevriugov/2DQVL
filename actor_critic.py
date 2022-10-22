from collections import defaultdict
from typing import Optional, Union, Type, Dict

import torch

from system import DynamicSystem


class Scale(torch.nn.Module):
    def __init__(self, factor: float):
        super(self.__class__, self).__init__()
        self.factor = factor

    def forward(self, input):
        return input * self.factor

    def extra_repr(self):
        return f'factor={self.factor}'


class Actor(torch.nn.Module):
    def __init__(
            self,
            num_observations: int = 6,
            num_actions: int = 2,
            dim: int = 32,
            depth: int = 3
    ):
        super().__init__()

        modules = []
        for i in range(depth):
            modules.append(torch.nn.Linear(
                num_observations if i == 0 else dim,
                num_actions if i == depth - 1 else dim
            ))
            modules.append(torch.nn.LeakyReLU() if i < depth - 1 else torch.nn.Sigmoid())

        self.net = torch.nn.Sequential(*modules)

    def forward(self, observation):
        x = observation.unsqueeze(0) if observation.ndim == 1 else observation
        out = self.net(x).squeeze(0)
        return out


class Critic(torch.nn.Module):
    def __init__(
            self,
            num_observations: int = 6,
            dim: int = 32,
            depth: int = 3,
            scale: float = 1.
    ):
        super().__init__()

        modules = []
        for i in range(depth):
            modules.append(torch.nn.Linear(
                num_observations if i == 0 else dim,
                1 if i == depth - 1 else dim
            ))
            if i < depth - 1:
                modules.append(torch.nn.LeakyReLU())
        modules.append(Scale(scale))

        self.net = torch.nn.Sequential(*modules)

    def forward(self, observation):
        x = observation.unsqueeze(0) if observation.ndim == 1 else observation
        x = self.net(x).squeeze(0)
        out = -(x ** 2)
        return out


class ActorCriticTrainer:
    def __init__(self, dynamics: DynamicSystem):
        self.dynamics = dynamics
        self.actor = dynamics.actor
        self.critic = dynamics.critic
        self.actor_optimizer = None
        self.critic_optimizer = None

        self.total_iterations = 0

    def init_optimizers(
            self,
            actor_optimizer: Union[torch.optim.Optimizer, Type] = torch.optim.Adam,
            actor_optimizer_params: Optional[Dict] = None,
            critic_optimizer: Union[torch.optim.Optimizer, Type] = torch.optim.Adam,
            critic_optimizer_params: Optional[Dict] = None,
    ):
        if isinstance(actor_optimizer, torch.optim.Optimizer):
            self.actor_optimizer = actor_optimizer
        else:
            actor_optimizer_params = actor_optimizer_params or {}
            self.actor_optimizer = actor_optimizer(self.actor.parameters(), **actor_optimizer_params)

        if isinstance(critic_optimizer, torch.optim.Optimizer):
            self.critic_optimizer = critic_optimizer
        else:
            critic_optimizer_params = critic_optimizer_params or {}
            self.critic_optimizer = critic_optimizer(self.critic.parameters(), **critic_optimizer_params)

    def run(
            self,
            iterations: int = 10,
            episodes: int = 1,
            grad_clip_norm: Optional[float] = None,
            verbose: bool = False
    ):
        dynamics = self.dynamics
        actor, critic = dynamics.actor, dynamics.critic

        actor_optimizer, critic_optimizer = self.actor_optimizer, self.critic_optimizer
        assert actor_optimizer is not None and critic_optimizer is not None, \
            'Actor/Critic optimizers are not initialized, run `init_optimizers`'
        actor_grad_norm = critic_grad_norm = None

        training_history = defaultdict(list)

        for iteration in range(self.total_iterations, self.total_iterations + iterations):
            actor_losses, critic_losses, rewards = [], [], []
            for episode in range(episodes):
                episode_actor_loss, episode_critic_loss, episode_reward = 0., 0., 0.

                t = 0
                state = dynamics.system.init_state()
                while t <= dynamics.total_time:
                    action = actor(dynamics.system.get_observation(state))
                    next_state = dynamics.make_transition(state, action)

                    episode_actor_loss += dynamics.actor_loss(state, action=action, next_state=next_state)
                    episode_critic_loss += dynamics.critic_loss(state, action=action, next_state=next_state)
                    episode_reward += dynamics.get_reward(state, action).item()

                    state = next_state.detach()
                    t += dynamics.sampling_time

                steps = dynamics.total_time / dynamics.sampling_time
                episode_actor_loss /= steps
                episode_critic_loss /= steps

                if episodes > 1 and verbose:
                    print(f'Episode {episode + 1:3d}/{episodes}: '
                          f'actor_loss={episode_actor_loss.item():.4f}, '
                          f'critic_loss={episode_critic_loss.item():.4f}, '
                          f'reward={episode_reward:.4f}')

                actor_losses.append(episode_actor_loss)
                critic_losses.append(episode_critic_loss)
                rewards.append(episode_reward)

            actor_loss = torch.stack(actor_losses).mean()
            critic_loss = torch.stack(critic_losses).mean()
            reward = sum(rewards) / len(rewards)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            if grad_clip_norm is not None:
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip_norm)
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            if grad_clip_norm is not None:
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip_norm)
            critic_optimizer.step()

            if verbose:
                print(f'Iteration {iteration + 1:3d}/{self.total_iterations + iterations}: '
                      f'actor_loss={actor_loss.item():.4f}, '
                      f'critic_loss={critic_loss.item():.4f}, '
                      f'reward={reward:.4f}'
                      + (f', actor_grad_norm={actor_grad_norm:.4f}' if actor_grad_norm is not None else '')
                      + (f', critic_grad_norm={critic_grad_norm:.4f}' if critic_grad_norm is not None else ''))
                if episodes > 1:
                    print()

            training_history['actor_loss'].append(actor_loss.item())
            training_history['critic_loss'].append(critic_loss.item())
            training_history['reward'].append(reward)

        self.total_iterations += iterations

        return training_history

    def reset(self):
        self.total_iterations = 0
