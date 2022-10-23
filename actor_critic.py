from collections import defaultdict
from typing import Optional, Union, Type, Dict

import torch

from system import DynamicSystem


class Scale(torch.nn.Module):
    def __init__(self, factor: float):
        super(self.__class__, self).__init__()
        self.factor = factor

    def forward(self, input):
        return input * self.factor/2 + self.factor/2

    def extra_repr(self):
        return f'factor={self.factor}'


class Actor(torch.nn.Module):
    def __init__(
            self,
            num_observations: int = 7,
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
        x = observation
        out = self.net(x)
        return out


class Critic(torch.nn.Module):
    def __init__(
            self,
            num_observations: int = 7,
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
        modules.append(torch.nn.Tanh())
        modules.append(Scale(scale))

        self.net = torch.nn.Sequential(*modules)

    def forward(self, observation):
        x = observation
        x = self.net(x)
        out = -x
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
            actor_subiters: int = 1,
            critic_subiters: int = 20,
            critic_batch_size: int = 1000,
            actor_batch_size: int = 4000,
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
            critic_losses = 0
            actor_losses = 0
            for i in range(critic_subiters):
                
                state = dynamics.system.sample_state(critic_batch_size)
                with torch.no_grad():
                    action = actor(dynamics.system.get_observation(state))
                    next_state = dynamics.make_transition(state, action)
                    
                critic_loss = dynamics.critic_loss(state, action, next_state)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                if grad_clip_norm is not None:
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip_norm)
                critic_optimizer.step()
                critic_losses += critic_loss.item() / critic_subiters
            for i in range(actor_subiters):
                state = dynamics.system.sample_state(actor_batch_size)
                
                action = actor(dynamics.system.get_observation(state))
                next_state = dynamics.make_transition(state, action)
                actor_loss = dynamics.actor_loss(state, action, next_state)

                actor_optimizer.zero_grad()
                actor_loss.backward()
                if grad_clip_norm is not None:
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip_norm)
                actor_optimizer.step()
                actor_losses += actor_loss.item() / actor_subiters
            if verbose:
                print(f'Iteration {iteration + 1:3d}/{self.total_iterations + iterations}: '
                      f'actor_loss={actor_losses:.4f}, '
                      f'critic_loss={critic_losses:.4f}, '
                      + (f', actor_grad_norm={actor_grad_norm:.4f}' if actor_grad_norm is not None else '')
                      + (f', critic_grad_norm={critic_grad_norm:.4f}' if critic_grad_norm is not None else ''))

            training_history['actor_loss'].append(actor_losses)
            training_history['critic_loss'].append(critic_losses)

        self.total_iterations += iterations

        return training_history

    def reset(self):
        self.total_iterations = 0
