from collections import defaultdict
from typing import Optional, Union, Type, Dict

import torch

from system import DynamicSystem
import numpy as np
from tqdm import tqdm


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
            num_observations: int = 8,
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
            modules.append(torch.nn.LeakyReLU() if i < depth - 1 else torch.nn.Identity())

        self.net = torch.nn.Sequential(*modules)

    def forward(self, observation):
        x = observation
        out = torch.nn.Tanh()(self.net(x))*0.5 + 0.5
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
        modules.append(torch.nn.Tanh())
        modules.append(Scale(scale))

        self.net = torch.nn.Sequential(*modules)

    def forward(self, observation):
        x = observation[:,:-2]
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
            actor_subiters: int = 200,
            critic_subiters: int = 1000,
            critic_batch_size: int = 1000,
            actor_batch_size: int = 1000,
            actor_window: int = 2,
            critic_window: int = 2,
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
            pbar = tqdm(range(critic_subiters))
            for i in pbar:

                state = dynamics.system.sample_state(critic_batch_size)
                with torch.no_grad():
                    next_state = state
                    reward = 0
                    for semi in range(critic_window):
                        action = actor(dynamics.system.get_observation(next_state))
                        reward += dynamics.system.get_reward(next_state,action) * dynamics.discount_factor**semi
                        
                        next_state = dynamics.make_transition(next_state, action)
                    target = reward[:,None] + (dynamics.discount_factor**(semi+1)) * critic(dynamics.get_observation(next_state))
                critic_loss = torch.nn.MSELoss()(critic(dynamics.get_observation(state)),target)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                if grad_clip_norm is not None:
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip_norm)
                critic_optimizer.step()
                if i % 10 == 0:
                    pbar.set_description(
                        f"Iteration {i}"
                        f"Loss {round(critic_loss.item())}"
                    )
            pbar = tqdm(range(actor_subiters))
            for i in pbar:
                state = dynamics.system.sample_state(actor_batch_size)
                
                next_state = state
                reward = 0
                for semi in range(actor_window):
                    action = actor(dynamics.system.get_observation(next_state.detach()))
                    next_state = dynamics.make_transition(next_state.detach(), action)
                    
                    reward += dynamics.get_reward(next_state,action) * dynamics.discount_factor ** semi
                    
                actor_loss = -(reward + dynamics.discount_factor ** (semi+1) * critic(dynamics.get_observation(next_state))).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                if grad_clip_norm is not None:
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip_norm)
                actor_optimizer.step()
                if i % 10 == 0:
                    pbar.set_description(
                        f"Iteration {i}"
                        f"Loss {round(actor_loss.item())}"
                    )

        self.total_iterations += iterations

        return training_history

    def reset(self):
        self.total_iterations = 0
