import torch
import torch.optim
from torch.optim import Optimizer
import math

def clone_param_dict(d):
    return {param_name: param_tensor.clone() for (param_name, param_tensor) in d.items()}

def copy_param_dict(from_, to):
    for (param_name, param_tensor) in to.items():
        param_tensor.copy_(from_[param_name])

class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.best_position = clone_param_dict(self.position)
        self.best_value = None
        self.velocity = velocity
        self.random_best_pos_multipliers = clone_param_dict(self.position)
        self.random_best_swarm_pos_multipliers = clone_param_dict(self.position)

    def randomize_random_multipliers(self):
        def randomize_dict(d):
            for param_tensor in d.values():
                param_tensor.uniform_()
        randomize_dict(self.random_best_pos_multipliers)
        randomize_dict(self.random_best_swarm_pos_multipliers)

class ParticleSwarmOptimizer(Optimizer):
    # state_dict is a dictionary obtained by running the state_dict() method on a module, ranges is a
    # dictionary with the same keys (parameter names) as the state dict, but the values are a 2-tuple of
    # (low, high) values from which the particles in the swarm will be initialized
    # evaluator is a function that takes no parameters and returns the numerical result of running the model forward.
    # Most often this will simply be module.forward, but alternatives are possible
    # Gradient optimizer is a PyTorch optimizer used for the gradient step of the swarm optimizer
    def __init__(self, num_particles, state_dict, ranges, evaluator):
        self.state_dict = state_dict
        self.ranges = ranges
        self.evaluator = evaluator

        self.particles = []
        self.best_swarm_position = None
        self.best_swarm_value = None

        assert(num_particles > 0)

        for i in range(num_particles):
            position = clone_param_dict(self.state_dict)
            for param_name in position.keys():
                position[param_name].uniform_()
                (low, high) = ranges[param_name]
                position[param_name] = low + (high - low) * position[param_name]

            velocity = clone_param_dict(self.state_dict)
            for param_name in velocity.keys():
                velocity[param_name].uniform_()
                (low, high) = ranges[param_name]
                diff = torch.abs(high - low)
                low_vel = -diff
                high_vel = diff
                velocity[param_name] = low_vel + (high_vel - low_vel) * velocity[param_name]

            self.particles.append(Particle(position, velocity))

        for particle in self.particles:
            self.blit_particle(particle)
            val = float(evaluator())
            particle.best_value = val
            if self.best_swarm_value is None or val < self.best_swarm_value:
                self.best_swarm_value = val
                if self.best_swarm_position is None:
                    self.best_swarm_position = clone_param_dict(particle.position)
                else:
                    copy_param_dict(particle.position, self.best_swarm_position)

        # These constants come from the paper "An off-the-shelf PSO" by Anthony Carlisle and Gerry Dozier
        self.phi1 = 2.8
        self.phi2 = 1.3
        self.phi = self.phi1 + self.phi2
        self.k = 2.0 / abs(2.0 - self.phi - math.sqrt(self.phi * self.phi - 4.0 * self.phi))

    def blit_particle(self, particle):
        copy_param_dict(particle.position, self.state_dict)

    def blit_best_swarm_position(self):
        copy_param_dict(self.best_swarm_position, self.state_dict)

    def step(self):
        for particle in self.particles:
            particle.randomize_random_multipliers()

            for param_name in particle.position.keys():
                v_prev = particle.velocity[param_name]
                r1 = particle.random_best_pos_multipliers[param_name]
                r2 = particle.random_best_swarm_pos_multipliers[param_name]
                x = particle.position[param_name]
                particle.velocity[param_name] =\
                    self.k * (v_prev + self.phi1 * r1 * (particle.best_position[param_name] - x) +
                              self.phi2 * r2 * (self.best_swarm_position[param_name] - x))
                particle.position[param_name].add_(particle.velocity[param_name])

                (min_values, max_values) = self.ranges[param_name]

                particle.velocity[particle.position[param_name] < min_values] = 0.0
                particle.velocity[particle.position[param_name] > max_values] = 0.0

                particle.position[param_name] = torch.min(max_values, torch.max(min_values, particle.position[param_name]))

            self.blit_particle(particle)
            val = float(self.evaluator())

            if val < particle.best_value:
                copy_param_dict(particle.position, particle.best_position)
                particle.best_value = val

            if val < self.best_swarm_value:
                copy_param_dict(particle.position, self.best_swarm_position)
                self.best_swarm_value = val

# Gradient Particle swarm optimizer for PyTorch
# Based on the paper "A new gradient based particle swarm optimization algorithm for accurate computation of global minimum"
# by Matthew M. Noel