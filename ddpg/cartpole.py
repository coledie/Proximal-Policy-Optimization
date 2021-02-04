"""
Pulled from: https://github.com/SpikeyCNS/spikey

Cart pole balancing game.
Florian. "Correct equations for the dynamics of the cart-pole system."
Center for Cognitive and Neural Studies(Coneural), 10 Feb 2007,
https://coneural.org/florian/papers/05_cart_pole.pdf
"""
import numpy as np


class CartPole:
    """
    Inverted pendulum / pole-cart / cart-pole reinforcement learning
         g=9.8      /
          |        / pole: Length = 1 m
          |       /
          V      /
                / θ (angle), theta_dot is angular velocity
         ______/_____
        |            | Cart: M = 1 kg
        |____________| ----> x_dot is velocity
          O        O
    L1--------x-------------------L2 x is poxition, with x limits of L1, L2)
    Actions: jerk left, jerk right (AKA bang-bang control)
    Goal: control x position of cart to keep pole close to upright,
    which is when θ = pi/2 (vertical).
    Florian. "Correct equations for the dynamics of the cart-pole system."
    Center for Cognitive and Neural Studies(Coneural), 10 Feb 2007,
    https://coneural.org/florian/papers/05_cart_pole.pdf
    """

    action_space = np.arange(-1, 1, 0.1)
    observation_space = None  # Defined in init

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        pass

    def step(self, action: float) -> (np.ndarray, 0, bool, {}):
        """
        Act within the environment.

        Parameters
        ----------
        action: float
            Directed force with negative being left.
        """
        PoleMass_Length = .1 * .5
        Total_Mass = 1. + .1
        Fourthirds = 4.0 / 3.0

        #
        force = action

        assert force < 10.0 * 1.2, "Action force too high."

        x, x_dot, theta, theta_dot = self._state

        temp = (
            force + PoleMass_Length * theta_dot * theta_dot * np.sin(theta)
        ) / Total_Mass

        thetaacc = (9.8 * np.sin(theta) - np.cos(theta) * temp) / (
            .5
            * (
                Fourthirds
                - .1 * np.cos(theta) * np.cos(theta) / Total_Mass
            )
        )

        xacc = temp - PoleMass_Length * thetaacc * np.cos(theta) / Total_Mass

        # Update the four state variables, using Euler's method:
        # https://en.wikipedia.org/wiki/Euler_method
        x = x + 0.02 * x_dot
        x_dot = x_dot + 0.02 * xacc
        theta = theta + 0.02 * theta_dot
        theta_dot = theta_dot + 0.02 * thetaacc

        state_new = np.array([x, x_dot, theta, theta_dot])

        ##
        x, x_dot, theta, theta_dot = state_new

        done = abs(x) > 2.5 or abs(theta) > 0.5 * np.pi

        rwd = int(not done)
        info = {}

        self._state = state_new
        return state_new, rwd, done, info

    def reset(self) -> np.ndarray:
        """
        Reset environment.
        """
        x = np.random.uniform(*[0.0, 0.0]) * np.random.choice([-1, 1])
        x_dot = np.random.uniform(*[-0.1, 0.1]) * np.random.choice([-1, 1])
        theta = np.random.uniform(*[0.0, 0.0]) * np.random.choice([-1, 1])
        theta_dot = np.random.uniform(*[-0.1, 0.1]) * np.random.choice([-1, 1])

        s = np.array([x, x_dot, theta, theta_dot])

        self._state = s
        return s
