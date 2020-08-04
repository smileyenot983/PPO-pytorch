# pylint:disable=missing-module-docstring
from gym.envs.registration import register

register(
    id="Pendubot-v0",
    entry_point="pendubot.envs.pendubot:Pendubot",
    max_episode_steps=1000,
)


