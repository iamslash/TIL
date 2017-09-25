# -*- coding: utf-8 -*-
import gym
from gym.envs.registration import register
import sys
import msvcrt

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    b'8' : UP,
    b'2' : DOWN,
    b'6' : RIGHT,
    b'4' : LEFT}

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)
env = gym.make('FrozenLake-v3')
env.render()

while True:
      key = msvcrt.getch()
      if key not in arrow_keys.keys():
          print("Game aborted")
          break
      action = arrow_keys[key]
      state, reward, done, info = env.step(action)
      env.render()
      print("State: ", state, "Action: ", action,
            "Reward: ", reward, "Info: ", info)

      if done:
          print("Finished with reward", reward)
          break    

def main():
    pass

if __name__ == "__main__":
    main()
