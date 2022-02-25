import gym
import numpy as np
import pygame, sys, time, random
from stable_baselines3 import PPO

from gym import spaces

class SnakeEnv1(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}
  # any constant will be set here
  difficulty = 20
  # Window size
  frame_size_x = 470
  frame_size_y = 300
  # Colors (R, G, B)
  black = pygame.Color(0, 0, 0)
  white = pygame.Color(255, 255, 255)
  red = pygame.Color(255, 0, 0)
  green = pygame.Color(0, 255, 0)
  blue = pygame.Color(0, 0, 255)

  # It is better to change Actions into Constants
  LEFT = 0
  RIGHT = 1
  UP = 2
  DOWN = 3

  def __init__(self): # we don't need to pass any arguments through constructor
    super(SnakeEnv1, self).__init__()
    pygame.init() # for any intialization should be inside constructor
    # Initialise game window
    pygame.display.set_caption('Snake Eater')
    self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
    # FPS (frames per second) controller
    self.fps_controller = pygame.time.Clock()

    # Game variables
    self.snake_pos = [100, 50]
    self.prev_snake_pos = [100, 50] # to calculate the distance bet current and prv to reward the action. (We should put it in reset as well)
    self.snake_body = [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]]

    self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10, random.randrange(1, (self.frame_size_y // 10)) * 10]
    self.food_spawn = True

    self.direction = 'RIGHT'
    self.change_to = self.direction
    self.score = 0
    self.counter = 0
    self.game_over = False # this is Game over function, we set zero because we don't need finish while Game over but we need to reset to intial status in case of Game over during training model
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    # number of actions == 4 so (from 0 to 3 inclusive), it is better to change numbers to constants
    no_of_actions = 4  # in case of snake there are 4 steps and actions (Discrete) (up,down,right,left)
    no_of_observations= 4
    self.action_space = spaces.Discrete(no_of_actions)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(no_of_observations,), dtype=np.float32)# we will just pass number of observations

  def step(self, action):
      self.counter += 1 # to avoid rotating around itself
      if self.counter > 100:
          return np.array([self.snake_pos[0], self.snake_pos[1], self.food_pos[0], self.food_pos[1]], dtype=np.float32), -100, True, {}
      if action == self.UP:
          self.change_to = 'UP'
      if action == self.DOWN:
          self.change_to = 'DOWN'
      if action == self.LEFT:
          self.change_to = 'LEFT'
      if action == self.RIGHT:
          self.change_to = 'RIGHT'

      # Making sure the snake cannot move in the opposite direction instantaneously ( it means we can't return 180 degree)
      if self.change_to == 'UP' and self.direction != 'DOWN':
          self.direction = 'UP'
      if self.change_to == 'DOWN' and self.direction != 'UP':
          self.direction = 'DOWN'
      if self.change_to == 'LEFT' and self.direction != 'RIGHT':
          self.direction = 'LEFT'
      if self.change_to == 'RIGHT' and self.direction != 'LEFT':
          self.direction = 'RIGHT'

      # Moving the snake
      if self.direction == 'UP':
          self.snake_pos[1] -= 10
      if self.direction == 'DOWN':
          self.snake_pos[1] += 10
      if self.direction == 'LEFT':
          self.snake_pos[0] -= 10
      if self.direction == 'RIGHT':
          self.snake_pos[0] += 10
      # Snake body growing mechanism (any time the snake moves, the body size increases whether eat or not)
      #but the body decreaes later (by pop() function ) if the snake doesn't eat food (so the size keeps the same)
      self.snake_body.insert(0, list(self.snake_pos))
      if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
          self.counter = 0 # reset counter that is used to avoid rotating itself
          self.score += 1   # the score increase when eating food
          self.food_spawn = False   # food will disappear after eating and will be shown in different position as below
      else:
          self.snake_body.pop()# Python list pop() is an inbuilt function in Python that removes and returns the last value from the List or the given index value

      # Spawning food on the screen (put food in another location on screen)
      if not self.food_spawn:
          self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10, random.randrange(1, (self.frame_size_y // 10)) * 10]
      self.food_spawn = True
      # Game Over conditions
      # Getting out of bounds
      if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x - 10:
          self.game_over = True
      if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y - 10:
          self.game_over = True
      # Touching the snake body
      for block in self.snake_body[1:]:
          if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
              self.game_over = True
      reward = 0
      if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
          reward = 100
      elif abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1]) > abs(self.prev_snake_pos[0] - self.food_pos[0]) + abs(self.prev_snake_pos[1] - self.food_pos[1]):
          reward = -1
      elif abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1]) < abs(self.prev_snake_pos[0] - self.food_pos[0]) + abs(self.prev_snake_pos[1] - self.food_pos[1]):
          reward = 1
      self.prev_snake_pos = self.snake_pos.copy()
      done = self.game_over
      info = {}
      return np.array([self.snake_pos[0], self.snake_pos[1], self.food_pos[0], self.food_pos[1]], dtype=np.float32), reward, done, info

  def reset(self):
      # we need the variable that are in init (constructor) which are changed every time we reset the Game
      # Game variables
      # Game variables
      self.snake_pos = [100, 50]
      self.prev_snake_pos = [100, 50]
      self.snake_body = [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]]
      self.counter = 0
      self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10,
                       random.randrange(1, (self.frame_size_y // 10)) * 10]
      self.food_spawn = True

      self.direction = 'RIGHT'
      self.change_to = self.direction

      self.score = 0
      self.game_over = False
      # we need to return snake and food positions in arrary numpy that has type float32 as we defined at oberservation space
      return np.array([self.snake_pos[0], self.snake_pos[1], self.food_pos[0], self.food_pos[1]], dtype=np.float32)  # reward, done, info can't be included

  def render(self, mode='human'):
      # GFX
      self.game_window.fill(self.black)
      for pos in self.snake_body:
          # Snake body
          # .draw.rect(play_surface, color, xy-coordinate)
          # xy-coordinate -> .Rect(x, y, size_x, size_y)
          pygame.draw.rect(self.game_window, self.green, pygame.Rect(pos[0], pos[1], 10, 10))

      # Snake food
      pygame.draw.rect(self.game_window, self.white, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))
      self.show_score(1, self.white, 'consolas', 20)
      # Refresh game screen
      pygame.display.update()
      # Refresh rate
      self.fps_controller.tick(self.difficulty)

  def close(self):
      pygame.quit()
      sys.exit()

  def show_score(self, choice, color, font, size):
      score_font = pygame.font.SysFont(font, size)
      score_surface = score_font.render('Score : ' + str(self.score), True, color)
      score_rect = score_surface.get_rect()
      if choice == 1:
          score_rect.midtop = (self.frame_size_x / 10, 15)
      else:
          score_rect.midtop = (self.frame_size_x / 2, self.frame_size_y / 1.25)
      self.game_window.blit(score_surface, score_rect)
      # pygame.display.flip()


# env = gym.make("CartPole-v1")
env = SnakeEnv1()

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
#model.save("snake_ai_model")
# model = PPO.load("snake_ai_model")
obs = env.reset()

for i in range(200):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    print(obs, reward, done)
    if done:
      obs = env.reset()




# env.close()