from mss import mss
import pydirectinput
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import os
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import DQN

class WebGame(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255,shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        #for extracting
        self.cap = mss()
        self.game_location = {'top': 300, 'left': 0, 'width': 600, 'height': 500}
        self.done_location = {'top': 450, 'left': 630, 'width': 660, 'height': 70}

        
        
    def step(self, action):
        action_map = {
            0:'space',
            1: 'down', 
            2: 'no_op'
        }
        if action !=2:
            pydirectinput.press(action_map[action])

        done, done_cap = self.get_done() 
        new_observation = self.get_observation()
        reward = 1  #get a point for every frame alive 
        info = {}
        truncated = False
        return new_observation, reward, done, truncated, info        
    
    def reset(self,seed=None):
        time.sleep(1)
        pydirectinput.click(x=300, y=300)
        pydirectinput.press('space')
        info = {}
        return self.get_observation(), info
        
    def render(self):
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:,:,:3])
        #giving some time to render and if we hit q we close the frame        
        if cv2.waitKey(5000) & 0xFF == ord('q'):
            self.close()
        

    def close(self):
        cv2.destroyAllWindows()
    
    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3].astype(np.uint8)   #only grabbing a part of this array cuz alpha channel unnecessary (the first 3 channels are RGB values fouth is alpha channel)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) #converting 3 channels to one channel (grey scale)       
        resized = cv2.resize(gray, (100,83)) #resize         
        channel = np.reshape(resized, (1,83,100))#add channels first
        return channel
    
    def get_done(self):
        done_cap = np.array(self.cap.grab(self.done_location))[:,:,:3]
        done_strings = ['GAME', 'GAHE']  #cuz sometimes it confuse or u could preprocess
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_strings:
            done = True
        return done, done_cap

    
    
env = WebGame()

# to see if observation is what we wanted 
# plt.imshow(env.get_observation()[0])   
# plt.show()


#check that the env is ok
#env_checker.check_env(env)

# class TrainAndLoggingCallback(BaseCallback):

#     def __init__(self, check_freq, save_path, verbose=1):
#         super(TrainAndLoggingCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.save_path = save_path

#     def _init_callback(self):
#         if self.save_path is not None:
#             os.makedirs(self.save_path, exist_ok=True)

#     def _on_step(self):
#         if self.n_calls % self.check_freq == 0:
#             model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
#             self.model.save(model_path)

#         return True
 

# CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'   #run this too to test trained model
# callback = TrainAndLoggingCallback(check_freq=300, save_path = CHECKPOINT_DIR)
model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=100000, learning_starts=1000)  #run this too to test trained model
# model.learn(total_timesteps=30000, callback=callback)

# Run this to see test a trained model
model.load('train/best_model_30000.zip') 
for episode in range(5):
    obs = env.reset()[0]
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done,truncated, info = env.step(int(action))
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(episode, total_reward))
  