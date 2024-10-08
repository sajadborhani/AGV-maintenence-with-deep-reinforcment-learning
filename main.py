import logging

import numpy as np
from constants import Strategy, TimeFrame
from environment import Company, CustomEnv
# from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from config import config1,config2, config3, ITERATION
import pickle
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
config_case = {
    0 : config1,
    1 : config2,
    2 : config3,
}

class TrainRecurrentPPo:
    def __init__(self, strategy = Strategy.PREDICTIVE) -> None:
        self.strategy = strategy
        self.load_state()
        self.load_model()

    def load_state( self ):
        self.file_name = f'ppo_recurrent_config{ self.strategy + 1  }'
        try : 
            with open(f'{self.file_name}_state.pickle','rb') as file:
                state = pickle.load( file )
                self.iteration = state['iteration']
                self.config = state['config']
        except FileNotFoundError:
            self.iteration = 0
            self.config = config_case[self.strategy]
            
        self.env = Company( **self.config )

    def load_model(self):
        try:
            self.model = RecurrentPPO.load(f'{self.file_name}_case{self.config["location_case"]}_model.model')
        except FileNotFoundError:
            self.model = RecurrentPPO("MlpLstmPolicy", self.env, verbose=2)

    def train_model(self):
        self.model.learn( total_timesteps= 10**10 )
        self.model.save(f'{self.file_name}_case{self.config["location_case"]}_model.model') 
        self.iteration += 1
        with open(f'{self.file_name}_state.pickle', 'wb') as file:
            state = {
                    'config' : self.config,
                    'iteration' : self.iteration
                }
            pickle.dump(state, file)
    def start(self):
        check_env(self.env)
        
        while self.iteration < ITERATION:
            self.train_model()
        self.config['location_case'] -= 1
        if self.config['location_case'] >= 0:
            self.file_name = f'ppo_recurrent_config{ self.config["env_id"] }'
            self.start()
        else:
            print(f'training ended for {Strategy.NAME[ self.config["strategy"] ]} maintenence')


# class TensorboardCallback(BaseCallback):
#     """
#     Custom callback for plotting additional values in tensorboard.
#     """

#     def __init__(self, verbose=0):
#         super(TensorboardCallback, self).__init__(verbose)

#     def _on_step(self) -> bool:
#         # Log scalar value (here a random variable)
#         value = np.random.random()
#         self.logger.record("random_value", value)
#         return True


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] ='1'
    file_name = 'rppo_test.model'
    env = CustomEnv(config3)
    Monitor(env)
    print(env.company.env_id)
    # check_env(env)
    try:
        model = RecurrentPPO.load(file_name)
        model.set_env(env)
        print('model loaded')
    except FileNotFoundError:
        model = RecurrentPPO('MultiInputLstmPolicy', env, policy_kwargs={'n_lstm_layers':2}, verbose=1, tensorboard_log="./tensorboard/")
        print('model created')
    for i in range(ITERATION):
        model.learn( total_timesteps= 100000, tb_log_name="AGV_SIMULATION")
        model.save(file_name)