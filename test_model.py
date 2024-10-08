from sb3_contrib import RecurrentPPO
from constants import TimeFrame
from environment import CustomEnv
from config import config3, config2, config1
import numpy as np
import os
from csv import DictWriter

def save_summary(file_name, info):
    print(info)
    with open(file_name, 'a') as f_object:
        if isinstance(info, list):
            field_names = list(info[0].keys())
        else:
            field_names = list(info.keys())
            
        dictwriter_object = DictWriter(f_object, fieldnames=field_names)
    
        if isinstance(info, list):
            for data in info:
                dictwriter_object.writerow(data)
        else:
            dictwriter_object.writerow(info)

        f_object.close()
    
if __name__ == '__main__':
    
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
    
    os.environ['CUDA_VISIBLE_DEVICES'] ='1'
    config = config3
    model = RecurrentPPO.load("rppo_test.model")
    config['env_id'] = 'model_test'
    config["location_case"] = 1 
    config["simulation_time"] = 6*30*TimeFrame.DAY 
    env = CustomEnv(config)
    env = Monitor(env)
    # cell and hidden state of the LSTM
    num_envs = 1
    file_name = f'6th_test.csv'
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    for i in range(10):
        lstm_states = None
        obs = env.reset()
        total_return = 0
        dones = False
        while not dones:
            print(obs)

            action, lstm_states = model.predict(obs, state=lstm_states,  deterministic=True)
            state = next(reversed(obs.values()))
            print(state)
            # action = choos_action(state, pd=False , pm= False)
            obs, rewards, dones, info = env.step(action)
            episode_starts = dones
            total_return += rewards
        info = env.company.get_summary()
        save_summary(file_name, info)
    print(f'total return : {total_return}')