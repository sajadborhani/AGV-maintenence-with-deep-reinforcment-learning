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


def choos_action(state,pd=False, pm=False):
    action = []
    if state[2] == 0:
        action.append(1)
    else:
        action.append(0)
    if state[4] == 0:
        action.append(1)
    else:
        action.append(0)
    if state[6] == 0:
        action.append(1)
    else:
        action.append(0)
    if state[8] == 0 and pm:
        action.append(1)
    else:
        action.append(0)
    if state[7]:
        if action[2] == 0:
            action[2] = 2
        
    if pd:
        if state[3]:
            if action[0] == 0:
                action[0] = np.random.choice([2,3])
        if state[5]:
            if action[1] ==0:
                action[0] = np.random.choice([2,3])
            
            

    return np.array(action)

def test_config(config):
    config["simulation_time"] = 6*30*TimeFrame.DAY 
    if config['env_id'] == 1:
        pd = False
        pm = False
    elif config['env_id'] == 2:
        pm = True
        pd = False
    elif config['env_id'] == 3:
        pm = True
        pd = True
    env = f'test_{config["env_id"]}'
    for case in range(3):
        config["location_case"] = case
        config['env_id'] = f'{env}_{config["location_case"]}'
        env = CustomEnv(config)
        # cell and hidden state of the LSTM
        file_name = f'new_data/test_{config["env_id"]}.csv'
        # Episode start signals are used to reset the lstm states
        for i in range(10):
            obs = env.reset()
            total_return = 0
            dones = False
            while not dones:
                obs = next(reversed(obs.values()))
                action = choos_action(obs, pd, pm)
                print(action)
                print(obs)
                obs, rewards, dones, info = env.step(action)
                total_return += rewards
            info = env.company.get_summary()
            save_summary(file_name, info)
        print(f'total return : {total_return}')

if __name__ == '__main__':
    test_config(config1)
    test_config(config2)
    test_config(config3)
