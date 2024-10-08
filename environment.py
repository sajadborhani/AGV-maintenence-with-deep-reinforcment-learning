from cmath import inf
import math
from gymnasium import Env
from collections import OrderedDict
from gymnasium.spaces import MultiDiscrete, Dict
import numpy as np
from config import AGV
from constants import  Events, LocationCase, Strategy, TimeFrame
import logging
from utiles import Agv, Maintainer
import subprocess
from torch.utils.tensorboard import SummaryWriter
import os

class Company():
    def __init__(
        self,
        env_id='test',
        agv_number=3, 
        technician_number= 2, 
        simulation_time= 1*TimeFrame.DAY,
        strategy = Strategy.PREDICTIVE,
        location_case = LocationCase.Depot_Net_Maintenence,
        log_level = logging.DEBUG
    ):
        self.config_logger(log_level)
        self.agv_number = agv_number
        self.simulation_time = simulation_time
        self.strategy = strategy
        self.env_id = env_id
        self.vehicles = [Agv(
                id = f'{i}',
                env_id=env_id,
                location_case= location_case,
                **AGV
            ) for i in range(agv_number)]

        self.maintainer = Maintainer( technician_number, 2)
        
        for agv in self.vehicles:
            agv.register_maintainer(self.maintainer)

        self.continue_action = [0,0,0,0]

            
        tensorboard_file = 'charts_'
        tensorboard_dir = 'tensorboard'
        num = 1
        file_name = f'{tensorboard_file}_{num}'
        
        while file_name in os.listdir(tensorboard_dir):
            num += 1
            file_name = f'{tensorboard_file}_{num}'
        target_dir = os.mkdir( os.path.join(tensorboard_dir, file_name))

        self.writer = SummaryWriter(target_dir)


    def get_summary(self):
        res = []
        for agv in self.vehicles:
            res.append(agv.get_time_logs())
        return res
    def check_state_has_abnormality(self,state):
        if isinstance(state, OrderedDict):
            state = next(reversed(state.values()))
        
        abnormale = False
        if state[2] < 1 or state[4] < 1 or state[6] < 1:
            abnormale = True
        if state[3] or state[5] or state[7]:
            abnormale = True
        return abnormale
    
    def get_next_time_step(self):
        
        times = []
        
        times .append( self.maintainer.get_next_time_step() )
        for agv in self.vehicles:
            times.append( agv.get_end_of_event() )
        self.logger.debug(f'time steps accuired : {times}')
        time = min(times)
        if time == inf:
            self.logger.warning(f'infinite time step is happening check what is rong')
            raise Exception
        
        return time

    def run(self):
        self.logger.debug(f'time : {self.current_time}')
        while self.current_time < self.simulation_time:
            self.current_time = self.get_next_time_step()
            self.maintainer.task_manager( self.current_time )
            for index,agv in enumerate( self.vehicles):
                if agv.get_end_of_event() == self.current_time:
                    agv.event_manager(self.current_time)
                    state = agv.get_state_logs()
                    if self.check_state_has_abnormality(state) and len(agv.maintenence_queue) == 0 :
                        self.agv_index = index
                        self.logger.debug(f'state wich needs a decision : {state}')
                        if next(reversed( state ))[0] != Events.STANDBY_IN_MAINTENENCE and len(agv.maintenence_queue)== 0:
                            return state
                    else:
                        agv.event_manager(self.current_time)
        if self.current_time > self.simulation_time:
            self.done = True
            state = self.vehicles[self.agv_index].get_state_logs()
            self.logger.debug(f'episod ended - last state returned is : {state}')
            return state
                    # break

    def step(self,actions):

        reward = 0
        self.logger.debug(f'choosing action for AGV #{self.agv_index}')
        self.vehicles[self.agv_index].current_time = self.current_time
        actions,punish = self.vehicles[self.agv_index].check_chosen_actions(actions)
        self.logger.debug(f'punishment for rong actions : {punish}')
        reward += self.vehicles[self.agv_index].apply_actions(actions)
        self.logger.debug(f'time that the agv will spend in maintenence  : {reward}')
        next_state = self.run()
        self.logger.debug(f'next state pass {next_state} ')
        self.writer.add_scalar(f'environment/reward', reward, self.log_index)
        self.log_index += 1
        return next_state, reward + punish, self.done

    def reset(self):
        self.logger.info('reseting environment')
        self.current_time = 0
        self.log_index =  0
        self.done = False
        for v in self.vehicles :
            v.reset()
        self.maintainer.reset()
        
        state = self.run()

        self.logger.info(f'state : {state}')
        # self.run_powershell(f'get-content {self.file_name} -wait -tail 10')
        return state

    def config_logger(self, level):
        self.logger = logging.getLogger(f'company_logger')
        self.logger.setLevel(level)
        formatter = logging.Formatter('%(name)s - %(funcName)s - %(levelname)s - %(message)s')
        self.file_name = f'logs/company.log'
        ch = logging.FileHandler(self.file_name, mode='w')
        ch.setFormatter(formatter)
        ch.setLevel(logging.NOTSET)
        self.logger.addHandler(ch)

    def run_powershell(self, cmd):
        completed = subprocess.run(["powershell", "-Command", cmd], capture_output=True)
        return completed

   
class CustomEnv(Env):

    def __init__(self, config) -> None:
        super(CustomEnv, self).__init__()
        self.env_id = config['env_id'] 
        self.config = config
        self.company = Company(**config)
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()
        self.config_logger(level= config['log_level'])
        
        self.logger.debug(f'observation space : {self.observation_space}')
        self.logger.debug(f'action space : {self.action_space}')
        self.logger.info(f'initializing completed info:')
        # self.run_powershell(f'get-content {self.file_name} -wait -tail 10')

    def get_action_space(self):
        '''
        actions:
        |  mechanincal      electrical       battry          preventive_M |
        |  continue         continue         continue        continue     |
        |  corrective       corrective       corrective      preventive   |
        |  pd_repair        pd_repair        charge      
        |  pd_replace       pd_replace                                                                      |
        '''
        
        if self.config['strategy'] == Strategy.CORRECTIVE:
            actions = [2,2,3,1]

        elif self.config['strategy'] == Strategy.PREVENTIVE:
            actions = [2,2,3,2]
        
        elif self.config['strategy'] == Strategy.PREDICTIVE:
            actions = [4,4,3,2]

        return MultiDiscrete(actions)
    
    def get_observation_space(self):
        '''
        '''
        preventive_maintenence = AGV['periodic_maintenence']/ TimeFrame.HOUR
        obs_space = MultiDiscrete([11, 101, 101,2,101,2,101,2, preventive_maintenence +1])
        dic = {}
        for i in range(4):
            dic[f'{i}'] = obs_space
        return  Dict(dic)  

    def step(self, actions):

        state, reward, done = self.company.step(actions)
        self.logger.debug(f' next state achived : {state}')
        self.logger.info(f'reward : {reward}')
        self.last_reward = reward
        return state, float( reward ), done, {}

    def render(self):
        pass
    
    def reset(self):
        state  = self.company.reset()
        self.last_reward = 0
        # print(f'state : {state}')
        return state

    def run_powershell(self, cmd):
        completed = subprocess.run(["powershell", "-Command", cmd], capture_output=True)
        return completed
    def config_logger(self, level):
        self.logger = logging.getLogger(f'env_logger')
        self.logger.setLevel(level)
        formatter = logging.Formatter('%(name)s - %(funcName)s - %(levelname)s - %(message)s')
        self.file_name= f'logs/Env_{self.env_id}.log'
        ch = logging.FileHandler(self.file_name, mode='w')
        ch.setFormatter(formatter)
        ch.setLevel(logging.NOTSET)
        self.logger.addHandler(ch)
