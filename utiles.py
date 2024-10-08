
from cmath import inf
from collections import OrderedDict, namedtuple
from distutils import log
import enum
import logging
import math
from queue import Queue, Empty
import numpy
from numpy.random import default_rng
from constants import Action, ConstRewards, MaintenenceType, TimeFrame
from config import mechanical_sensors, electrical_sensors, battry_sensors
from scipy.stats import dweibull
from stable_baselines3.common.env_checker import check_env
import subprocess
import os
from torch.utils.tensorboard import SummaryWriter


random = default_rng()
MaintenenceTask = namedtuple('MaintenenceTask', ['agv', 'maintenence', 'part', 'sensor_name', 'cost', 'time', 'end', 'id'])

class Sensor():
    def __init__(self, part, normal_range, abnormal_range, pd_repair_dist, pd_replace_dist, sensor_name,abnormal_mttf_dist) -> None:
        self.name = sensor_name
        self.part = part
        self.abnormal_mttf_dist = abnormal_mttf_dist
        self.normal_range = normal_range
        self.abnormal_range = abnormal_range
        self.pd_repair_dist = pd_repair_dist
        self.pd_replace_dist = pd_replace_dist
        self.reset()

    def reset(self):
        self.__is_abnormal = False
        if self.part != Part.ELECTRICAL:
            self.value = random.uniform(self.normal_range[0], self.normal_range[1])
        else:
            self.value = random.choice([True, False])
        self.action = Action.CONTINUE

    def update_value(self, shape, p_health ):
        temp = random.random()
        if temp > 2 * dweibull( shape ).sf( 1 - p_health ) or self.is_abnormal():
            if self.part == Part.MECHANICAL:
                new_value =  (self.abnormal_range[1]-self.abnormal_range[0]) *  random.random() + self.abnormal_range[0]
            else:
                new_value = False
            abnormal = True
        else:
            if self.part == Part.MECHANICAL:
                new_value =  (self.normal_range[1]-self.normal_range[0]) *  random.random() + self.normal_range[0]
            else:
                new_value = True
            abnormal = False
        self.value = new_value
        self.__is_abnormal = abnormal
        return abnormal

    def is_abnormal(self):
        return self.__is_abnormal

    def change_action(self, action):
        self.action = action


class Battry():
    def __init__(self, battry_dist_param, logger) -> None:
        self.logger = logger
        self.normal_range = (20, 100)
        self.abnormal_range = (0, 20)
        self.max_time_usage = 30* TimeFrame.HOUR
        self.charging_time = 10*TimeFrame.HOUR
        self.dist_param = battry_dist_param
        self.reset()

    def reset(self, time=0):
        self.action = Action.Battry.CONTINUE
        self.charging = Action.Battry.CONTINUE
        self.value = 100
        self.p_health = 100
        self.abnormal = False
        self.life_time = self.dist_param
        self.broken = False
        self.end_of_life = time + self.life_time
        self.previos_p_health = 0

    def reduce_health(self, time, current_time):
        change = math.floor((time / self.life_time)*100)
        self.p_health = self.p_health - change
        if self.p_health <= 0 :
            self.p_health = 0
            self.broken = True
        self.end_of_life = current_time + math.floor((self.p_health/100) * self.life_time)
        return self.broken

    def consume(self, time):
        new_value = math.floor( self.value - (time/self.max_time_usage)*100)
        if new_value < 20:
            self.abnormal = True
        if new_value <= 0:
            new_value = 0
            self.broken = True
        self.value = new_value

        return self.abnormal

    def charge(self, time):
        new_value = math.floor( self.value + (time/self.charging_time)*100)
        if new_value > 100:
            new_value = 100
        elif new_value > 20:
            self.abnormal = False
        elif new_value < 0 :
            new_value = 100
        self.value = new_value

        return self.abnormal

    def apply_corrective_maintenence(self, current_time):
        self.reset(current_time)
    
    def apply_preventive_maintenence(self, current_time):
        if self.p_health > 99:
            self.p_health = 99
            self.broken = False
        elif self.p_health == 0:
            self.broken = True
        else:
            self.p_health = math.floor((99 - self.p_health) * random.random() + self.p_health)
            self.broken = False

        self.end_of_life = math.floor((self.p_health/100) * self.life_time + current_time)

    def is_broken(self):
        return self.broken

class Part():
    MECHANICAL = 0
    ELECTRICAL = 1
    BATTRY     = 2
    NAME = {
        0 : 'MECHANICAL',
        1 : 'ELECTRICAL',
        2 : 'BATTRY'
    }
    def __init__(self, part, dist_params, logger:logging.Logger) -> None:
        self.logger = logger
        self.part = part
        self.dist_params = dist_params
    def reset(self, time=0):
        self.logger.debug(f'reinitializing {Part.NAME[self.part]} part')
        if self.part == self.MECHANICAL :
            sensors = mechanical_sensors
        elif self.part == Part.ELECTRICAL:
            sensors = electrical_sensors    
        
        self.sensors = [ Sensor(**sensor)  for sensor in sensors]
        self.p_health = 100
        self.life_time = math.floor( self.dist_params[0] * random.weibull(  self.dist_params[1] ) )
        self.end_of_life = time + self.life_time
        self.action = 0
        self.abnormal = False
        self.previos_p_health = 0
        self.broken = False
        self.logger.debug(f'p_health : {self.p_health} abnormal : {self.abnormal} end of life : {self.end_of_life}')

    def reduce_health(self, time, current_time):
        '''time means the time frame that agv was working (on any task or just mocing )'''
        change = math.floor( (time/ self.life_time) * 100)
        self.p_health -= change
        if self.p_health <= 0 :
            self.p_health = 0
            self.broken = True
        self.logger.debug(f'reducing {Part.NAME[self.part]} parts health to {self.p_health} with {change}% change')
        self.end_of_life = current_time + math.floor((self.p_health/100) * self.life_time)
            
        return self.broken

    def update_sensors(self, current_time):
        eol = math.floor( (self.p_health/100) * self.life_time) + current_time 
        if eol <= self.end_of_life:
            self.end_of_life = eol
        shape = self.dist_params[1]
        p_health = self.p_health / 100
        for sensor in self.sensors:
            abnormal = sensor.update_value(shape, p_health)
            if abnormal:
                a,b = sensor.abnormal_mttf_dist
                mttf =  math.floor( random.normal(a,b) * TimeFrame.HOUR )
                if current_time + mttf < self.end_of_life:
                    self.end_of_life = current_time + mttf
                    if self.previos_p_health != 0:
                        self.previos_p_health = self.p_health
                    self.p_health = math.floor(((self.end_of_life - current_time) / self.life_time) * 100)
                    if self.p_health < 0:
                        self.p_health = 0
                        self.broken = True
            if abnormal and not self.abnormal:
                self.abnormal = abnormal
        return self.abnormal

    def apply_corrective_maintenence(self, current_time):
        self.reset(time = current_time)
    
    def apply_preventive_maintenence(self, current_time):
        if self.previos_p_health != 0:
            pp_health = self.previos_p_health
        else:
            pp_health = self.p_health
        self.previos_p_health = 0
        
        p_health = (100 - pp_health) * random.random() + pp_health
        if p_health > 100 or p_health < 0:
            p_health = 100
        
        self.p_health = p_health
        self.end_of_life = math.floor( (p_health/100) * self.life_time + current_time )
        if self.p_health>0:
            self.broken = False

    def apply_predictive_maintenence(self, current_time, sensor_name):
        abnormal = False
        for sensor in self.sensors:
            if sensor.name == sensor_name:
                sensor.reset()
            if sensor.is_abnormal():
                abnormal = True
        pp_health = self.previos_p_health
        if not abnormal:
            self.reset(current_time)
            p_health = (99 - pp_health) * random.random() + pp_health
            if p_health >= 100 or p_health < 0:
                p_health = 99
            self.p_health = p_health
            self.end_of_life = math.floor( (p_health/100) * self.life_time + current_time )
            self.abnormal = False
        if self.p_health > 0 :
            self.broken= False
    def get_pd_repair_dist(self, sensor_name):
        for sensor in self.sensors:
            if sensor.name == sensor_name:
                return sensor.pd_repair_dist
    def get_pd_replace_dist(self, sensor_name):
        for sensor in self.sensors:
            if sensor.name == sensor_name:
                return sensor.pd_replace_dist

    def is_broken(self):
        return self.broken

class Maintainer():
    def __init__(self, technician_num, production_machins_num) -> None:
        self.production_machins_num = production_machins_num
        self.technician_num = technician_num
        self.config_logger()

    def reset(self):
        self.task_queue = []
        self.next_task_id = 1
        self.technicians = [None for i in range(self.technician_num)]
        self.production_machins = [ random.normal(400 *  TimeFrame.HOUR, 120*TimeFrame.HOUR) for mechin in range( self.production_machins_num)]


    def register_task(self, task:MaintenenceTask):
        task = task._replace(id = self.next_task_id)
        self.next_task_id += 1
        self.logger.info(f'new task registered and added to maintenence queue -> {task}')
        self.task_queue.append(task)
        return task.id

    def get_waiting_time(self):
        times = [0 for tech in range(self.technician_num)]
        tech_turn = 0
        for task in self.task_queue:
            times[tech_turn] +=  task.time

        tech_turn += 1
        if tech_turn >= self.technician_num:
            tech_turn = 0
        return min(times) 

    def get_task(self):
        try :
            task = self.task_queue.pop(0)
            self.logger.debug(f'taking a task from queue to process -> {task}')
            return task
        except IndexError:
            self.logger.debug(f'no task in maintenence queue')
            return None

    def assign_task(self, tech):
        task = self.get_task()
        if task is not None:
            task = task._replace( end = self.current_time + task.time )
        self.logger.debug(f'assigining {task} to technician_{tech}')
        self.technicians[tech] = task

    def task_completed(self, task:MaintenenceTask):
        if task.agv is not None:
            task.agv.current_time = self.current_time
            task.agv.maintenence_task_done(task)
            self.logger.info(f'task {task} is completed')
        else:
            self.logger.info(f'production machin maintenence is done {task} ')

    def task_manager(self, current_time):
        self.logger.debug(f'{len(self.task_queue)} remaining tasks ')
        self.current_time = current_time
        for index, failure_time in enumerate( self.production_machins):
            if self.current_time >= failure_time:
                maintenence_time = math.floor( random.uniform(1,15) * TimeFrame.HOUR)
                task =  MaintenenceTask(None, MaintenenceType.PRODUCTION_MACHINES_MAINTENENCE, None, None, 0, maintenence_time, None, self.next_task_id)
                self.register_task(task )
                self.next_task_id += 1
                self.logger.info(f'a production machin failed and maintenence task registered {task}')
                self.production_machins[index] = self.current_time + random.normal(400 *  TimeFrame.HOUR, 120*TimeFrame.HOUR)

        for tech,task in enumerate( self.technicians):
            if task is None:
                self.assign_task(tech)
            elif self.current_time >= task.end:
                self.task_completed(task)
                self.assign_task(tech)
            else:
                self.logger.debug(f'technician_{tech} is doing task : {task}')

    def get_next_time_step(self):
        times = [inf]
        for task in self.technicians:
            if task is not None:
                times.append(task.end)
        return min(times)

    def config_logger(self):
        self.logger = logging.getLogger('technician_logger')
        # self.logger.level = logging.INFO
        self.logger.setLevel(logging.DEBUG)
        # self.logger.addHandler(logging.FileHandler('AVG.log',mode='w'))
        formatter = logging.Formatter('%(name)s - %(funcName)s - %(levelname)s - %(message)s')
        ch = logging.FileHandler('logs/tech.log', mode='w')
        ch.setFormatter(formatter)
        ch.setLevel(logging.NOTSET)
        # add ch to logger
        self.logger.addHandler(ch)

Event     = namedtuple('Event' , ['start', 'end', 'pause', 'event'])

from constants import Events, LocationCase

class Agv():

    def __init__(self, id,env_id, mechanical_dist_params, electrical_dist_params, battry_dist_params, location_case:int, periodic_maintenence, log_level, buffer_size = 100) -> None:
        self.id = id
        self.env_id = env_id
        self.mechanical_dist_params = mechanical_dist_params
        self.electrical_dist_params = electrical_dist_params
        self.battry_dist_params = battry_dist_params
        self.task_buffer = Queue(buffer_size)
        self.config_logger(log_level)
        self.periodic_maintenence_interval = periodic_maintenence
        self.periodic_maintenence_processing = False
        self.mechanincal =  Part(Part.MECHANICAL, self.mechanical_dist_params, self.logger)
        self.electrical = Part(Part.ELECTRICAL, self.electrical_dist_params, self.logger)
        self.battry     = Battry(self.battry_dist_params, self.logger)
        self.location_case = LocationCase(location_case)
        self.log_chart_number = 0
        self.pm_delay_number = 0
        tensorboard_file = 'charts_'
        tensorboard_dir = 'tensorboard'
        num = 1
        file_name = f'{tensorboard_file}_{num}'

        while file_name in os.listdir(tensorboard_dir):
            num += 1
        num -= 1    
        file_name = f'{tensorboard_file}_{num}'
        target_dir = os.mkdir( os.path.join(tensorboard_dir, file_name))
        
        self.writer = SummaryWriter(target_dir)

    def reset(self, current_time = 0):
        self.logger.debug('using reset for agv')
        self.__working_time = 0
        self.__working_time_checkpoint = 0
        self.__maintenence_time = 0
        self.__broken_time  = 0
        self.__moving_time  = 0
        self.__standby_time = 0
        self.__corrective_maintenence_time = 0
        self.__preventive_maintenence_time = 0
        self.__predictive_maintenence_time = 0
        self.__battry_charge_time            = 0  
        self.task_generator_checkpoint = 0
        self.periodic_maintenence = self.periodic_maintenence_interval
        self.maintenence_queue = []
        self.undone_maintenence = []
        self.current_time = current_time
        self.progressing_event = Event(0, TimeFrame.MINUTE, None, Events.STANDBY_IN_DEPOT)
        self.mechanincal.reset(current_time)
        self.electrical.reset(current_time)
        self.battry.reset(current_time)
        self.logger.debug(f'initial state : {self.export_agv_state()}')
        self.state_log = [self.export_agv_state() for i in range(4)]

    def get_time_logs(self):
        res = {
            'AGV_id' : self.id,
            'working_time' : self.__working_time, 
            'maintenence_time' : self.__maintenence_time, 
            'broken_time' : self.__broken_time,  
            'moving_time' : self.__moving_time,  
            'standby_time' : self.__standby_time, 
            'corrective_maintenence_time' : self.__corrective_maintenence_time, 
            'preventive_maintenence_time' : self.__preventive_maintenence_time, 
            'predictive_maintenence_time' : self.__predictive_maintenence_time, 
            'battry_charge_time' : self.__battry_charge_time,            
            }
        return res

    def register_maintainer(self, maintainer:Maintainer):
        self.maintainer = maintainer

    def is_abnormal(self):
        if self.mechanincal.abnormal or self.electrical.abnormal or self.battry.abnormal:
            return True
        else:
            return False

    def is_broken(self):
        if self.mechanincal.p_health == 0 or self.electrical.p_health == 0 or self.battry.p_health == 0:
            return True
        else:
            return False

    def task_generator(self, time_frame):
        punish = 0
        unregistered = 0
        for i in range(math.floor( time_frame / TimeFrame.MINUTE) ):
            r = random.random()
            if r > .90:
                self.logger.debug(f'new task added to agv buffer')
                if not self.task_buffer.full():
                    self.task_buffer.put( int(3600000 * random.random()) + 1800000)
                else:
                    punish += ConstRewards.FULL_BUFFER
                    unregistered += 1
        self.logger.debug(f'{unregistered} unregistered tasks are denied')
        return punish
    
    def arriving_at_maintenence(self):
        self.logger.info(f'AGV arrived at maintenence')
        self.progressing_event = Event(self.current_time, self.current_time, None, Events.STANDBY_IN_MAINTENENCE)
        
    def arriving_at_depot(self):
        self.logger.info(f'AGV arrived at depot')
        self.progressing_event = Event(self.current_time,self.current_time , None, Events.STANDBY_IN_DEPOT)

    def arriving_at_site_net(self):
        self.logger.info(f'AGV arrived at site-net')
        self.assign_new_task_in_sitenet()

    def task_ended(self):
        self.logger.info(f'task ended -> {self.progressing_event} standby in sitenet')
        self.progressing_event = Event(self.current_time, self.current_time, None, Events.STANDBY_IN_SITENET)

    def manage_standby_in_sitenet(self):
        
        if len(self.maintenence_queue) > 0:
            self.logger.info(f'maintenence decided - going to maintenence')
            time = self.location_case.get_edge_distance(LocationCase.net_maintenence_edge)
            self.progressing_event = Event(self.current_time, self.current_time + time, None, Events.MOVING_SITENET_TO_MAINTENENCE)
        else:
            self.assign_new_task_in_sitenet()

    def assign_new_task_in_sitenet(self):
        try:
            task_time = self.task_buffer.get_nowait()
            self.progressing_event = Event(self.current_time, self.current_time + task_time, None, Events.TASK)
            self.logger.info(f'new task assigned ending at {self.current_time + task_time}')
        except Empty:
            self.logger.info(f" no task in AGV's buffer, going to depot")
            time = self.location_case.get_edge_distance(LocationCase.net_depot_edge)
            self.progressing_event = Event(self.current_time, self.current_time + time, None, Events.MOVING_SITENET_TO_DEPOT )
    
    def depot_task_check(self):
        if len(self.maintenence_queue) > 0:
            self.logger.info(f'maintenence decided for agv - going to maintenence from depot')
            time = self.location_case.get_edge_distance(LocationCase.depot_maintenence_edge)
            self.progressing_event = Event(self.current_time, self.current_time + time, None, Events.MOIVING_DEPOT_TO_MAINTENENCE)
        elif not self.task_buffer.empty():
            self.logger.info(f'going to site-net to do new tasks')
            time = self.location_case.get_edge_distance(LocationCase.net_depot_edge)
            self.progressing_event = Event(self.current_time, self.current_time + time, None, Events.MOVING_DEPOT_TO_SITENET )
        else:
            self.logger.info('no new task is buffer and no maintenence decided - staying at depot for another minute')
            self.progressing_event = Event(self.current_time, self.current_time + TimeFrame.MINUTE, None, Events.STANDBY_IN_DEPOT)
    
    def process_maintenences(self):
        self.logger.debug(f'processing if there is any maintenence decided for the agv')

        if self.battry.value < 5:
            charge = True
            for task in self.maintenence_queue:
                if task.maintenence == MaintenenceType.BATTRY_CHARGE:
                    charge = False
            if charge:
                time = self.get_maintenence_time(Part.BATTRY, MaintenenceType.BATTRY_CHARGE, 'battry_charge')
                task = MaintenenceTask(self, MaintenenceType.BATTRY_CHARGE, Part.BATTRY, 'battry_charge', 0, time, None, None)
                self.maintenence_queue.append(task)
        if len(self.maintenence_queue) > 0 :
            while len(self.maintenence_queue) != 0:
                task = self.maintenence_queue.pop(0)
                id = self.maintainer.register_task(task)
                self.undone_maintenence.append(id)
            self.logger.debug(f'{len(self.undone_maintenence)} maintenence task are regestered and contuing the process')
            self.progressing_event = Event(self.current_time, inf, None, Events.MAINTENENCE)
        else:
            if not self.is_broken():
                self.logger.info(f'no maintenence to register- going back to work cycle')
                if not self.task_buffer.empty():
                    self.logger.info(f'going to site-net to do new tasks')
                    time = self.location_case.get_edge_distance(LocationCase.net_maintenence_edge)
                    self.progressing_event = Event(self.current_time, self.current_time + time, None, Events.MOVING_DEPOT_TO_SITENET )
                else:
                    self.logger.info('no new task is buffer - going to depot')
                    time = self.location_case.get_edge_distance(LocationCase.net_depot_edge)
                    self.progressing_event = Event(self.current_time, self.current_time + time, None, Events.MOVING_MAINTENENCE_TO_DEPOT)
            else:
                self.logger.debug(f'AGV is broken and in maintenence but agent has not decided to repair it giving it another minute')
                self.progressing_event = Event(self.current_time, self.current_time+TimeFrame.MINUTE, None, Events.STANDBY_IN_MAINTENENCE)            
        self.logger.debug(f'progressing event : {self.progressing_event}')

    def get_end_of_event(self):
        return self.progressing_event.end

    def event_manager(self, current_time):
        # managing events and assigning reward
        reward = 0
        self.logger.debug(f'state : {self.export_agv_state()}')
        self.logger.debug(f'mechanical broken : {self.mechanincal.broken}')
        self.logger.debug(f'electrical broken : {self.electrical.broken}')
        self.logger.debug(f'battry charge : {self.battry.value}')
        
        self.logger.debug(f' working time : {self.__working_time} - hour : {self.__working_time / TimeFrame.HOUR}')
        self.logger.debug(f' maintenence time : {self.__maintenence_time} - hour : {self.__maintenence_time / TimeFrame.HOUR}')
        self.logger.debug(f' moving time : {self.__moving_time} - hour : {self.__moving_time / TimeFrame.HOUR}')
        self.logger.debug(f' broken time : {self.__broken_time} - hour : {self.__broken_time / TimeFrame.HOUR}')
        self.logger.debug(f' standby time : {self.__standby_time} - hour : {self.__standby_time / TimeFrame.HOUR}')
        self.logger.debug(f'corrective maintenence time : {self.__corrective_maintenence_time} - hour : {self.__corrective_maintenence_time/ TimeFrame.HOUR}')
        self.logger.debug(f'preventive maintenence time : {self.__preventive_maintenence_time} - hour : {self.__preventive_maintenence_time/ TimeFrame.HOUR}')
        self.logger.debug(f'predictive maintenence time : {self.__predictive_maintenence_time} - hour : {self.__predictive_maintenence_time/ TimeFrame.HOUR}')
        self.logger.debug(f'battry_charge time : {self.__battry_charge_time} - hour : {self.__battry_charge_time/ TimeFrame.HOUR}')
        
        self.logger.debug(f'processing event is {Events.NAME[self.progressing_event.event]}')
        self.time_frame = self.progressing_event.end - self.progressing_event.start
        frame = current_time - self.task_generator_checkpoint
        reward += self.task_generator(frame)
        self.task_generator_checkpoint = current_time
        self.current_time = current_time
        if self.is_broken():
            reward -= frame * 10**6
            
        self.logger.debug(f'current time :{self.current_time} and event : {self.progressing_event}')
        
        if self.current_time < self.progressing_event.end:
            self.logger.debug(f' normally this should not happen because event manager is called at the end of an event')
        
        if self.current_time >= self.progressing_event.end and self.progressing_event.event in Events.ACTIVE_EVENTS:
            
            self.logger.debug('reducing agv health after a working state')
            if not self.is_broken():
                self.mechanincal.reduce_health(self.time_frame, self.current_time)
                self.mechanincal.update_sensors(self.current_time)
                self.electrical.reduce_health(self.time_frame, self.current_time)
                self.electrical.update_sensors(self.current_time)
                self.battry.reduce_health(self.time_frame, self.current_time)
                self.battry.consume(self.time_frame)
            
            if self.progressing_event.event == Events.MOIVING_DEPOT_TO_MAINTENENCE or self.progressing_event.event == Events.MOVING_SITENET_TO_MAINTENENCE:
                self.arriving_at_maintenence()
                self.__moving_time += self.time_frame    
            elif self.progressing_event.event == Events.MOVING_MAINTENENCE_TO_DEPOT or self.progressing_event.event == Events.MOVING_SITENET_TO_DEPOT:
                self.arriving_at_depot()
                self.__moving_time += self.time_frame    
            
            elif self.progressing_event.event == Events.MOVING_MAINTENENCE_TO_SITENET or self.progressing_event.event == Events.MOVING_DEPOT_TO_SITENET:
                self.arriving_at_site_net()
                self.__moving_time += self.time_frame    
            
            elif self.progressing_event.event == Events.TASK:
                self.task_ended()
                self.__working_time += self.time_frame

        elif self.progressing_event.event in Events.STANDBY_EVENTS:
            self.__standby_time += self.time_frame
            if self.progressing_event.event == Events.STANDBY_IN_MAINTENENCE:
                self.process_maintenences()
            elif self.is_broken():
                self.__broken_time += self.time_frame
                self.logger.info(f'AGV is broken going to maintenence until agent decideds a maintenence')
                if self.progressing_event.event == Events.STANDBY_IN_SITENET:
                    time = self.location_case.get_edge_distance(LocationCase.net_maintenence_edge)
                    event = Events.MOVING_SITENET_TO_MAINTENENCE
                elif self.progressing_event.event == Events.STANDBY_IN_DEPOT:
                    time = self.location_case.get_edge_distance(LocationCase.depot_maintenence_edge)
                    event = Events.MOIVING_DEPOT_TO_MAINTENENCE

                self.progressing_event = Event(self.current_time, self.current_time + time, None, event)
            else:    
                self.logger.debug(f'AGV is in standby mode')
                if self.progressing_event.event == Events.STANDBY_IN_SITENET:
                    self.manage_standby_in_sitenet()
                elif self.progressing_event.event == Events.STANDBY_IN_DEPOT:
                    self.depot_task_check()
        del self.state_log[0]
        self.state_log.append(self.export_agv_state())
        return reward
    

    def export_agv_state(self):
        state = []
        state.append(self.progressing_event.event)
        state.append(self.task_buffer.qsize())
        state.append(self.mechanincal.p_health)
        state.append(self.mechanincal.abnormal)
        state.append(self.electrical.p_health)
        state.append(self.electrical.abnormal)
        state.append(self.battry.p_health)
        state.append(self.battry.abnormal)
        preventive = self.periodic_maintenence - self.current_time
        if preventive < 0:
            preventive = 0
        else:
            preventive = math.floor(preventive / TimeFrame.HOUR)
        state.append(preventive)
        return state

    def get_state_logs(self):
        state = OrderedDict()
        for index,st in enumerate( self.state_log ):
            state [f'{index}'] = st
        return state
      
    def get_maintenence_cost(self,maintenence:MaintenenceType, time:int):
        cost = 0
        if maintenence == MaintenenceType.BATTRY_CHARGE:
            return 0
        if maintenence == MaintenenceType.PREVENTIVE:
            cost += math.floor( 500 * random.random())
        elif maintenence == MaintenenceType.PD_REPLACE:
            cost += math.floor( 1500 * random.random() + 500)
        elif maintenence == MaintenenceType.CORRECTIVE:
            cost += math.floor( 2500 * random.random() + 1500)
        cost += (time/TimeFrame.HOUR) * 100
        return math.floor( cost)

    def get_maintenence_time(self, part:Part, maintenence:MaintenenceType, sensor_name= None):
        if part == Part.MECHANICAL:
            if maintenence == MaintenenceType.PD_REPAIR and sensor_name is not None:
                loc, scale = self.mechanincal.get_pd_repair_dist(sensor_name)
            elif maintenence == MaintenenceType.PD_REPLACE and sensor_name is not None:
                loc, scale = self.mechanincal.get_pd_replace_dist(sensor_name)
            elif maintenence == MaintenenceType.CORRECTIVE and sensor_name is None:
                loc, scale = (22,3)
            else:
                self.logger.warning(f'{MaintenenceType.NAME[maintenence]} maintenence for {Part.NAME[part]} and sensor {sensor_name} is invalide ')
                raise AttributeError
        elif part == Part.ELECTRICAL:
            if maintenence == MaintenenceType.PD_REPAIR and sensor_name is not None:
                loc, scale = self.electrical.get_pd_repair_dist(sensor_name)
            elif maintenence == MaintenenceType.PD_REPLACE and sensor_name is not None:
                loc, scale = self.electrical.get_pd_replace_dist(sensor_name)
            elif maintenence == MaintenenceType.CORRECTIVE and sensor_name is None:
                loc, scale = (22,3)
            else:
                self.logger.warning(f'{MaintenenceType.NAME[maintenence]} maintenence for {Part.NAME[part]} and sensor {sensor_name} is invalide ')
                raise AttributeError
                
        elif part == Part.BATTRY:
            if maintenence == MaintenenceType.BATTRY_CHARGE:
                return (10* TimeFrame.HOUR / 100) * ( 100 - self.battry.value )
            elif sensor_name is None and maintenence == MaintenenceType.CORRECTIVE:
                loc, scale = (22,3)

        elif part is None and maintenence == MaintenenceType.PREVENTIVE:            
                return 60*60*1000
        else:
            self.logger.warning(f'AGV-{self.__id} -- get sensor pd maintenence time got unexpected values : part= {Part.NAME[ part]}, maintenence_type = {MaintenenceType.NAME[ maintenence]}, sensorname = {sensor_name}')
            raise Exception

        return math.floor(TimeFrame.HOUR * random.normal(loc, scale) + TimeFrame.HOUR/2) 

    def do_preventive_maintenence(self):
        self.mechanincal.apply_preventive_maintenence(self.current_time)
        self.electrical.apply_preventive_maintenence(self.current_time)
        self.battry.apply_preventive_maintenence(self.current_time)
        self.periodic_maintenence_processing = False
        self.periodic_maintenence = self.current_time + self.periodic_maintenence_interval
        self.logger.debug(f'periodic maintenence decided for agv next pm is at {self.periodic_maintenence}')
        self.logger.debug(f'preventive / periodic maintenence applied')

    def maintenence_task_done(self, task:MaintenenceTask):
        self.__maintenence_time += task.time
        if task.maintenence == MaintenenceType.PREVENTIVE:
            self.__preventive_maintenence_time += task.time
            self.do_preventive_maintenence()
        elif task.maintenence == MaintenenceType.CORRECTIVE:
            self.__corrective_maintenence_time += task.time
            if task.part == Part.MECHANICAL:
                self.mechanincal.apply_corrective_maintenence(task.end)
            elif task.part == Part.ELECTRICAL:
                self.electrical.apply_corrective_maintenence(task.end)
            elif task.part == Part.BATTRY:
                self.battry.apply_corrective_maintenence(task.end)
        elif task.maintenence == MaintenenceType.PD_REPAIR or task.maintenence == MaintenenceType.PD_REPLACE:
            self.__predictive_maintenence_time += task.time
            if task.part == Part.MECHANICAL:
                self.mechanincal.apply_predictive_maintenence(task.end, task.sensor_name)
            elif task.part == Part.ELECTRICAL:
                self.electrical.apply_predictive_maintenence(task.end, task.sensor_name)
        elif task.maintenence == MaintenenceType.BATTRY_CHARGE:
            self.__battry_charge_time += task.time
            self.battry.charge(task.time)
            self.battry.charging = Action.Battry.CONTINUE
        # elif task.maintenence == MaintenenceType.PRODUCTION_MACHINES_MAINTENENCE:
        #     pass
        else:
            self.logger.info(f'chack what has happend during this maintenence : {task}')
        self.logger.debug(f'maintenence task : {task} is done and removing {task.id} from list')
        self.undone_maintenence.remove(task.id)
        self.logger.debug(self.undone_maintenence)
        if len(self.undone_maintenence) == 0:
            self.logger.info(f'all registred maintenences are done and event is standby in maintenence')
            self.progressing_event = Event(task.end , task.end, None, Events.STANDBY_IN_MAINTENENCE)
            self.logger.debug(f'event : { self.progressing_event } ')
              
    def check_chosen_actions(self, actions):
        '''
        returns a tupel (masked actions, punish)
        '''
        self.logger.debug(f'decided actions : {actions}')
        punish = 0
        preventive = actions[3]
        actions = actions[0:3]
        # broken = self.is_broken()
        # abnormal = self.is_abnormal()
        # if not broken or not abnormal:
        #     for act in actions:
        #         if 
        if preventive == Action.Preventive.PREVENTIVE_MAINTENENCE:
            if self.current_time < self.periodic_maintenence  :
                punish += ConstRewards.INVALID_ACTION
                preventive = Action.Preventive.CONTINUE
            else:
                self.logger.debug(f'periodic maintenence decided correctly at {self.current_time} and schedule is for {self.periodic_maintenence}')
                
        elif preventive == Action.Preventive.CONTINUE and self.current_time > self.periodic_maintenence:
            punish += ( self.periodic_maintenence - self.current_time )*10**5

        for part, action in enumerate(actions):
            if part == Part.MECHANICAL:
                if action == Action.CORRECTIVE_MAINTENENCE and  self.mechanincal.p_health > 20:
                    punish += ConstRewards.INVALID_ACTION                    
                    actions[part] = Action.CONTINUE
                
                elif action != Action.CORRECTIVE_MAINTENENCE and self.mechanincal.p_health == 0 :
                    punish += ConstRewards.BIG_PUNISH 

                elif (action == Action.PD_REPAIR or action == Action.PD_REPLACE) \
                    and (not self.mechanincal.abnormal or self.mechanincal.p_health ==0):
                    punish += ConstRewards.INVALID_ACTION                    
                    actions[part] = Action.CONTINUE

            if part == Part.ELECTRICAL:
                if action == Action.CORRECTIVE_MAINTENENCE and self.electrical.p_health > 20 :
                    punish += ConstRewards.INVALID_ACTION                  
                    actions[part] = Action.CONTINUE
                
                elif action != Action.CORRECTIVE_MAINTENENCE and self.electrical.p_health == 0 :
                    punish += ConstRewards.BIG_PUNISH 

                elif (action == Action.PD_REPAIR or action == Action.PD_REPLACE) \
                    and (not self.electrical.abnormal or self.mechanincal.p_health ==0 ):
                    punish += ConstRewards.INVALID_ACTION                    
                    actions[part] = Action.CONTINUE
            if part == Part.BATTRY:
                if action == Action.CORRECTIVE_MAINTENENCE and self.battry.p_health > 20 :
                    punish += ConstRewards.INVALID_ACTION 
                    actions[part] = Action.Battry.CONTINUE
                    
                
                elif action != Action.CORRECTIVE_MAINTENENCE and self.battry.p_health == 0 :
                    punish += ConstRewards.BIG_PUNISH 

                if action == Action.Battry.CHARGE and self.battry.value >20:
                    actions[part] = Action.Battry.CONTINUE
                    punish += ConstRewards.INVALID_ACTION 

        return numpy.append(actions, preventive), punish
    
    def tensorboard_logger(self, pm_delay= None):
        buffer = self.task_buffer.qsize()
        maintenence_tasks = len(self.maintainer.task_queue)
        time_to_start_maintenence = self.maintainer.get_waiting_time()
        self.writer.add_scalar(f'AGV_{self.id}/buffer_task', buffer, self.log_chart_number )
        self.writer.add_scalar(f'AGV_{self.id}/maintenence_buffer', maintenence_tasks, self.log_chart_number)
        self.writer.add_scalar(f'AGV_{self.id}/waiting_time', time_to_start_maintenence, self.log_chart_number)
        time_to_break_down = min( (self.mechanincal.end_of_life - self.current_time), \
                                    (self.electrical.end_of_life - self.current_time),\
                                        (self.battry.end_of_life - self.current_time)  ) 
        
        self.writer.add_scalar(f'AGV_{self.id}/remaining_time_to_breakdown', time_to_break_down, self.log_chart_number)
        
        if pm_delay is not None:
            self.writer.add_scalar(f'AGV_{self.id}/preventive_maintenence_delay', pm_delay, self.pm_delay_number)
            self.pm_delay_number += 1
        self.log_chart_number += 1

    def apply_actions(self, actions):
        self.logger.debug(f'applying actions {actions}')
        working_time = self.__working_time - self.__working_time_checkpoint
        self.__working_time_checkpoint = self.__working_time
        maintenence_time = 0
        preventive = actions[3]
        actions = actions[0:3]
        pm_delay = None
        
        if preventive == Action.Preventive.PREVENTIVE_MAINTENENCE:
            time = self.get_maintenence_time(None, MaintenenceType.PREVENTIVE)
            cost = self.get_maintenence_cost(MaintenenceType.PREVENTIVE, time)
            self.maintenence_queue.append(
                MaintenenceTask(self, MaintenenceType.PREVENTIVE, None, None, cost, time, None, None )
            )
            self.periodic_maintenence_processing = True
            maintenence_time += time
            pm_delay = self.periodic_maintenence
        else:
            for part, action in enumerate(actions):
                self.logger.debug(f'decided actions {action} for {Part.NAME[part]} part')
                if action == Action.CORRECTIVE_MAINTENENCE:
                    self.logger.debug(f'processing action CORRECTIVE MAINTENENCE for {Part.NAME[part]} part')
                    time = self.get_maintenence_time(part, MaintenenceType.CORRECTIVE)
                    cost = self.get_maintenence_cost(MaintenenceType.CORRECTIVE, time)
                    self.maintenence_queue.append(
                        MaintenenceTask(self, MaintenenceType.CORRECTIVE, part, None,cost,time, None, None)
                        )
                    if part == Part.MECHANICAL:
                        self.mechanincal.action = Action.CORRECTIVE_MAINTENENCE
                    elif part == Part.ELECTRICAL:
                        self.electrical.action = Action.CORRECTIVE_MAINTENENCE
                    elif part == Part.BATTRY:
                        self.battry.action = Action.Battry.CORRECTIVE_MAINTENENCE
                    maintenence_time += time
                else:
                    if part == Part.BATTRY:
                        if action == Action.Battry.CHARGE:
                            self.logger.debug(f'processing action battry charge decided')
                            time = self.get_maintenence_time(Part.BATTRY, MaintenenceType.BATTRY_CHARGE, 'battry_charge')
                            cost = 0
                            self.maintenence_queue.append(
                                MaintenenceTask(self, MaintenenceType.BATTRY_CHARGE, Part.BATTRY, 'battry_charge', cost, time, None, None)
                            )
                            self.battry.action = Action.Battry.CHARGE
                            maintenence_time += time
                    else:
                        if action == Action.PD_REPAIR or action == Action.PD_REPLACE:
                            self.logger.debug(f'predictive repair decided for {Part.NAME[part]}')

                            if part == Part.MECHANICAL:
                                for sensor in self.mechanincal.sensors:
                                    if sensor.is_abnormal():
                                        time = self.get_maintenence_time(part, action,sensor.name)
                                        cost = self.get_maintenence_cost(MaintenenceType.PD_REPAIR, time)
                                        self.maintenence_queue.append(
                                            MaintenenceTask(self, action , part, sensor.name,cost,time, None, None)
                                            )
                                        sensor.action = action
                                        maintenence_time += time
                            elif part == Part.ELECTRICAL:
                                for sensor in self.electrical.sensors:
                                    if sensor.is_abnormal():
                                        time = self.get_maintenence_time(part, action ,sensor.name)
                                        cost = self.get_maintenence_cost(action , time)
                                        self.maintenence_queue.append(
                                            MaintenenceTask(self, action , part, sensor.name,cost,time, None, None)
                                            )
                                        sensor.action = action
                                        maintenence_time += time
        self.tensorboard_logger(pm_delay= pm_delay)
        self.logger.debug(f'maintenence queue :{self.maintenence_queue}')
        return -1 * maintenence_time + working_time

    # def step(self, currnt_time, actions):    
    #     self.logger.info(f'decided actions : {actions}')    
    #     actions, punish = self.check_chosen_actions(actions)
    #     self.logger.info(f'masked actions : {actions}')    
    #     maintenence_time = self.apply_actions(actions)
    #     reward = self.event_manager(currnt_time)
    #     self.logger.debug(f' working time : {self.__working_time} - hour : {self.__working_time / TimeFrame.HOUR}')
    #     self.logger.debug(f' maintenence time : {self.__maintenence_time} - hour : {self.__maintenence_time / TimeFrame.HOUR}')
    #     self.logger.debug(f' moving time : {self.__moving_time} - hour : {self.__moving_time / TimeFrame.HOUR}')
    #     self.logger.debug(f' broken time : {self.__broken_time} - hour : {self.__broken_time / TimeFrame.HOUR}')
    #     self.logger.debug(f' standby time : {self.__standby_time} - hour : {self.__standby_time / TimeFrame.HOUR}')
    #     self.logger.debug(f'corrective maintenence time : {self.__corrective_maintenence_time} - hour : {self.__corrective_maintenence_time/ TimeFrame.HOUR}')
    #     self.logger.debug(f'preventive maintenence time : {self.__preventive_maintenence_time} - hour : {self.__preventive_maintenence_time/ TimeFrame.HOUR}')
    #     self.logger.debug(f'predictive maintenence time : {self.__predictive_maintenence_time} - hour : {self.__predictive_maintenence_time/ TimeFrame.HOUR}')
    #     self.logger.debug(f'battry_charge time : {self.__battry_charge_time} - hour : {self.__battry_charge_time/ TimeFrame.HOUR}')
    #     return -1 * maintenence_time + punish + reward

    def run_powershell(self, cmd):
        completed = subprocess.run(["powershell", "-Command", cmd], capture_output=True)
        return completed

    def config_logger(self, level=logging.DEBUG):
        
        self.logger = logging.getLogger(f'agv_logger_{self.id}')
        # self.logger.level = logging.INFO
        self.logger.setLevel(level)
        self.file_name = f'logs/AGV_{self.env_id}_{self.id}.log'
        # self.logger.addHandler(logging.FileHandler('AVG.log',mode='w'))
        formatter = logging.Formatter('%(name)s - %(funcName)s - %(levelname)s - %(message)s')
        ch = logging.FileHandler(self.file_name, mode='w')
        ch.setFormatter(formatter)
        ch.setLevel(logging.NOTSET)
        # add ch to logger
        self.logger.addHandler(ch)
