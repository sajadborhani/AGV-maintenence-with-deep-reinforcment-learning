from constants import TimeFrame, Strategy, LocationCase
import logging

ITERATION = 100

config1 = {
    'agv_number'        : 3,
    'technician_number' : 2, 
    'simulation_time'   : 365 * TimeFrame.DAY,
    'strategy'          : Strategy.CORRECTIVE,
    'location_case'     : LocationCase.Depot_Net_Maintenence,
    'log_level'         : logging.DEBUG,
    'env_id'               : 1
}

config2 = {
    'agv_number'        : 3,
    'technician_number' : 2, 
    'simulation_time'   : 365 * TimeFrame.DAY,
    'strategy'          : Strategy.PREVENTIVE,
    'location_case'     : LocationCase.Depot_Net_Maintenence,
    'log_level'         : logging.DEBUG,
    'env_id'               : 2
}

config3 = {
    'agv_number'        : 3,
    'technician_number' : 2, 
    'simulation_time'   :  30* TimeFrame.DAY,
    'strategy'          : Strategy.PREDICTIVE,
    'location_case'     : LocationCase.Depot_Net_Maintenence,
    'log_level'         : logging.DEBUG,
    'env_id'               : 3
}

def generate_config_with_diffrent_location_cases(config: dict, location_case):
    conf = config.copy()
    conf['location_case'] = location_case
    return conf

mechanical_sensors = [
    {
            'part' : 0,
            'sensor_name' :'energy_consumption' ,
            'normal_range' : (0,1000)   , 
            'abnormal_range' : (1000,1500)  , 
            'abnormal_mttf_dist' : (14,5),
            'pd_repair_dist' :(4,1), 
            'pd_replace_dist' :(2,1) , }
            ,
            {
            'part' : 0,
            'sensor_name' :'vibration' ,
            'normal_range' : (1.8, 2.8)   , 
            'abnormal_range' : (2.9, 5)   , 
            'abnormal_mttf_dist' : (8,3),
            'pd_repair_dist' :(4,1), 
            'pd_replace_dist' :(2,1) , 
            },
            {
            'part' : 0,
            'sensor_name' :'temperature' ,
            'normal_range' : (20,40)   , 
            'abnormal_range' : (40,60)   , 
            'abnormal_mttf_dist' : (15,4),
            'pd_repair_dist' :(4,1), 
            'pd_replace_dist' : (2,1) },
            
            {
            'part' : 0,
            'sensor_name' :'noise' ,
            'normal_range' :(20, 30)   , 
            'abnormal_range' : (30, 50)   , 
            'abnormal_mttf_dist' : (25,2),
            'pd_repair_dist' :(4,1), 
            'pd_replace_dist' : (2,1) },
            
            {
            'part' : 0,
            'sensor_name' :'electrical_break' ,
            'normal_range' :  (1.95,2.05)   , 
            'abnormal_range' : (.95, 1.05)  , 
            'abnormal_mttf_dist' : (10,3),
            'pd_repair_dist' :(4,1), 
            'pd_replace_dist' : (2,1)  }
            ,
            {
            'part' : 0,
            'sensor_name' :'mechanical_break' ,
            'normal_range' :  (.95,1.05)    , 
            'abnormal_range' : (0,.95)  , 
            'abnormal_mttf_dist' : (10,3),
            'pd_repair_dist' :(4,1), 
            'pd_replace_dist' : (2,1) , }
            ,
            ]
electrical_sensors = [
       {
            'part' :1,
            'sensor_name' :'emmiters' ,
            'normal_range' :  True   , 
            'abnormal_range' : False  , 
            'abnormal_mttf_dist' :(5,2),
            'pd_repair_dist' :(4,1), 
            'pd_replace_dist' : (2,1) , }
            ,
            {
            'part' :1,
            'sensor_name' :'visual' ,
            'normal_range' :  True    , 
            'abnormal_range' : False  , 
            'abnormal_mttf_dist' :(20,5),
            'pd_repair_dist' :(4,1), 
            'pd_replace_dist' : (2,1) , }
            ,
         
]
battry_sensors = {
            'part' : 2,
            'sensor_name' :'battry_charge' ,
            'normal_range' :  (20,80)  , 
            'abnormal_range' :(0,20)  , 
            'abnormal_mttf_dist' :(6,1),
            'pd_repair_dist' :(8,1), 
            'pd_replace_dist' : (8,1) ,
             }

AGV = {
    'periodic_maintenence' : 165 * TimeFrame.HOUR , 
    'log_level' :  logging.DEBUG,
    'mechanical_dist_params' : (100*TimeFrame.HOUR,10),
    'electrical_dist_params' : (150*TimeFrame.HOUR,10), 
    'battry_dist_params' : 500*TimeFrame.HOUR
}