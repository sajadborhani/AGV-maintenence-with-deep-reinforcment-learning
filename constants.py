
class ActionV1:
    class Major:
        COUNTINUE                      = 0
        CORRECTIVE_MAINTENENCE         = 1
        PREVENTIVE_MAINTENENCE         = 2
        NAME = {
            0 : 'COUNTINUE',
            1 : 'CORRECTIVE_MAINTENENCE',
            2 : 'PREVENTIVE_MAINTENENCE'
        }
    class Minor:
        COUNTINUE                      = 0
        REPAIR_PREDICTIVE_NAINTENENCE  = 1
        REPLACE_PREDICTIVE_MAINTENENCE = 2
        NAME = {
            0 : 'COUNTINUE',
            1 : 'REPAIR_PREDICTIVE_NAINTENENCE',
            2 : 'REPLACE_PREDICTIVE_MAINTENENCE',
            3 : 'BATTRY_CHARGE'
        }
    class Battry:
        CONTINUE = 0
        CHARGE = 1

class Action:
    CONTINUE = 0
    CORRECTIVE_MAINTENENCE = 1
    PD_REPAIR = 2
    PD_REPLACE= 3

    class Battry:
        CONTINUE = 0
        CORRECTIVE_MAINTENENCE = 1
        CHARGE = 2
    
    class Preventive:
        CONTINUE = 0
        PREVENTIVE_MAINTENENCE = 1
        
        
class MaintenenceType:
    CORRECTIVE    = 0
    PREVENTIVE    = 1
    PD_REPAIR     = 2
    PD_REPLACE    = 3
    PRODUCTION_MACHINES_MAINTENENCE = 4
    BATTRY_CHARGE = 5
    STANDBY = 6
    NAME          = {
        0 : 'CORRECTIVE',
        1 : 'PREVENTIVE',
        2 : 'PD_REPAIR',
        3 : 'PD_REPLACE',
        4 : 'PRODUCTION_MACHINES_MAINTENENCE',
        5 : 'BATTRY_CHARGE',
        6 : 'STANDBY'
    }
class Events:

    '''
    stopped events are not useful
    maybe delete?
    '''

    TASK = 0
    MAINTENENCE = 1
    STANDBY_IN_DEPOT = 2
    MOVING_DEPOT_TO_SITENET       = 3
    MOIVING_DEPOT_TO_MAINTENENCE  = 4
    MOVING_SITENET_TO_MAINTENENCE = 5
    MOVING_SITENET_TO_DEPOT       = 6
    MOVING_MAINTENENCE_TO_DEPOT   = 7
    MOVING_MAINTENENCE_TO_SITENET = 8
    STANDBY_IN_SITENET = 9
    STANDBY_IN_MAINTENENCE        = 10
    ACTIVE_EVENTS = [0,3,4,5,6,7,8]
    STANDBY_EVENTS = [2,9,10]
    NAME = {
         0 : 'TASK' ,
         1 : 'MAINTENENCE' ,
         2 : 'STANDBY_IN_DEPOT' ,
         3 : 'MOVING_DEPOT_TO_SITENET'       ,
         4 : 'MOVING_DEPOT_TO_MAINTENENCE'  ,
         5 : 'MOVING_SITENET_TO_MAINTENENCE' ,
         6 : 'MOVING_SITENET_TO_DEPOT'       ,
         7 : 'MOVING_MAINTENENCE_TO_DEPOT'   ,
         8 : 'MOVING_MAINTENENCE_TO_SITENET' ,
         9 : 'STANDBY_IN_SITENET' ,
         10: 'STANDBY_IN_MAINTENENCE'

    }



class TimeFrame:
    HOUR      = 3600000
    MINUTE    = 60000
    SECOND    = 1000
    DAY       = 86400000
    DECI_SEC  = 100
    CENTI_SEC = 10
    MILI_SEC  = 1 
    YEAR      = 365 * 86400000
    MONTH     = 31 * 86400000

class ConstRewards:
    BIG_PUNISH = -1 * 10**9
    INVALID_ACTION = -1 * 10**6
    FULL_BUFFER    = -1 * 10**4

class Strategy:
    CORRECTIVE = 0
    PREVENTIVE = 1
    PREDICTIVE = 2
    NAME = {
    0 : 'CORRECTIVE',
    1 : 'PREVENTIVE',
    2 : 'PREDICTIVE',
    }

class LocationCase:
    def __init__(self, location_case):
        self.location_case = location_case
    
    net_depot_edge = 0
    net_maintenence_edge = 1
    depot_maintenence_edge = 2

    DepotNet_Maintenence  = 0
    DepotNetMaintenence   = 1
    Depot_Net_Maintenence = 2
    DIAMETER = 3
    distance = {
        0 : [10 * TimeFrame.MINUTE , 30 * TimeFrame.MINUTE, 30 * TimeFrame.MINUTE , 29.5 * TimeFrame.MINUTE ],
        1 : [30 * TimeFrame.MINUTE , 30 * TimeFrame.MINUTE, 30 * TimeFrame.MINUTE , 26   * TimeFrame.MINUTE ],
        2 : [10 * TimeFrame.MINUTE , 10 * TimeFrame.MINUTE, 10 * TimeFrame.MINUTE , 8.5  * TimeFrame.MINUTE ],
    }

    def get_edge_distance(self , edge):
        return LocationCase.distance[self.location_case][edge]
