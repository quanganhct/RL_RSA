from enum import Enum

'''
Return code of service. Indicate whether a service is successful privisioned or failed 
'''
class FailedCode(Enum):
    SUCCESS = 0
    PATH = 1
    MODULATION = 2
    FREQ_SLOT = 3
    OSNR = 4
    PREV_OSNR = 5