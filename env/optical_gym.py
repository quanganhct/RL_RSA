import gymnasium as gym
from optical_network import Request, Path, Modulation, Lightpath, Network

class PathSelectionAction:
    def __init__(self, req:Request, path:Path, network:Network) -> None:
        self.lightpath = Lightpath(req, path, network)
        

class ModulationSelectionAction:
    def __init__(self, action1:PathSelectionAction, modulation:Modulation, starting_slot:int) -> None:
        self.prev_action = action1
        self.prev_action.lightpath.set_modulation(modulation)
        self.starting_slot = starting_slot


class RSAEnv(gym.Env):
    def __init__(self, network:Network):
        super().__init__()

        ''' @routing_dict = {
            link_id: {
                slot: path_id/None
            }
        }
        '''
        self.routing_dict = {}
        self.network = network

    

    
