import gymnasium as gym


class RSAEnv(gym.Env):
    def __init__(self):
        super().__init__()

        ''' @routing_dict = {
            link_id: {
                slot: path_id/None
            }
        }
        '''
        self.routing_dict = {}

        
