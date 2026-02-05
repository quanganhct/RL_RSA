
from itertools import islice
from enum import Enum
import networkx as nx
import math
import numpy as np
from env import constant


class Link:
    pass

class Network:
    pass

class Modulation(Enum):
    BPSK = 1
    QPSK = 2
    QAM8 = 3
    QAM16 = 4

def get_eligible_modulation(reach):
    if reach > 2500:
        return [Modulation.BPSK]
    elif reach > 1250:
        return [Modulation.BPSK, Modulation.QPSK]
    elif reach > 625:
        return [Modulation.BPSK, Modulation.QPSK, Modulation.QAM8]
    else:
        return [Modulation.BPSK, Modulation.QPSK, Modulation.QAM8, Modulation.QAM16]

class Request:
    static_id = 0
    '''
    @delay_limit: request can delay until start_time + delay_limit, after this period, 
    if not granted, request will be denied
    @data_rate: data rate in Gbps
    '''
    def __init__(self, start: int, end: int, data_rate, start_time, holding_time, delay_limit, nb_slot = None):
        self.start = start
        self.end = end
        self.data_rate = data_rate
        self.start_time = start_time
        self.holding_time = holding_time
        self.nb_slot = nb_slot
        self.delay_limit = delay_limit
        self.granted = False
        self.id = Request.static_id
        Request.static_id += 1
        
    def __str__(self):
        return "Start: {}, End: {}, Request: {}".format(self.start, self.end, self.data_rate)

class Node:
    def __init__(self, _id):
        self.id = _id

        '''
            @self.out_link = {dest_node_id: Link}
            @self.in_link = {source_node_id: Link}
        '''
        self.out_link = {}
        self.in_link = {}

    def add_out_link(self, link: Link):
        self.out_link[link.destination.id] = link
           
    def add_in_link(self, link: Link):
        self.in_link[link.source.id] = link

class Link:
    static_id = 0
    def __init__(self, source: Node, destination: Node, link_length=200):
        self.source = source
        self.destination = destination
        self.id = Link.static_id
        self.link_length = link_length
        Link.static_id += 1

class Path:
    static_id = 0
    '''
    @node_ids = [node1_id, node2_id, ...]
    @link_ids = [link1_id, link2_id, ...]
    '''
    def __init__(self, list_nodes_id, list_link_id):
        self.id = Path.static_id
        Path.static_id += 1
        self.node_ids = list_nodes_id
        self.link_ids = list_link_id

class NonEligibleLightpath(Exception):
    pass

class NonEligibleModulation(Exception):
    pass

class LightpathNotRouted(Exception):
    pass

class ConflictSlot(Exception):
    pass

class Lightpath:
    static_id = 0

    def __init__(self, request: Request, path: Path, network: Network):
        if request.start != path.node_ids[0] or request.end != path.node_ids[-1]:
            raise NonEligibleLightpath("Source and destination of request and path don't match!")
        self.id = Lightpath.static_id
        Lightpath.static_id += 1
        self.request = request
        self.path = path
        self.start_time = None
        self.end_time = None
        self.starting_slot = None
        self.eligible_modulation = self.compute_modulation(network)
        self.nb_slot = None
        self.modulation = None
        self.snr_threshold = None

    '''
    Return all eligible modulation and number of slots needed
    @return: [(modulation1, nb_slot), ...]
    '''
    def compute_modulation(self, network: Network):
        reach = sum([network.map_link[link_id].link_length for link_id in self.path.link_ids])
        modulations = get_eligible_modulation(reach)
        data_rate = self.request.data_rate
        result = dict([(modulation, int(math.ceil(data_rate/(modulation.value * 12.5))) + 1) \
                  for modulation in modulations])
        return result

    def set_modulation(self, modulation: Modulation):
        if modulation not in self.eligible_modulation:
            raise NonEligibleModulation("Non eligible modulation. Check @eligible_modulation for proper modulation format")
        
        self.modulation = modulation
        self.nb_slot = self.eligible_modulation[modulation]
        # actual bandwidth used is nb_slot - 1, we should remove 1 from guardband
        self.snr_threshold = 2**(self.request.data_rate/((self.nb_slot-1) * 12.5))

    def set_start_time(self, start_time):
        self.start_time = start_time
        self.end_time = start_time + self.request.holding_time

class Network(nx.DiGraph):
    K = 4
    def __init__(self, nb_slot=380):
        '''
        @map_link_slot = {
            link_id: {
                slot: lightpath_id / None
            }
        }

        @map_node = {
            node_id: Node
        }

        @map_link = {
            link_id: Link
        }

        @map_request = {
            request_id: Request
        }

        @map_path = {
            path_id: Path
        }

        @shortest_path_pool = {
            (start_id, destination_id): [Path1, Path2]
        }
        '''
        self.nb_slot = nb_slot
        self.map_link_slot = {}
        self.clock = 0
        self.map_node: dict[int, Node] = {}
        self.map_link: dict[int, Link] = {}
        self.map_request: dict[int, Request] = {}
        self.map_path: dict[int, Path] = {}
        self.shortest_path_pool: dict[(int, int), list[Path]] = {}
        self.sorted_request_list: list[Request] = []
        self.waiting_request: list[Request] = []

    def reset_id():
        Request.static_id = 0
        Link.static_id = 0
        Path.static_id = 0
        Lightpath.static_id = 0

    def add_node(self, node_id):
        super().add_node(node_id)
        node = Node(node_id)
        self.map_node[node.id] = node

    def add_link(self, source_id, dest_id, link_length=200):
        source: Node = self.map_node[source_id]
        destination: Node = self.map_node[dest_id]
        
        link = Link(source, destination, link_length)
        self.map_link[link.id] = link
        source.add_out_link(link)
        destination.add_in_link(link)

        super().add_edge(source_id, dest_id, weight=link.link_length, weight_2=1, link_id=link.id)

    def get_link(self, source_id, dest_id):
        return self.map_node[source_id].out_link[dest_id]
    
    def add_request(self, source_id, dest_id, data_rate, start_time, holding_time):
        req = Request(source_id, dest_id, data_rate, start_time, holding_time)
        self.map_request[req.id] = req

    def get_k_shortest_path(self, source_id, dest_id):
        key = (source_id, dest_id)
        if key in self.shortest_path_pool and not self.shortest_path_pool[key]:
            return self.shortest_path_pool[key]
        
        self.shortest_path_pool[key] = []
        paths = list(islice(nx.shortest_simple_paths(self, source_id, \
                                dest_id, weight='weight_2'), Network.K))
        
        for list_node_ids in paths:
            list_link_ids = [self.get_link(list_node_ids[i], list_node_ids[i+1]).id for i in range(0, len(list_node_ids)-1)]
            pobj = Path(list_node_ids, list_link_ids)
            self.shortest_path_pool[key].append(pobj)
            self.map_path[pobj.id] = pobj

        return self.shortest_path_pool[key]
    
    def initialize_network(self):
        for link_id in self.map_link:
            self.map_link_slot[link_id] = dict([(i, None) for i in range(self.nb_slot)])

    def reset_network(self):
        self.clock = 0
        for link_id in self.map_link_slot:
            for slot in self.map_link_slot[link_id]:
                self.map_link_slot[link_id][slot] = None

    def remove_lightpath(self, lightpath: Lightpath):
        if lightpath.starting_slot is None:
            raise LightpathNotRouted("Lightpath %s is not routed"%(lightpath.id))
        
        for link_id in lightpath.path.link_ids:
            for slot in range(lightpath.starting_slot, lightpath.starting_slot+lightpath.nb_slot):
                self.map_link_slot[link_id][slot] = None

    def route_lightpath(self, lightpath: Lightpath, starting_slot: int):
        if lightpath.modulation is None:
            raise Exception("Choose modulation first")
        
        for link_id in lightpath.path.link_ids:
            for slot in range(starting_slot, starting_slot + lightpath.nb_slot):
                if self.map_link_slot[link_id][slot] is not None:
                    raise ConflictSlot("Link %s slot %s occupied by %s, cannot route %s"\
                        %(link_id, slot, self.map_link_slot[link_id][slot], lightpath.id))
        
        lightpath.starting_slot = starting_slot
        lightpath.set_start_time(self.clock)
        for link_id in lightpath.path.link_ids:
            for slot in range(starting_slot, starting_slot + lightpath.nb_slot):
                self.map_link_slot[link_id][slot] = lightpath.id
        lightpath.request.granted = True
        self.waiting_request.remove(lightpath.request)

    def sort_all_request(self):
        self.sorted_request_list: list[Request] = list(self.map_request.values())
        self.sorted_request_list.sort(key=lambda x: x.start_time)


    '''
    Increment time clock, terminate outdated request.
    @return requests that are waiting to be granted
    '''
    def time_tick(self):
        if self.clock == 0:
            self.sort_all_request()
        
        for request in self.sorted_request_list:
            if request.start_time == self.clock:
                self.waiting_request.append(self.sorted_request_list.pop(0))
            else:
                break

        for request in self.waiting_request:
            if request.start_time + request.delay_limit < self.clock:
                self.waiting_request.remove(request)

        self.clock += 1
        return self.waiting_request
        
    def compute_SNR(self, lightpath:Lightpath, starting_slot: int):
        nb_spans = sum([math.ceil(self.map_link[id].link_length/constant.fiber_span) for id in lightpath.path.link_ids])
        main_center = (starting_slot + lightpath.nb_slot)/2
        fcenter = constant.f0_c + main_center * constant.ref_bw
        G_ASE = nb_spans * constant.n_sp * np.e**(2 * constant.alpha * constant.fiber_span) \
            * constant.h * fcenter

        
