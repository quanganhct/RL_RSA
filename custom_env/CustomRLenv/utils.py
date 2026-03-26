import networkx as nx
import numpy as np
import math 
import re

from itertools import islice
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union


@dataclass
class Modulation:
    name: str
    # maximum length in km
    maximum_length: Union[int, float]
    # number of bits per Hz per sec.
    spectral_efficiency: int
    # minimum OSNR that allows it to work
    minimum_osnr: Optional[float] = field(default=None)
    # maximum in-band cross-talk
    inband_xt: Optional[float] = field(default=None)


@dataclass
class Path:
    path_id: int
    node_list: Tuple[str]
    hops: int
    length: Union[int, float]
    best_modulation: Optional[Modulation] = field(default=None)
    current_modulation: Optional[Modulation] = field(default=None)


@dataclass(repr=False)
class Service:
    service_id: int
    source: str
    source_id: int
    destination: Optional[str] = field(default=None)
    destination_id: Optional[str] = field(default=None)
    arrival_time: Optional[float] = field(default=None)
    holding_time: Optional[float] = field(default=None)
    bit_rate: Optional[float] = field(default=None)
    path: Optional[Path] = field(default=None)
    best_modulation: Optional[Modulation] = field(default=None)
    service_class: Optional[int] = field(default=None)
    number_slots: Optional[int] = field(default=None)
    core: Optional[int] = field(default=None)
    launch_power: Optional[float] = field(default=None)
    accepted: bool = field(default=False)

    def __str__(self):
        msg = "{"
        msg += "" if self.bit_rate is None else f"br: {self.bit_rate}, "
        msg += "" if self.service_class is None else f"cl: {self.service_class}, "
        return f"Serv. {self.service_id} ({self.source} -> {self.destination})" + msg

modulations = (
    # the first (lowest efficiency) modulation format needs to have maximum length
    # greater or equal to the longest path in the topology.
    # Here we put 100,000 km to be on the safe side
    Modulation(
        name="BPSK",
        maximum_length=100_000,
        spectral_efficiency=1,
        minimum_osnr=12.6,
        inband_xt=-14,
    ),
    Modulation(
        name="QPSK",
        maximum_length=2_000,
        spectral_efficiency=2,
        minimum_osnr=12.6,
        inband_xt=-17,
    ),
    Modulation(
        name="8QAM",
        maximum_length=1_000,
        spectral_efficiency=3,
        minimum_osnr=18.6,
        inband_xt=-20,
    ),
    Modulation(
        name="16QAM",
        maximum_length=500,
        spectral_efficiency=4,
        minimum_osnr=22.4,
        inband_xt=-23,
    ),
    Modulation(
        name="32QAM",
        maximum_length=250,
        spectral_efficiency=5,
        minimum_osnr=26.4,
        inband_xt=-26,
    ),
    Modulation(
        name="64QAM",
        maximum_length=125,
        spectral_efficiency=6,
        minimum_osnr=30.4,
        inband_xt=-29,
    ),
)

def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d


def read_txt_file(file, undirected_file=True):
    graph = nx.DiGraph()
    num_nodes = 0
    num_links = 0
    id_link = 0
    with open(file, "r") as lines:
        # gets only lines that do not start with the # character
        nodes_lines = [value for value in lines if not value.startswith("#")]
        for idx, line in enumerate(nodes_lines):
            if idx == 0:
                num_nodes = int(line)
                for id in range(1, num_nodes + 1):
                    graph.add_node(str(id), name=str(id))
            elif idx == 1:
                num_links = int(line)
            elif len(line) > 1:
                info = line.replace("\n", "").split(" ")
                graph.add_edge(
                    info[0],
                    info[1],
                    id=id_link,
                    index=id_link,
                    weight=1,
                    length=int(info[2]),
                )
                id_link += 1

                if undirected_file:
                    graph.add_edge(
                        info[1],
                        info[0],
                        id=id_link,
                        index=id_link,
                        weight=1,
                        length=int(info[2]),
                    )
                    id_link += 1

    return graph

# File from sndlib, refer to data/germany/germany.txt for file format
@dataclass
class Node:
    id: int
    name: str
    long: float
    lat: float

'''
Read file following snd format. See ./data/germany/sndlib_germany.txt for the reference 
how the format looks like
'''
def read_sndlib_txt_file(file):
    node_pattern = r'(.+?)\s*\(\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*\)'
    link_pattern = r'\(\s*(.+?)\s+(.+?)\s*\)'

    read_node_flag = True
    node_id = 1
    link_id = 0
    map_node_name = {}

    graph = nx.DiGraph()

    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Nodes:"):
                read_node_flag = True
                continue
            
            if line.startswith("Links:"):
                read_node_flag = False
                continue

            if read_node_flag:
                matches = re.findall(node_pattern, line)
                for name, long, lat in matches:
                    node = Node(node_id, name, float(long), float(lat))
                    map_node_name[name] = node
                graph.add_node(str(node_id), name=str(node_id))
                node_id += 1
            else:
                match = re.search(link_pattern, line)
                city1 = match.group(1).strip()
                city2 = match.group(2).strip()
                long1, lat1 = map_node_name[city1].long, map_node_name[city1].lat
                long2, lat2 = map_node_name[city2].long, map_node_name[city2].lat
                d = distance((lat1, long1), (lat2, long2))
                graph.add_edge(str(map_node_name[city1].id),
                    str(map_node_name[city2].id),
                    id=link_id,
                    index=link_id,
                    weight=1,
                    length=int(d))
                link_id += 1

                graph.add_edge(str(map_node_name[city2].id),
                    str(map_node_name[city1].id),
                    id=link_id,
                    index=link_id,
                    weight=1,
                    length=int(d))
                link_id += 1
    return graph

def get_precomputed_path(G, source, target, k=5, alpha=2, weight='weight'):
    if k is None:
        p = nx.shortest_path_length(G, source, target)
        print(f"Compute precomputed path: ({source}, {target}, {p+alpha})")
        return list(nx.all_simple_paths(G, source, target, cutoff=p+alpha))
    else:
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

def get_path_weight(graph, path, weight="length"):
    return np.sum([graph[path[i]][path[i + 1]][weight] for i in range(len(path) - 1)])

def get_best_modulation_format(
    length: float, modulations: Sequence[Modulation]
) -> Modulation:
    # sorts modulation from the most to the least spectrally efficient
    sorted_modulations = sorted(
        modulations, key=lambda x: x.spectral_efficiency, reverse=True
    )
    for i in range(len(modulations)):
        if length <= sorted_modulations[i].maximum_length:
            return sorted_modulations[i]
    raise ValueError(
        "It was not possible to find a suitable MF for a path with {} km".format(length)
    )

'''
@undirected_file: there are file with undirected link and directed link. Check data file before use
@sndformat: Check if the data file is following snd format. See ./data/germany/sndlib_germany.txt 
for the reference how the format looks like
@alpha: with n=length of shortest path, precompute all paths with length <= n+alpha
@k_paths: compute precomputed paths as k-shortest path. If it is None, then return alpha precomputed path
'''
def get_topology(file_name, topology_name, k_paths=5, undirected_file=True, sndformat=False, alpha=2):
    global modulations

    k_shortest_paths = {}
    if not sndformat:
        topology = read_txt_file(file_name, undirected_file=undirected_file)
    else:
        topology = read_sndlib_txt_file(file_name)
    idp = 0
    num_path = 0
    for idn1, n1 in enumerate(topology.nodes()):
        for idn2, n2 in enumerate(topology.nodes()):
            if idn1 < idn2:
                paths = get_precomputed_path(topology, n1, n2, k=k_paths, alpha=alpha, weight='length')
                # print(n1, n2, len(paths))
                num_path = max(len(paths), num_path)
                lengths = [
                    get_path_weight(topology, path, weight="length") for path in paths
                ]
                
                selected_modulations = [
                    get_best_modulation_format(length, modulations)
                    for length in lengths
                ]
                
                objs = []

                for path, length, modulation in zip(
                    paths, lengths, selected_modulations
                ):
                    objs.append(
                        Path(
                            path_id=idp,
                            node_list=path,
                            hops=len(path) - 1,
                            length=length,
                            best_modulation=modulation,
                        )
                    )  # <== The topology is created and a best modulation is just automatically attached.  In our new implementation, the best modulation will be variable depending on available resources and the amount of crosstalk it will cause.
                    print("\t", objs[-1])
                    idp += 1
                k_shortest_paths[n1, n2] = objs
                k_shortest_paths[n2, n1] = objs
    topology.graph["name"] = topology_name
    topology.graph["ksp"] = k_shortest_paths
    topology.graph["max_numpath"] = num_path
    if modulations is not None:
        topology.graph["modulations"] = modulations
    topology.graph["k_paths"] = k_paths
    topology.graph["node_indices"] = []
    for idx, node in enumerate(topology.nodes()):
        topology.graph["node_indices"].append(node)
        topology.nodes[node]["index"] = idx
    return topology