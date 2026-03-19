import networkx as nx
import numpy as np

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


def read_txt_file(file):
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

def get_precomputed_path(G, source, target, k, weight=None):
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

def get_topology(file_name, topology_name, k_paths=5):
    global modulations

    k_shortest_paths = {}
    if file_name.endswith(".txt"):
        topology = read_txt_file(file_name)
    else:
        raise ValueError("Supplied topology is unknown")
    idp = 0
    for idn1, n1 in enumerate(topology.nodes()):
        for idn2, n2 in enumerate(topology.nodes()):
            if idn1 < idn2:
                paths = get_precomputed_path(topology, n1, n2, k_paths, weight="length")
                print(n1, n2, len(paths))
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
    if modulations is not None:
        topology.graph["modulations"] = modulations
    topology.graph["k_paths"] = k_paths
    topology.graph["node_indices"] = []
    for idx, node in enumerate(topology.nodes()):
        topology.graph["node_indices"].append(node)
        topology.nodes[node]["index"] = idx
    return topology