# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 12:06:57 2026

@author: Momo
"""

import numpy as np
import networkx as nx
from custom_env.optical_rl_gym.envs.rmsa_env import RMSAEnv
from custom_env.CustomRLenv import utils
import math
import torch
from typing import List

from custom_env.CustomRLenv.utils import Path, Modulation, Service, spectrum_feature_points, transform_graph
from env import constant
from custom_env.CustomRLenv.osnr import compute_ase_nli, compute_min_gap_osnr

class CustomRMSAEnv(RMSAEnv):
    """
    Custom RMSA Environment with:
    - Node features: degree, betweenness, traffic load estimate
    - Edge features: spectrum occupancy, length, OSNR, NLI, ASE, fragmentation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Precompute static graph properties (degree, betweenness)
        # self.graph = self.topology#.to_networkx()
        self.node_degree = dict(self.topology.degree())
        self.num_nodes = self.topology.number_of_nodes()
        self.node_betweenness = nx.betweenness_centrality(self.topology)
        self._update_candidate_paths()
        self.max_path_len = self.topology.graph['max_hop']
        # In case of alpha shortest path, @max_num_path is the maximum path per pair nodes, and it is 
        # k_paths in case of k shortest path
        self.max_num_path = self.topology.graph["max_numpath"]

        #Use the below graph to create GNN
        #TODO: the betweenness centrality of the transformed graph should be related to the old one ?
        self.transformed_graph = transform_graph(self.topology)
        self.tgraph_node_degree = dict(self.transformed_graph.degree())
        self.tgraph_node_betweenness = nx.betweenness_centrality(self.transformed_graph)

        self.list_modulations = list(self.topology.graph["modulations"])
        self.launch_power = 10 ** ((constant.launch_power_dbm - 30) / 10)
        self.granularities = self.compute_granularity()

        self.generated_req_lifetime = []
        
        self.edge_index =  [tuple(map(int, t)) for t in self.topology.edges()]
        
        self.edge_id = {e:i for i, e in enumerate(self.edge_index)}
        
       

    def compute_granularity(self):
        granularity = []
        for bit_rate in constant.bit_rates:
            for m in utils.modulations:
                nbslot = math.ceil(bit_rate/(m.spectral_efficiency * 12.5))
                granularity.append(nbslot)
        return set(granularity)

    # @p: power in dbm
    def set_launch_power(self, p):
        self.launch_power = 10 ** ((p - 30) / 10)

    # -----------------------------
    # Node feature vector x_v
    # -----------------------------
    def get_node_features(self):
        features = []
        for v in self.topology.nodes():
            degree = self.node_degree[v]
            betweenness = self.node_betweenness[v]
            # Traffic load estimate: e.g., number of ongoing connections at node
            # traffic_load = self.estimate_node_load(v)
            # features.append([degree, betweenness, traffic_load])
            features.append([degree, betweenness])
        return np.array(features, dtype=np.float32)

    def estimate_node_load(self, node_id):
        # Simple example: sum of active connections passing through node
        load = 0
        for link_id, link in enumerate(self.topology.edges()):
            # print("node_id = ", type(node_id))
            # print("link = ", link)
            # print("link_id = ", link_id)
            if node_id in link:
                load += len(link.active_requests)
        return load
    
    def _path_to_edges(self, path):
        return list(zip(path[:-1], path[1:]))
    
    def _update_candidate_paths(self):
        
        self.candidate_paths = {}
        self.candidate_paths_length = {}
        self.candidate_paths_hops = {}
        self.candidate_paths_minimum_osnr = {}
        self.candidate_paths_inband_xt = {}
        self.candidate_paths_impairment = {}

        
        for src_dst, paths in self.k_shortest_paths.items():
            src = src_dst[0]
            dst = src_dst[1]
            candidate_p = []
            candidate_p_length = []
            candidate_p_hops = []
            candidate_p_minimum_osnr = []
            candidate_p_inband_xt = []
            candidate_p_impairment = []
            num_path = len(self.topology.graph['ksp'][(src, dst)])
            for i in range(num_path):
                pobj: Path = self.k_shortest_paths[(src, dst)][i]
                path = pobj.node_list
                # path = self.k_shortest_paths[(src, dst)][i].node_list
                path = self._path_to_edges(path)
                path_link_id = []
                for j , p in enumerate(path):
                    link_id = self.topology.edges()[p]['id']
                    path_link_id.append(link_id)
                    
                candidate_p.append(path_link_id)
                candidate_p_length.append(self.topology.graph['ksp'][(src, dst)][i].length)
                candidate_p_hops.append(self.topology.graph['ksp'][(src, dst)][i].hops)
                candidate_p_minimum_osnr.append(self.topology.graph['ksp'][(src, dst)][i].best_modulation.minimum_osnr)
                candidate_p_inband_xt.append(self.topology.graph['ksp'][(src, dst)][i].best_modulation.inband_xt)
                
                candidate_p_impairment.append([candidate_p_length[-1], 
                                               candidate_p_inband_xt[-1], 
                                               candidate_p_minimum_osnr[-1]])
           
                #make sure you use the node id and link id properly
        
            self.candidate_paths[(src, dst)] = candidate_p
            self.candidate_paths_length[(src, dst)] = candidate_p_length
            self.candidate_paths_hops[(src, dst)] = candidate_p_hops
            self.candidate_paths_minimum_osnr[(src, dst)] = candidate_p_minimum_osnr
            self.candidate_paths_inband_xt[(src, dst)] = candidate_p_inband_xt
            self.candidate_paths_impairment[(src, dst)] = candidate_p_impairment
     
    
    #TODO: 1. add minimum gap osnr - threshold of the services that shared link with the path
    def get_candidate_path_features(self, path:Path):
        s: Service
        min_gap = 5
        shared_services = self.get_running_service_share_links(path)
        running_services = self.topology.graph["running_services"]
        set_running_service_idx = set([s.service_id for s in running_services])
        if len(shared_services) == 0:
            return min_gap

        for s in self.topology.graph["running_services"]:
            if s.service_id not in shared_services:
                continue

            power_nli = sum([s.nli_inf_from[sid] if sid in set_running_service_idx else 0 \
                             for sid in s.nli_inf_from.keys()])
            nli = power_nli / s.launch_power
            ase = s.ase_inf / s.launch_power
            osnr = nli + ase

            osnr = 10 * np.log10(1 / osnr)
            gap = osnr - s.path.current_modulation.minimum_osnr
            min_gap = min(min_gap, gap)

        return min_gap
    
    def get_candidate_paths(self):
        if not hasattr(self, "current_service"):
            return [np.zeros(1) for i in range(self.max_num_path)]
        
        
        src = self.current_service.source
        dst = self.current_service.destination
        candidates = self.candidate_paths[(src, dst)] 
        
        
        path_features = []
        pad_paths = []
        
        for path_idx, path in enumerate(self.k_shortest_paths[src, dst]):
            # print(f"path = {path}")
            path_feat = self.get_candidate_path_features(path)
            path_features.append(path_feat)
            
            pad_path = candidates[path_idx] + [-1] * (self.max_path_len - len(candidates[path_idx]))
            
            pad_paths.append(pad_path)
        
        return np.array(pad_paths), np.array(path_features)
    
    def get_candidate_paths_length(self):
        if not hasattr(self, "current_service"):
            return  [0 for i in range(self.max_num_path)]

        src = self.current_service.source
        dst = self.current_service.destination
        
        return self.candidate_paths_length[(src, dst)] 
    
    def get_candidate_paths_hops(self):
        if not hasattr(self, "current_service"):
            return [1 for i in range(self.max_num_path)]

        src = self.current_service.source
        dst = self.current_service.destination
        
        return self.candidate_paths_hops[(src, dst)]
    
    def get_candidate_paths_minimum_osnr(self):
        if not hasattr(self, "current_service"):
            return [0 for i in range(self.max_num_path)]

        src = self.current_service.source
        dst = self.current_service.destination
        
        return self.candidate_paths_minimum_osnr[(src, dst)]
    
    def get_candidate_paths_inband_xt(self):
        if not hasattr(self, "current_service"):
            return [0 for i in range(self.max_num_path)]

        src = self.current_service.source
        dst = self.current_service.destination
        
        return self.candidate_paths_inband_xt[(src, dst)]
    
    def get_candidate_paths_impairment(self):
        if not hasattr(self, "current_service"):
            return [[0,0,0] for i in range(self.max_num_path)]

        src = self.current_service.source
        dst = self.current_service.destination
        
        return self.candidate_paths_impairment[(src, dst)]
    
    # -----------------------------
    # Compute demand embedding for RL state
    # -----------------------------
    def compute_demand_embedding(self):
        """
        Returns a vector representing the current request:
        - source/destination one-hot
        - requested bitrate normalized
        """
        if not hasattr(self, "current_service"):
            return np.zeros(self.num_nodes * 2 + 1)

        src = self.current_service.source
        dst = self.current_service.destination
        bitrate = self.current_service.bit_rate

        # One-hot for source and destination
        src_onehot = np.zeros(self.num_nodes)
        dst_onehot = np.zeros(self.num_nodes)
        # -1 because node are [1..num_node]
        # But index should be [0..num_nodes-1]
        src_onehot[int(src)-1] = 1
        dst_onehot[int(dst)-1] = 1

        # Normalize bitrate by max_bitrate
        bitrate_norm = np.array([bitrate / getattr(self, "max_bitrate", 100)])

        return np.concatenate([src_onehot, dst_onehot, bitrate_norm])
    
    def compute_min_gap_snr_on_link(self, link):
        s:Service

        min_gap = 100
        list_running_service = self.topology.graph["running_services"]
        set_running_service_idx = set([s.service_id for s in list_running_service])

        running_service_on_link = self.topology.edges[link]["running_services"]

        for s in running_service_on_link:
            power_nli = sum([s.nli_inf_from[sid] if sid in set_running_service_idx else 0 \
                             for sid in s.nli_inf_from.keys()])
            nli = power_nli / s.launch_power
            ase = s.ase_inf / s.launch_power
            osnr = nli + ase

            osnr = 10 * np.log10(1 / osnr)
            gap = osnr - s.path.current_modulation.minimum_osnr
            min_gap = min(min_gap, gap)
        
        return min_gap

    # -----------------------------
    # Edge feature vector x_e
    # -----------------------------
    def get_edge_features(self):
        if  self.current_service is None:
            num_feature = 4 * self.topology.number_of_edges()
            return np.zeros([self.topology.number_of_edges(), num_feature])
        
        features = []
        for link_id, link in enumerate(self.topology.edges()):
            # Availability vector: 1=free, 0=occupied
            s_e = self.topology.graph["available_slots"][link_id]
            
            first_slot, val, length = CustomRMSAEnv.rle(s_e)
            max_length_available_block = max(length)

            # Physical length
            l_e = self.topology.edges[link]['length']  # km or meters
            
            # link utilization 
            utilization = self.topology.edges[link]['utilization']  # 
            
            # num running services on link
            running_services = len(self.topology.edges[link]['running_services'])  # 
            
            # num running services on link
            external_fragmentation = self.topology.edges[link]['external_fragmentation']  #
            
            # num running services on link
            compactness = self.topology.edges[link]['compactness']  #

            # Impairments
            # osnr = getattr(link, "OSNR", 0.0)
            # nli = getattr(link, "NLI", 0.0)
            # ase = getattr(link, "ASE", 0.0)

            # Compute abp fr formula for each link
            H_e = self.compute_fragmentation_abp_on_link(link_id)
            
            # Compute min(SNR - SNRT) for every link with all req running on that link
            min_snr_gap = self.compute_min_gap_snr_on_link(link)

            # edge_vec = np.concatenate([s_e, [l_e, osnr, nli, ase, H_e]])
            
            #TODO get maximum available slot of each link

            #TODO: add max available slots as feature and remove utilization
            edge_vec = np.array([H_e,
                                  min_snr_gap, 
                                  running_services, 
                                  max_length_available_block])
            
            # edge_vec = np.concatenate([s_e, [l_e, 
            #                                  utilization, 
            #                                  running_services, 
            #                                  external_fragmentation, 
            #                                  compactness,  
            #                                  H_e, min_snr_gap]])
            features.append(edge_vec)
        return np.array(features, dtype=np.float32)

    # Path features for modulation selection
    def get_path_features(self, path_idx):
        
        if  self.current_service is None:
            num_feature = 1 + 1 + 2 * len(self.topology.graph['modulations'])
            return np.zeros(num_feature)
        
        
        src, dest = self.current_service.source, self.current_service.destination
        selected_path: Path = self.k_shortest_paths[src, dest][path_idx]
        path_length = selected_path.length
        nb_hops = selected_path.hops
        frag_score = np.zeros(len(utils.modulations))
        nbslot_array = np.zeros(len(utils.modulations))

        for i in range(len(utils.modulations)):
            mod:Modulation = utils.modulations[i]
            print(self.current_service)
            if mod.spectral_efficiency > selected_path.best_modulation.spectral_efficiency:
                continue

            nb_slot = self.get_number_slots_given_modulation(mod)
            nbslot_array[i] = nb_slot

            # Compute abp frag of the edges on the links of @path_idx for each modulation
            abp = 0
            for j in range(len(selected_path.node_list)-1):
                link_id = self.topology[selected_path.node_list[j]][selected_path.node_list[j+1]]["id"]
                abp += self.compute_fragmentation_abp_on_link_for_path_modulation(link_id, mod)
            abp = abp / (len(selected_path.node_list)-1)
            frag_score[i] = abp
        
        path_vec = np.concatenate([[path_length, nb_hops],
                                  nbslot_array,
                                  frag_score])
        return path_vec

    # Modulation features for slot selection
    def get_modulation_features(self, path_idx, modulation_idx):
        
        if  self.current_service is None:
            num_feature = 2 * self.num_spectrum_resources 
            return np.zeros(num_feature)
        
        available_slots = super().get_available_slots(
                self.k_shortest_paths[
                    self.current_service.source, self.current_service.destination
                ][path_idx])
        
        selected_path:Path = self.k_shortest_paths[
                        self.current_service.source, self.current_service.destination
                    ][path_idx]
        modulation:Modulation = self.topology.graph['modulations'][modulation_idx]
        slots = self.get_number_slots_given_modulation(modulation)
        fp = spectrum_feature_points(available_slots, slots)
        
        shared_running_services = self.get_running_service_share_links(selected_path)
        #TODO: min gap (SNR - SNRT) for all the service sharing links with @path_idx on a whole spectrum
        min_gap = np.zeros(len(fp))
        for i in range(len(fp)):
            if available_slots[i] == 0 or fp[i] == 0:
                continue

            mgap, sid = compute_min_gap_osnr(self, self.current_service, selected_path, \
                                             modulation, i, shared_running_services)
            min_gap[i] = mgap
        
        mod_vec = np.concatenate([fp, min_gap])
        return mod_vec

    def get_running_service_share_links(self, path:Path) -> List[int]:
        _result = set()
        for i in range(len(path.node_list)-1):
            s = self.topology[path.node_list[i]][path.node_list[i+1]]["running_services"]
            if len(s) > 0:
                set_id = set([service.service_id for service in s])
                _result.update(set_id)
        return list(_result)

    # -----------------------------
    # Example: get impairment for a chosen path
    # -----------------------------
    def get_path_impairment(self, path_idx):
        """
        Compute a simple impairment metric for a path:
        Could be sum of per-link OSNR penalties, nonlinear penalties, etc.
        """
        path_links = self.candidate_paths[path_idx]
        impairment = sum([self.network.links[link].impairment for link in path_links])
        return impairment

    def get_number_slots(self, path: Path) -> int:
        """
        Method that computes the number of spectrum slots necessary to accommodate the service request into the path.
        The method already adds the guardband.
        """
        modulation = path.current_modulation
        if modulation is None:
            modulation = path.best_modulation
        return self.get_number_slots_given_modulation(modulation)
    
    def get_number_slots_given_modulation(self, modulation:Modulation):
        """
        Method that computes the number of spectrum slots necessary to accommodate the service request into the path.
        The method already adds the guardband.
        """
        return (
            math.ceil(
                self.current_service.bit_rate
                / (modulation.spectral_efficiency * self.channel_width)
            )
            + 1
        )
    
    def compute_fragmentation_abp_on_link(self, link_id:int):
        """
        Example: H_e = fraction of free slots that are non-contiguous
        """
        available_slots = self.topology.graph["available_slots"][link_id]
        numerator = 0
        denominator = 0
        first_slot, val, length = CustomRMSAEnv.rle(available_slots)
        for g in self.granularities:
            numerator += np.dot(val, np.floor(length/float(g))).flatten()[0]
            denominator += np.floor(np.dot(val, length).flatten()/float(g))[0]
        return float(numerator) / float(denominator) if denominator != 0 else 0

    def compute_fragmentation_abp_on_link_for_path_modulation(self, link_id:int, mod:Modulation):
        nbslots = self.get_number_slots_given_modulation(mod)
        available_slots = self.topology.graph["available_slots"][link_id]

        first_slot, val, length = CustomRMSAEnv.rle(available_slots)
        numerator = np.dot(val, np.floor(length/float(nbslots))).flatten()[0]
        denominator = np.floor(np.dot(val, length).flatten()/float(nbslots))[0]
        return float(numerator) / float(denominator) if denominator != 0 else 0

    def compute_fragmentation_abp(self):
        abp = 0
        for edge in self.topology.edges:
            eid = self.topology[edge[0]][edge[1]]["id"]
            abp += self.compute_fragmentation_abp_on_link(eid)
        return abp / len(self.topology.edges)

    def reward(self):
        alpha = 0.4
        beta = 0.3
        gamma = 0.3
        s:Service
        
        # Compute minimum gap of OSNR and OSNR_threshold
        min_gap = 100
        list_running_service = self.topology.graph["running_services"]
        set_running_service_idx = set([s.service_id for s in list_running_service])

        for s in list_running_service:
            power_nli = sum([s.nli_inf_from[sid] if sid in set_running_service_idx else 0 \
                             for sid in s.nli_inf_from.keys()])
            nli = power_nli / s.launch_power
            ase = s.ase_inf / s.launch_power
            osnr = nli + ase

            osnr = 10 * np.log10(1 / osnr)
            gap = osnr - s.path.current_modulation.minimum_osnr
            min_gap = min(min_gap, gap)

        # Compute fragmentation rate
        fr_rate = self.compute_fragmentation_abp()


        # Compute normalized bitrate
        bitrate = self.current_service.bit_rate/max(constant.bit_rates)

        reward = alpha * min_gap/5 + beta * fr_rate + gamma * bitrate
        return reward
    
    # -----------------------------
    # Custom step to update features
    # -----------------------------
    def step(self, selected_slot):
        self.selected_slot_id = selected_slot

        self.generated_req_lifetime.append((self.current_service.arrival_time, self.current_service.holding_time))

        src, dest = self.current_service.source, self.current_service.destination
        num_path = len(self.k_shortest_paths[src, dest])
        # registering overall statistics
        self.actions_output[self.selected_path_id, self.selected_slot_id] += 1
        previous_network_compactness = (
            self._get_network_compactness()
        )  # used for compactness difference measure

        # starting the service as rejected
        self.current_service.accepted = False
        if (
            self.selected_path_id < num_path and self.selected_slot_id < self.num_spectrum_resources
        ):  # action is for assigning a path
            selected_path: Path = self.k_shortest_paths[src, dest][self.selected_path_id]
            selected_path.current_modulation = self.list_modulations[self.selected_mod_id]

            # check if the modulation selected from the agent is legit
            if selected_path.current_modulation.spectral_efficiency <= selected_path.best_modulation.spectral_efficiency:

                slots = self.get_number_slots(selected_path)
                self.logger.debug(
                    "{} processing action {} path {} mod {} and initial slot {} for {} slots".format(
                        self.current_service.service_id,
                        [self.selected_path_id, self.selected_mod_id, self.selected_slot_id],
                        self.selected_path_id, self.selected_mod_id, self.selected_slot_id, slots
                    )
                )
                if self.is_path_free(
                    selected_path,
                    self.selected_slot_id,
                    slots,
                ):
                    
                    
                    self.current_service.path = selected_path
                    self.current_service.initial_slot = self.selected_slot_id
                    self.current_service.number_slots = slots
                    self.current_service.center_frequency = constant.frequency_start \
                        + constant.frequency_slot_bandwidth * self.selected_slot_id \
                        + constant.frequency_slot_bandwidth * (slots / 2.0)
                    self.current_service.bandwidth = constant.frequency_slot_bandwidth * slots
                    self.current_service.launch_power = self.launch_power
                    
                    #TODO make this work I am using 0,0,0 to check the code
                    osnr, ase, nli = compute_ase_nli(self, self.current_service)
                    if osnr >= selected_path.current_modulation.minimum_osnr + constant.osnr_margin:

                        self._provision_path(
                            self.k_shortest_paths[src, dest][self.selected_path_id],
                            self.selected_slot_id,
                            slots,
                        )
                        # self.current_service.center_frequency = constant.frequency_start \
                        #     + constant.frequency_slot_bandwidth * initial_slot \
                        #     + constant.frequency_slot_bandwidth * (slots / 2.0)
                        
                        # self.current_service.bandwidth = constant.frequency_slot_bandwidth * slots

                        self.current_service.accepted = True
                        self.actions_taken[self.selected_path_id, self.selected_slot_id] += 1
                        if (
                            self.bit_rate_selection == "discrete"
                        ):  # if discrete bit rate is being used
                            self.slots_provisioned_histogram[
                                slots
                            ] += 1  # populate the histogram of bit rates
                        self._add_release(self.current_service)
                    else:
                        self.current_service.accepted = False

        if not self.current_service.accepted:
            self.actions_taken[self.max_num_path, self.num_spectrum_resources] += 1

        self.topology.graph["services"].append(self.current_service)

        # generating statistics for the episode info
        if self.bit_rate_selection == "discrete":
            blocking_per_bit_rate = {}
            for bit_rate in self.bit_rates:
                if self.bit_rate_requested_histogram[bit_rate] > 0:
                    # computing the blocking rate per bit rate requested in the increasing order of bit rate
                    blocking_per_bit_rate[bit_rate] = (
                        self.bit_rate_requested_histogram[bit_rate]
                        - self.bit_rate_provisioned_histogram[bit_rate]
                    ) / self.bit_rate_requested_histogram[bit_rate]
                else:
                    blocking_per_bit_rate[bit_rate] = 0.0

        cur_network_compactness = (
            self._get_network_compactness()
        )  # measuring compactness after the provisioning

        reward = self.reward()
        info = {
            "service_blocking_rate": (self.services_processed - self.services_accepted)
            / self.services_processed,
            "episode_service_blocking_rate": (
                self.episode_services_processed - self.episode_services_accepted
            )
            / self.episode_services_processed,
            "bit_rate_blocking_rate": (
                self.bit_rate_requested - self.bit_rate_provisioned
            )
            / self.bit_rate_requested,
            "episode_bit_rate_blocking_rate": (
                self.episode_bit_rate_requested - self.episode_bit_rate_provisioned
            )
            / self.episode_bit_rate_requested,
            "network_compactness": cur_network_compactness,
            "network_compactness_difference": previous_network_compactness
            - cur_network_compactness,
            "avg_link_compactness": np.mean(
                [
                    self.topology[lnk[0]][lnk[1]]["compactness"]
                    for lnk in self.topology.edges()
                ]
            ),
            "avg_link_utilization": np.mean(
                [
                    self.topology[lnk[0]][lnk[1]]["utilization"]
                    for lnk in self.topology.edges()
                ]
            ),
        }

        # informing the blocking rate per bit rate
        # sorting by the bit rate to match the previous computation
        if self.bit_rate_selection == "discrete":
            for bit_rate, blocking in blocking_per_bit_rate.items():
                info[f"bit_rate_blocking_{bit_rate}"] = blocking
            info["fairness"] = max(blocking_per_bit_rate.values()) - min(
                blocking_per_bit_rate.values()
            )

        self._new_service = False
        self._next_service()
         

        next_state, reward, done, info = self.observation(), reward, \
                self.episode_services_processed == self.episode_length, info

        # Update node & edge features 
        # print(f'src = {self.current_service.source_id} , dst = {self.current_service.destination_id}')
        # print(f'reward = {reward}')
      
        next_state = self._obs(next_state)
        
        # next_state["node_features"] = self.get_node_features()
        # next_state["edge_features"] = self.get_edge_features()
        # next_state["demand_embedding"] = self.compute_demand_embedding()
        # next_state["candidate_paths"] =  self.get_candidate_paths()
        # next_state["candidate_paths_length"] =  self.get_candidate_paths_length()
        # next_state["candidate_paths_hops"] =  self.get_candidate_paths_hops()
        # next_state["candidate_paths_minimum_osnr"] =  self.get_candidate_paths_minimum_osnr()
        # next_state["candidate_paths_inband_xt"] =  self.get_candidate_paths_inband_xt()
        # next_state["candidate_paths_impairment"] =  self.get_candidate_paths_impairment()

        # Keep demand embedding and impairment
        # next_state["demand_embedding"] = self.compute_demand_embedding()
        
        # if "path" in info:
        #     path_idx = info["path"]
        #     impairment = self.get_path_impairment(path_idx)
        # else:
        #     impairment = 0
        # reward -= 0.1 * impairment
        # info["impairment_penalty"] = impairment
        
        # path_mask, mod_mask, spec_mask = self.get_mask()
        # next_state["masks"] = (path_mask, mod_mask, spec_mask) 
        # next_state["path_spectrum"]  = self.get_path_spectrum()

        # # score of initial slot 
        # next_state["spectrum_initial_slot"] = self.get_spectrum_fr_score()
        
        return next_state, reward, done, info
        
    def get_available_blocks(self, path, modulation=None):
        # get available slots across the whole path
        # 1 if slot is available across all the links
        # zero if not
        
        available_slots = super().get_available_slots(
            self.k_shortest_paths[
                self.current_service.source, self.current_service.destination
            ][path]
        )
        
        if modulation is None:
            # getting the number of slots necessary for this service across this path
            slots = self.get_number_slots(
                self.k_shortest_paths[
                    self.current_service.source, self.current_service.destination
                ][path]
            )
        else:
            # getting the number of slots necessary for this service across this path
            slots = self.get_number_slots_given_modulation(modulation)
            
        # getting the blocks
        initial_indices, values, lengths = RMSAEnv.rle(available_slots)

        # selecting the indices where the block is available, i.e., equals to one
        available_indices = np.where(values == 1)

        # selecting the indices where the block has sufficient slots
        sufficient_indices = np.where(lengths >= slots)

        # getting the intersection, i.e., indices where the slots are available in sufficient quantity
        # and using only the J first indices
        final_indices = np.intersect1d(available_indices, sufficient_indices)

        return initial_indices[final_indices], lengths[final_indices]
    
    # -----------------------------
    # Get the scores for initial slot
    # -----------------------------
    def get_spectrum_fr_score(self):
        dim = self.max_num_path
        spec_mask = np.ones([dim, 
                            len(self.topology.graph['modulations']), 
                            self.num_spectrum_resources])
        
        if not hasattr(self, "current_service"):
            return spec_mask
        
        for path_idx in range(dim):
            available_slots = super().get_available_slots(
                self.k_shortest_paths[
                    self.current_service.source, self.current_service.destination
                ][path_idx])

            for mod_idx in range(len(self.topology.graph['modulations'])):
                modulation:Modulation = self.topology.graph['modulations'][mod_idx]
                slots = self.get_number_slots_given_modulation(modulation)

                fp = spectrum_feature_points(available_slots, slots)
                spec_mask[path_idx][mod_idx] = fp
        
        return spec_mask

    def get_path_spectrum(self):
        path_spectrum = []
        num_path = len(self.k_shortest_paths[
                    self.current_service.source, self.current_service.destination
                ])
        for index in range(num_path):
            # get available block given best
            spectrum = self.get_available_slots(
                self.k_shortest_paths[
                    self.current_service.source, self.current_service.destination
                ][index])
        
            path_spectrum.append(spectrum)
        return path_spectrum
    
    
    # Check the mask for k-path and for alpha shortest path
    # -----------------------------
    # Optional: helper for masks
    # -----------------------------
    def get_paths_mask(self):
        dim = self.max_num_path
        if not hasattr(self, "current_service"):
            path_mask = np.ones(dim)
            
            return  path_mask, 
        
        src = self.current_service.source
        dst = self.current_service.destination
        
        path_mask = np.ones(dim)
        num_paths = len(self.k_shortest_paths[src, dst])
        
        
        # Masking sheme
        for path_idx in range(dim):
            # get available block given best
            if path_idx < num_paths:
                initial_indices, _= self.get_available_blocks(path_idx)
                #if no path with best modultation mask path + mod + spec
                if len(initial_indices) == 0:
                    path_mask[path_idx] = 0
            else:# mask paths that don't exist
                path_mask[path_idx:] = [0] * (dim - num_paths)
                break
                
                
        return path_mask
    
    def get_mods_mask(self, path_idx):
        src = self.current_service.source
        dst = self.current_service.destination
        mod_mask = np.ones(len(self.topology.graph['modulations']))
        # if path availavle with best modulation max infeasible mod + spec
        for modulation_idx, modulation in enumerate(self.topology.graph['modulations']):
            max_length = modulation.maximum_length
            path_length = self.topology.graph['ksp'][(src, dst)][path_idx].length
            
            # if no path with best modultation mask path
            if max_length < path_length:
                mod_mask[modulation_idx] = 0
                
        return mod_mask
      
    def get_slot_mask(self, path_idx, mod_idx):
        spec_mask = np.ones(self.num_spectrum_resources)
        
        # get starting slot with enough available slots
        mod = self.topology.graph['modulations'][mod_idx]
        initial_indices, _ = self.get_available_blocks(path_idx, mod)
        
        spec_mask = np.zeros(self.num_spectrum_resources)
        #if no slot available mask
        if len(initial_indices) != 0:
            # Don't mask indice where we find enough avaliable slots
            spec_mask[initial_indices] = 1
        return spec_mask
          
        
    def get_mask(self):
        dim = self.max_num_path
        if not hasattr(self, "current_service"):
            path_mask = np.ones(dim)
            
            mod_mask = np.ones([dim, 
                                len(self.topology.graph['modulations'])])
            
            spec_mask = np.ones([dim, 
                                len(self.topology.graph['modulations']), 
                                self.num_spectrum_resources])
            return  path_mask, mod_mask, spec_mask
        
        
        src = self.current_service.source
        dst = self.current_service.destination
        
        path_mask = np.ones(dim)
        
        mod_mask = np.ones([dim, 
                            len(self.topology.graph['modulations'])])
        
        spec_mask = np.ones([dim, 
                            len(self.topology.graph['modulations']), 
                            self.num_spectrum_resources])
        # Masking sheme
        for path_idx in range(dim):
            # get available block given best
            initial_indices, _= self.get_available_blocks(path_idx)
            #if no path with best modultation mask path + mod + spec
            if len(initial_indices) == 0:
                path_mask[path_idx] = 0
                mod_mask[path_idx, :] = 0
                spec_mask[path_idx, :, :] = 0
                continue
            
            # if path availavle with best modulation max infeasible mod + spec
            for modulation_idx, modulation in enumerate(self.topology.graph['modulations']):
                max_length = modulation.maximum_length
                path_length = self.topology.graph['ksp'][(src, dst)][path_idx].length
                
                # if no path with best modultation mask path
                if max_length < path_length:
                    mod_mask[path_idx, modulation_idx] = 0
                    spec_mask[path_idx, modulation_idx, :] = 0
                    continue
                
                # get starting slot with enough available slots
                initial_indices, _ = self.get_available_blocks(path_idx, modulation)
                
                #if no slot available mask
                if len(initial_indices) == 0:
                    spec_mask[path_idx, modulation_idx, :] = 0
                    continue
                
                spec_to_mask = np.zeros(self.num_spectrum_resources)
                # Don't mask indice where we find enough avaliable slots
                spec_to_mask[initial_indices] = 1
                spec_mask[path_idx, modulation_idx,:] = spec_to_mask
                
        return path_mask, mod_mask, spec_mask
                

    # ============================================================
    # PATH STEP
    # ============================================================

    def step_path(self, obs, action):

        self.selected_path_id = int(action)

        obs = self._obs(obs)

        return obs, {}  
    
    
    # ============================================================
    # MODULATION STEP
    # ============================================================

    def step_modulation(self, obs, action):

        self.selected_mod_id = int(action)

        obs = self._obs(obs)

        return obs, {}
    
        
    def _obs(self, obs):
        
        
        mod_mask = None
        slot_mask = None
        
        # GRAPH STATE
        edge_features = self.get_edge_features()
        obs["edge_features"] =  torch.tensor(edge_features, 
                                         dtype=torch.float32)
        # the library can be updated so that nodes are directly from 0
        # edge_index = self.edge_index
        obs["edge_index"] = torch.tensor(self.edge_index, dtype=torch.long).T - 1  # [2, num_edges]
        
     
        # PATH CANDIDATES [should be list of list as size might differ]
        candidate_paths, candidate_paths_features =  self.get_candidate_paths()
        
        obs["candidate_paths"] = torch.tensor(candidate_paths, dtype=torch.long)
        obs["candidate_paths_features"] = torch.tensor(candidate_paths_features, 
                                         dtype=torch.float32)
        
        path_mask = self.get_paths_mask()  
        path_mask = torch.tensor(path_mask, dtype=torch.bool)  
        # PATH FEATURES  
        if self.selected_path_id is None:
            obs["path_features"] = None
            obs["mod_features"] = None
            
        else:
            path_features  = self.get_path_features(
                self.selected_path_id)
             
            obs["path_features"]  =  torch.tensor(path_features, 
                                         dtype=torch.float32)
            mod_mask = self.get_mods_mask(self.selected_path_id)
            mod_mask = torch.tensor(mod_mask, dtype=torch.bool)
            
            # Modulation FEATURES 
            if self.selected_mod_id is None:
                obs["mod_features"] = None
            else:
                mod_features =  self.get_modulation_features(self.selected_path_id,
                    self.selected_mod_id
                )
                obs["mod_features"]  = torch.tensor(mod_features, 
                                         dtype=torch.float32)
                slot_mask = self.get_slot_mask(self.selected_path_id,
                    self.selected_mod_id)
                slot_mask = torch.tensor(slot_mask, dtype=torch.bool)
            
     
        # MASKS
        mask =  {
                "path": path_mask,
                "mod": mod_mask,
                "slot": slot_mask
            }
        obs["masks"] = mask
        
        return obs
        
        
        
    def customreset(self, only_episode_counters=True):
        self.selected_path_id = None
        self.selected_mod_id = None
        self.selected_slot_id =  None
       
        obs = super().reset(only_episode_counters)
        
        return self._obs(obs)
      
        
    
    def customreset_old(self, only_episode_counters=True):
       
        obs = super().reset(only_episode_counters)
        return self._obs(obs)
        # print(obs['topology'].graph["node_indices"])
        
        # obs["node_features"] = self.get_node_features()
        obs["edge_features"] = self.get_edge_features()
        obs["edge_index"] = self.edge_index()
        # obs["demand_embedding"] = self.compute_demand_embedding()
        obs["candidate_paths"] =  self.get_candidate_paths()
        # obs["candidate_paths_length"] =  self.get_candidate_paths_length()
        # obs["candidate_paths_hops"] =  self.get_candidate_paths_hops()
        # obs["candidate_paths_minimum_osnr"] =  self.get_candidate_paths_minimum_osnr()
        # obs["candidate_paths_inband_xt"] =  self.get_candidate_paths_inband_xt()
        # obs["candidate_paths_impairment"] =  self.get_candidate_paths_impairment()
        
        path_mask, mod_mask, spec_mask = self.get_mask()
        obs["masks"] = (path_mask, mod_mask, spec_mask)
        obs["path_spectrum"]  = self.get_path_spectrum()
       
        return obs
    
    