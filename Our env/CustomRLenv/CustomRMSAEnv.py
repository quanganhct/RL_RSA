# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 12:06:57 2026

@author: Momo
"""

import numpy as np
import networkx as nx
from optical_rl_gym.envs.rmsa_env import RMSAEnv
import math

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
        
            for i in range(self.k_paths):
                path = self.topology.graph['ksp'][(src, dst)][i].node_list
                path = self.k_shortest_paths[(src, dst)][i].node_list
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
     
    def get_candidate_paths(self):
        if not hasattr(self, "current_service"):
            return [np.zeros(1) for i in range(self.k_paths)]

        src = self.current_service.source
        dst = self.current_service.destination
        
        return self.candidate_paths[(src, dst)]
    
    def get_candidate_paths_length(self):
        if not hasattr(self, "current_service"):
            return  [0 for i in range(self.k_paths)]

        src = self.current_service.source
        dst = self.current_service.destination
        
        return self.candidate_paths_length[(src, dst)] 
    
    def get_candidate_paths_hops(self):
        if not hasattr(self, "current_service"):
            return [1 for i in range(self.k_paths)]

        src = self.current_service.source
        dst = self.current_service.destination
        
        return self.candidate_paths_hops[(src, dst)]
    
    def get_candidate_paths_minimum_osnr(self):
        if not hasattr(self, "current_service"):
            return [0 for i in range(self.k_paths)]

        src = self.current_service.source
        dst = self.current_service.destination
        
        return self.candidate_paths_minimum_osnr[(src, dst)]
    
    def get_candidate_paths_inband_xt(self):
        if not hasattr(self, "current_service"):
            return [0 for i in range(self.k_paths)]

        src = self.current_service.source
        dst = self.current_service.destination
        
        return self.candidate_paths_inband_xt[(src, dst)]
    
    def get_candidate_paths_impairment(self):
        if not hasattr(self, "current_service"):
            return [[0,0,0] for i in range(self.k_paths)]

        src = self.current_service.source
        dst = self.current_service.destination
        
        return self.candidate_paths_impairment[(src, dst)]
    
    
    # -----------------------------
    # Edge feature vector x_e
    # -----------------------------
    def get_edge_features(self):
        features = []
        for link_id, link in enumerate(self.topology.edges()):
            # Spectrum occupancy vector: 1=occupied, 0=free
            s_e = self.spectrum_usage[link_id]

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

            # Fragmentation / contiguity metric H_e
            H_e = self.compute_fragmentation(link_id)
            

            # edge_vec = np.concatenate([s_e, [l_e, osnr, nli, ase, H_e]])
            edge_vec = np.concatenate([s_e, [l_e, 
                                             utilization, 
                                             running_services, 
                                             external_fragmentation, 
                                             compactness,  
                                             H_e]])
            features.append(edge_vec)
        return np.array(features, dtype=np.float32)
    
    def compute_fragmentation(self, link_id):
        """
        Example: H_e = fraction of free slots that are non-contiguous
        """
        free_slots = np.where(np.array(self.spectrum_usage[link_id]) == 0)[0]
        if len(free_slots) == 0:
            return 1.0  # fully fragmented
        contiguous_blocks = 1
        for i in range(1, len(free_slots)):
            if free_slots[i] != free_slots[i-1] + 1:
                contiguous_blocks += 1
        H_e = contiguous_blocks / len(self.spectrum_usage[link_id])
        return H_e
    
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

    
    
    # -----------------------------
    # Custom step to update features
    # -----------------------------
    def step(self, action):
        next_state, reward, done, info = super().step(action)

        # Update node & edge features 
        # print(f'src = {self.current_service.source_id} , dst = {self.current_service.destination_id}')
        # print(f'reward = {reward}')
      
        next_state["node_features"] = self.get_node_features()
        next_state["edge_features"] = self.get_edge_features()
        next_state["demand_embedding"] = self.compute_demand_embedding()
        next_state["candidate_paths"] =  self.get_candidate_paths()
        next_state["candidate_paths_length"] =  self.get_candidate_paths_length()
        next_state["candidate_paths_hops"] =  self.get_candidate_paths_hops()
        next_state["candidate_paths_minimum_osnr"] =  self.get_candidate_paths_minimum_osnr()
        next_state["candidate_paths_inband_xt"] =  self.get_candidate_paths_inband_xt()
        next_state["candidate_paths_impairment"] =  self.get_candidate_paths_impairment()

        # Keep demand embedding and impairment
        next_state["demand_embedding"] = self.compute_demand_embedding()
        
        # if "path" in info:
        #     path_idx = info["path"]
        #     impairment = self.get_path_impairment(path_idx)
        # else:
        #     impairment = 0
        # reward -= 0.1 * impairment
        # info["impairment_penalty"] = impairment
        
        path_mask, mod_mask, spec_mask = self.get_mask()
        next_state["masks"] = (path_mask, mod_mask, spec_mask) = self.get_mask()
        next_state["path_spectrum"]  = self.get_path_spectrum()
        
        
        return next_state, reward, done, info
    
    def get_number_slots_given_modulation(self, path, modulation):
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
    
    
        
    def get_available_blocks(self, path, modulation=None):
        # get available slots across the whole path
        # 1 if slot is available across all the links
        # zero if not
        
        available_slots = self.get_available_slots(
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
            slots = self.get_number_slots_given_modulation(
                self.k_shortest_paths[
                    self.current_service.source, self.current_service.destination
                ][path], modulation
            )
            
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
    
    def get_path_spectrum(self):
        path_spectrum = []
        for path_idx in range(self.k_paths):
            # get available block given best
            spectrum = self.get_available_slots(
                self.k_shortest_paths[
                    self.current_service.source, self.current_service.destination
                ][path_idx])
        
            path_spectrum.append(spectrum)
        return path_spectrum
    
    
    # -----------------------------
    # Optional: helper for masks
    # -----------------------------
    def get_mask(self):
        
        if not hasattr(self, "current_service"):
            path_mask = np.ones(self.k_paths)
            
            mod_mask = np.ones([self.k_paths, 
                                len(self.topology.graph['modulations'])])
            
            spec_mask = np.ones([self.k_paths, 
                                len(self.topology.graph['modulations']), 
                                self.num_spectrum_resources])
            return  path_mask, mod_mask, spec_mask
        
        
        src = self.current_service.source
        dst = self.current_service.destination
        
        path_mask = np.ones(self.k_paths)
        
        mod_mask = np.ones([self.k_paths, 
                            len(self.topology.graph['modulations'])])
        
        spec_mask = np.ones([self.k_paths, 
                            len(self.topology.graph['modulations']), 
                            self.num_spectrum_resources])
        # Masking sheme
        for path_idx in range(self.k_paths):
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
                
                
                
                
                
        
        
    def customreset(self, only_episode_counters=True):
       
        obs = super().reset(only_episode_counters)
        # print(obs['topology'].graph["node_indices"])
        obs["node_features"] = self.get_node_features()
        obs["edge_features"] = self.get_edge_features()
        obs["demand_embedding"] = self.compute_demand_embedding()
        obs["candidate_paths"] =  self.get_candidate_paths()
        obs["candidate_paths_length"] =  self.get_candidate_paths_length()
        obs["candidate_paths_hops"] =  self.get_candidate_paths_hops()
        obs["candidate_paths_minimum_osnr"] =  self.get_candidate_paths_minimum_osnr()
        obs["candidate_paths_inband_xt"] =  self.get_candidate_paths_inband_xt()
        obs["candidate_paths_impairment"] =  self.get_candidate_paths_impairment()
        
        path_mask, mod_mask, spec_mask = self.get_mask()
        obs["masks"] = (path_mask, mod_mask, spec_mask)
        obs["path_spectrum"]  = self.get_path_spectrum()
       
        return obs
    