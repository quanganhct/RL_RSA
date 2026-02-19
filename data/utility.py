import pandas as pd

import math

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

def read_network_file(filepath) -> tuple[pd.DataFrame, dict]:
    read_node_flag = True
    list_nid, list_city, list_population = [], [], []
    list_lat, list_long = [], []
    adjacent = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "Nodes:":
                read_node_flag = True
                continue
            
            if line == "Links:":
                read_node_flag = False
                continue

            if read_node_flag:
                nid, city, population, lat, long = line.split(",")
                list_nid.append(int(nid))
                list_city.append(city)
                list_population.append(int(population))
                list_lat.append(float(lat))
                list_long.append(float(long))
            else:
                nid1, nid2, length = line.split(" ")
                if int(nid1) not in adjacent:
                    adjacent[int(nid1)] = {}
                adjacent[int(nid1)][int(nid2)] = length

    nodedf = pd.DataFrame({"nid": list_nid, "city": list_city, "population": list_population, \
                           "lat": list_lat, "long": list_long})
    return nodedf, adjacent

        


        
