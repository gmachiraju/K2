import os
import sys
import numpy as np
import networkx as nx
sys.path.insert(1, '/scr/gmachi/prospection/K2/src')
from utils import deserialize, serialize, deserialize_model
from utils import construct_sprite
from utils import convert_graph2df
from utils import visualize_concept_density_hexbin, visualize_cluster_density_hexbin

import pandas as pd
import geopandas
import libpysal
from libpysal.weights import Queen, lag_spatial, KNN
from esda.moran import Moran


def moran_concepts(IDs, proc):
    moran_dict = {}
    for i,ID in enumerate(IDs):
        G_og = "/scr/biggest/gmachi/datasets/celldive_lung/for_ml/for_prospect/S" + str(ID) + ".obj"
        G_og = deserialize(G_og)
        S = construct_sprite(G_og, proc, key_in="emb", key_out="concept")
        if i == 0:
            df = convert_graph2df(S, key="concept")
            categorical = df['concept'].values
        print("ID: ", ID)

        mis = []
        for C in set(categorical):
            coll = visualize_concept_density_hexbin(S, concept=C, viz_flag=False)
            offsets = coll.get_offsets()
            arr = coll.get_array()
            df2 = pd.DataFrame(offsets, columns=["x", "y"])
            df2["concept"] = arr.data
            geometry = geopandas.points_from_xy(df2['x'], df2['y'], df2['concept'])
            gdf = geopandas.GeoDataFrame(df2, geometry=geometry)
            w = Queen.from_dataframe(gdf, use_index=False)
            mi = Moran(gdf['concept'].values, w)
            mis.append(mi.I)
        moran_dict[ID] = mis
    return moran_dict

def moran_clusters(IDs, proc):
    moran_dict = {}
    for i,ID in enumerate(IDs):
        G_og = "/scr/biggest/gmachi/datasets/celldive_lung/for_ml/for_prospect/S" + str(ID) + ".obj"
        G_og = deserialize(G_og)
        S = construct_sprite(G_og, proc, key_in="emb", key_out="concept")
        if i == 0:
            df = convert_graph2df(S, key="concept")
            categorical = df['concept'].values
        print("ID: ", ID)
        
        # assign clusters
        kmeans = deserialize("/scr/biggest/gmachi/datasets/celldive_lung/kmeans_on_raw.obj")
        X = np.array(list(nx.get_node_attributes(S, "raw").values()))
        cs = kmeans.predict(X)
        nx.set_node_attributes(S, dict(zip(S.nodes, cs)), "cluster")
        
        mis = []
        for C in set(categorical):
            coll = visualize_cluster_density_hexbin(S, cluster=C, viz_flag=False)
            offsets = coll.get_offsets()
            arr = coll.get_array()
            df2 = pd.DataFrame(offsets, columns=["x", "y"])
            df2["cluster"] = arr.data
            geometry = geopandas.points_from_xy(df2['x'], df2['y'], df2['cluster'])
            gdf = geopandas.GeoDataFrame(df2, geometry=geometry)
            w = Queen.from_dataframe(gdf, use_index=False)
            mi = Moran(gdf['cluster'].values, w)
            mis.append(mi.I)
        moran_dict[ID] = mis
    return moran_dict
    

def main():
    proc_path = "/scr/gmachi/prospection/K2/notebooks/spatial-bio/outputs/gridsearch_results/k2processors/k9.processor"
    proc = deserialize_model(proc_path)

    filenames = os.listdir("/scr/biggest/gmachi/datasets/celldive_lung/for_ml/for_prospect/")
    IDs = [int(f.split("S")[1].split(".obj")[0]) for f in filenames]
    
    # moran_dict = moran_concepts(IDs, proc)
    # serialize(moran_dict, "/scr/gmachi/prospection/K2/notebooks/spatial-bio/outputs/moran_dict.obj")
    moran_dict = moran_clusters(IDs, proc)
    serialize(moran_dict, "/scr/gmachi/prospection/K2/notebooks/spatial-bio/outputs/moran_dict_cluster.obj")

if __name__ == "__main__":
    main()