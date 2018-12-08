import snap
import numpy as np
import pandas as pd
import collections
import pickle

##############################################
########## Graph Loading Functions ###########
##############################################

mapping_names = [
    'IID_to_Ingredient_Mapping',
    'Ingredient_to_Category_Mapping',
    'Category_to_Ingredient_Mapping',
    'FID_to_Flavor_Mapping',
    'Flavor_to_CAS_Mapping',
    'CAS_to_List_of_Flavors_Mapping',
    'RID_to_List_of_Ingredients_Mapping',
    'Cuisine_to_List_of_Ingredients_Mapping',
    'Ingredient_to_List_of_Cuisines_Mapping',
    'RID_to_Cuisine_Mappings',
    'Cuisine_to_List_of_RIDs_Mapping',
    'Cuisine_to_Regions',
    'Region_to_Cuisines'
]
# Load the dictionary of names to mappings
def load_mappings():
    mappings = {}
    for name in mapping_names:
        filename = '../data/mappings/{}.pkl'.format(name)
        with open(filename, 'rb') as f:
            mappings[name] = pickle.load(f)
    return mappings

# Load mappings for graph embeddings
embedding_names = [
    'ocn',
    'fph',
    'ucn',
    'sn'
]
def load_embeddings():
    mappings = {}
    for name in embedding_names:
        filename = '../data/mappings/{}_emb_map.pkl'.format(name)
        with open(filename, 'rb') as f:
            mappings[name] = pickle.load(f)
    return mappings
            
# Load an undirected graph from a binary file
def load_graph(filename):
    FIn = snap.TFIn(filename)
    return snap.TUNGraph.Load(FIn)

# Load Weights Dictionary for a given file
def load_weights(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

ingredient_flavor_graph_file = '../data/graphs/ingredient_flavor.graph'
ingredient_recipe_graph_file = '../data/graphs/ingredient_recipe.graph'
# Load the Ingredient-Flavor and Ingredient-Recipe Graphs and the data mappings
def load_basic_graphs():
    return load_graph(ingredient_flavor_graph_file), load_graph(ingredient_recipe_graph_file), load_mappings()

# Load the Original Complement Network
ocn_graph_file = '../data/graphs/ocn.graph'
ocn_weights_file = '../data/weights/ocn_weights.pkl'
def load_ocn():
    return load_graph(ocn_graph_file), load_weights(ocn_weights_file) 

# Load the Food Pairing Hypothesis Network
fph_graph_file = '../data/graphs/fph.graph'
fph_weights_file = '../data/weights/fph_weights.pkl'
def load_fph():
    return load_graph(fph_graph_file), load_weights(fph_weights_file)

# Load the Updated Complement Network
ucn_graph_file = '../data/graphs/ucn.graph'
ucn_weights_file = '../data/weights/ucn_weights.pkl'
def load_ucn():
    return load_graph(ucn_graph_file), load_weights(ucn_weights_file)

# Load the Substitution Network
sn_graph_file = '../data/graphs/sn.graph'
sn_weights_file = '../data/weights/sn_weights.pkl'
def load_sn():
    return load_graph(sn_graph_file), load_weights(sn_weights_file) 

##############################################
############# General Functions ##############
##############################################

def euclidean_distance(X, Y):
  return np.sqrt(((X - Y) ** 2).sum())

def get_nbr_set(G, NId):
    return set([Nbr for Nbr in G.GetNI(NId).GetOutEdges()])

def get_common_neighbors(G, ANId, BNId):
    ANbr = get_nbr_set(G, ANId)
    BNbr = get_nbr_set(G, BNId)
    return ANbr.intersection(BNbr), ANbr, BNbr

# Compute the Pointwise Mutual Information metric from the paper (Note NR = Number of Recipes)
def PMI(IRG, AIId, BIId, NR, Threshold=0):
    CommonNeighbors, ANbr, BNbr = get_common_neighbors(IRG, AIId, BIId)
    NumInCommon = len(CommonNeighbors)
    if NumInCommon >= Threshold:
        return True, np.log(NumInCommon) - np.log(len(ANbr)) - np.log(len(BNbr)) + np.log(NR)
    else:
        return False, float('-inf')

# Compute the Jaccard Index for two nodes in the graph G (where the intersection must be at least of size threshold)
def JI(G, ANId, BNId, Threshold=0):
    CommonNeighbors, ANbr, BNbr = get_common_neighbors(G, ANId, BNId)
    NUMER = len(CommonNeighbors)
    DENOM = len(ANbr.union(BNbr))
    return  float(NUMER) / DENOM if NUMER >= Threshold else 0

# Compute the Food Pairing Hypothesis Factor for two Ingredients A and B
def FPHF(IFG, IRG, AIId, BIId, MedFF, Threshold=0):
    FF, RF = JI(IFG, AIId, BIId), JI(IRG, AIId, BIId, Threshold)
    Score = RF * ((FF - MedFF) ** 2)
    return Score > 0, Score

# Compute the Co-Occurrence Factor for two Ingredients A and B
def COF(IFG, IRG, AIId, BIId, NR, MedFF, Threshold=0):
    FF = JI(IFG, AIId, BIId)
    B, PMIScore = PMI(IRG, AIId, BIId, NR, Threshold)
    return B, PMIScore + np.sqrt(((FF - MedFF) ** 2))

# Compute the Substitution Factor for two Ingredients A and B
def SF(IFG, IRG, AIId, BIId, FlavorThreshold=0):
    FF, RF = JI(IFG, AIId, BIId, FlavorThreshold), JI(IRG, AIId, BIId)
    Score = FF / (1 + RF)
    return Score > 0, Score

def MedFF(IFG, IIds):
    FFs = [JI(IFG, AIId, BIId) for i, AIId in enumerate(IIds[:-1]) for BIId in IIds[i+1:]]
    return (min(FFs) + max(FFs)) / 2

def MeanFF(IFG, IIds):
    return np.mean([JI(IFG, AIId, BIId) for i, AIId in enumerate(IIds[:-1]) for BIId in IIds[i+1:]])

def StdFF(IFG, IIds):
    return np.std([JI(IFG, AIId, BIId) for i, AIId in enumerate(IIds[:-1]) for BIId in IIds[i+1:]])

def MeanCommonFlavors(IFG, IIds):
    return np.mean([len(get_common_neighbors(IFG, AIId, BIId)[0]) for i, AIId in enumerate(IIds[:-1]) for BIId in IIds[i+1:]])
