import snap
import numpy as np
import pandas as pd
import utils as ut
import collections
import pickle

##############################################
########### Data Reading Functions ###########
##############################################

########## Ingredient Data ##########

# Ingredient Information (IID, Ingredient Name, Category)
ingredient_info_file = '../data/ingr_comp/ingr_info.tsv'
def get_ingredient_info(filename=ingredient_info_file):
    ingredient_data = pd.read_table(filename).values
    
    # Map from IID to Ingredient
    iid_to_ingredient = {}
    # Map from Ingredient to Category
    ingredient_to_category = {}
    # Map from Category to List of Ingredients
    category_to_ingredients = collections.defaultdict(list)

    # Iterate through all rows in our ingredients data file
    for IID, Ingredient, Category in ingredient_data:
        iid_to_ingredient[IID] = Ingredient
        ingredient_to_category[Ingredient] = Category
        category_to_ingredients[Category].append(Ingredient)
    return iid_to_ingredient, ingredient_to_category, category_to_ingredients

# Flavor Information (FID, Flavor Name, CAS Number)
flavor_info_file = '../data/ingr_comp/comp_info.tsv'
def get_flavor_info(filename=flavor_info_file):
    flavor_data = pd.read_table(filename).values
    
    # Map from FID to Flavor
    fid_to_flavor = {}
    # Map from Flavor to CAS Number
    flavor_to_cas = {}
    # Map from CAS to List of Flavors
    cas_to_flavors = collections.defaultdict(list)
    
    # Iterate through all rows in our flavors data file
    for FID, Flavor, CAS in flavor_data:
        fid_to_flavor[FID] = Flavor
        flavor_to_cas[Flavor] = CAS
        cas_to_flavors[CAS].append(Flavor)
    return fid_to_flavor, flavor_to_cas, cas_to_flavors

# Ingredients to Flavors (IID, FID)
ingredients_flavors_file = '../data/ingr_comp/ingr_comp.tsv'
def get_ingredient_flavor_info(filename=ingredients_flavors_file):
    ingredient_flavor_data = pd.read_table(filename).values
    
    # Map from IID to FID
    iid_to_fid = collections.defaultdict(list)
    
    # Iterate through all rows in our ingredient to flavor graph data file
    for IId, FId in ingredient_flavor_data:
        iid_to_fid[IId].append(FId)
    return iid_to_fid

############ Recipe Data ############

def clean_cuisine(cuisine):
    if cuisine == 'china':
        return 'chinese'
    elif cuisine == 'france':
        return 'french'
    elif cuisine == 'germany':
        return 'german'
    elif cuisine == 'india':
        return 'indian'
    elif cuisine == 'italy':
        return 'italian'
    elif cuisine == 'japan':
        return 'japanese'
    elif cuisine == 'korea':
        return 'korean'
    elif cuisine == 'mexico':
        return 'mexican'
    elif cuisine == 'scandinavia':
        return 'scandinavian'
    elif cuisine == 'thailand':
        return 'thai'
    elif cuisine == 'vietnam':
        return 'vietnamese'
    else:
        return cuisine

# Recipe Information (Cuisine, List of Ingredients)
menu_recipes_file = '../data/scirep-cuisines-detail/menu_recipes.txt'
epic_recipes_file = '../data/scirep-cuisines-detail/epic_recipes.txt'
allr_recipes_file = '../data/scirep-cuisines-detail/allr_recipes.txt'
recipe_filenames = [menu_recipes_file, epic_recipes_file, allr_recipes_file]
def get_recipes_info(filenames=recipe_filenames):
    # Map from RID to List of Ingredients
    rid_to_ingredients = {}
    # Map from Cuisine to List of Ingredients
    cuisine_to_ingredients = collections.defaultdict(set)
    # Map from Ingredient to List of Cuisines
    ingredient_to_cuisines = collections.defaultdict(set)
    # Map from RID to Cuisine
    rid_to_cuisine = {}
    # Map from Cuisine to List of RIDs
    cuisine_to_rids = collections.defaultdict(list)
    
    # Helper function for reading in data from each file
    def read_recipe_file(filename, StartRId):
        RId = StartRId
        with open(filename) as file:
            for line in file:
                line = line.split()
                Cuisine, Ingredients = line[0], line[1:]
                Cuisine = Cuisine.lower()
                Cuisine = clean_cuisine(Cuisine)
                rid_to_ingredients[RId] = Ingredients
                for Ingredient in Ingredients: 
                    cuisine_to_ingredients[Cuisine].add(Ingredient)
                    ingredient_to_cuisines[Ingredient].add(Cuisine)
                rid_to_cuisine[RId] = Cuisine
                cuisine_to_rids[Cuisine].append(RId)
                RId += 1
        return RId
                
    StartRId = 0
    for filename in filenames:
        StartRId = read_recipe_file(filename, StartRId)
        
    return rid_to_ingredients, cuisine_to_ingredients, ingredient_to_cuisines, rid_to_cuisine, cuisine_to_rids

# Region Information (Cuisine, Region)
regions_file = '../data/scirep-cuisines-detail/map.txt'
def get_region_info(filename=regions_file):
    # Map from Cuisine to Regions
    cuisine_to_regions = collections.defaultdict(list)
    # Map from Region to Cuisines
    region_to_cuisines = collections.defaultdict(list)
    
    # Iterate through each line in our region to cuisine mapping file
    with open(filename) as file:
        for line in file:
            line = line.split()
            if len(line) != 2: continue
            Cuisine, Region = line
            Cuisine = Cuisine.lower()
            Cuisine = clean_cuisine(Cuisine)
            cuisine_to_regions[Cuisine].append(Region)
            region_to_cuisines[Region].append(Cuisine)
            
    return cuisine_to_regions, region_to_cuisines

##############################################
####### Basic Graph Building Functions #######
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
# Save the list of mappings
def save_mappings(mappings):
    for i, mapping in enumerate(mappings):
        filename = '../data/mappings/{}.pkl'.format(mapping_names[i])
        with open(filename, 'wb') as f:
            pickle.dump(mapping, f, pickle.HIGHEST_PROTOCOL)

# Save a graph as a binary file
def save_graph(G, filename):
    FOut = snap.TFOut(filename)
    G.Save(FOut)
    FOut.Flush()
    
# Build the Ingredient Flavor Graph
ingredient_flavor_graph_file = '../data/graphs/ingredient_flavor.graph'
def build_ingredient_flavor_graph(iid_to_ingredient, fid_to_flavor, iid_to_fid):
    IFG = snap.TUNGraph.New()
    
    # Add Nodes and Update Ids to Unique Integers, then Add Edges
    IIds, FIds = iid_to_ingredient.keys(), fid_to_flavor.keys()
    
    # Nodes in the ingredient mapping
    old_to_new_iids, new_iid_to_ingredient = {}, {}
    for IId in IIds:
        NewId = IFG.AddNode()
        new_iid_to_ingredient[NewId] = iid_to_ingredient[IId]
        old_to_new_iids[IId] = NewId
        
    # Nodes in the flavor mapping
    old_to_new_fids, new_fid_to_flavor = {}, {}
    for FId in FIds:
        NewId = IFG.AddNode()
        new_fid_to_flavor[NewId] = fid_to_flavor[FId]
        old_to_new_fids[FId] = NewId
    
    # Add Edges
    for IId, FIds in iid_to_fid.iteritems():
        for FId in FIds:
            IFG.AddEdge(old_to_new_iids[IId], old_to_new_fids[FId])
        
    # Return constructed graph and new, updated mappings
    return IFG, new_iid_to_ingredient, new_fid_to_flavor

# Build the Ingredient Recipe Graph (Assumes IFG already build first for node numbering)
ingredient_recipe_graph_file = '../data/graphs/ingredient_recipe.graph'
def build_ingredient_recipe_graph(iid_to_ingredient, rid_to_ingredients):
    IRG = snap.TUNGraph.New()
    
    # Add all of the ingredients
    IIds = iid_to_ingredient.keys()
    ingredient_to_iid = {Ingredient:iid for iid, Ingredient in iid_to_ingredient.iteritems()}
    for IId in IIds: IRG.AddNode(IId)
    
    # Add all of the recipes, update rids in the process
    new_rid_to_ingredients, old_to_new_rids = {}, {}
    for RId, Ingredients in rid_to_ingredients.iteritems():
        NewId = IRG.AddNode()
        for Ingredient in Ingredients:
            IRG.AddEdge(NewId, ingredient_to_iid[Ingredient])
        new_rid_to_ingredients[NewId] = Ingredients
        old_to_new_rids[RId] = NewId

    # Return constructed graph and new, updated mappings
    return IRG, new_rid_to_ingredients, old_to_new_rids

# Prune the nodes in the graph such that Ingredient nodes in IFG have an edge in IRG
def prune_graphs(IFGraph, IRGraph, IIds):
    Remove = []
    for NI in IRGraph.Nodes():
        # Skip Recipe Nodes
        if NI.GetId() not in IIds: continue
        # If the ingredient does not appear in any recipes, remove it
        if NI.GetDeg() == 0: Remove.append(NI.GetId())
    for NId in Remove:
        IFGraph.DelNode(NId)
        IRGraph.DelNode(NId)

# Update Mappings Involving RIDs with New RIDs
def new_cuisine_mappings(old_to_new, rid_to_cuisine, cuisine_to_rids):
    # First simple update the one to one mapping rid to cuisine
    new_rid_to_cuisine = {}
    for RId, Cuisine in rid_to_cuisine.iteritems():
        new_rid_to_cuisine[old_to_new[RId]] = Cuisine
    # Note update rid lists for each cuisine
    new_cuisine_to_rids = collections.defaultdict(list)
    for Cuisine, RIds in cuisine_to_rids.iteritems():
        for RId in RIds:
            new_cuisine_to_rids[Cuisine].append(old_to_new[RId])
    # Return updated mappings
    return new_rid_to_cuisine, new_cuisine_to_rids
    
##############################################
########## Weight Saving Function ############
##############################################

def save_weights(weights, filename):
    with open(filename, 'wb') as f:
        pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)
    
##############################################
######## Original Complement Network #########
##############################################

ocn_graph_file = '../data/graphs/ocn.graph'
ocn_weights_file = '../data/weights/ocn_weights.pkl'
def build_original_complement_network(RecipeThreshold=20):
    # Get original Bipartite Graphs
    IFG, IRG, Mappings = ut.load_basic_graphs()
    IIds = Mappings['IID_to_Ingredient_Mapping'].keys()
    RIds = Mappings['RID_to_List_of_Ingredients_Mapping'].keys()
    
    OCN = snap.TUNGraph.New()
    Weights = {}
    
    # Add Nodes to Complement Network
    for IId in IIds:
        # Skip nodes that were pruned
        if not IRG.IsNode(IId): continue
        # Otherwise Add Node
        OCN.AddNode(IId)
        
    # Go through all pairs of nodes and add an edge weighted by PMI
    NumRecipes = len(RIds)
    NIds = [NI.GetId() for NI in OCN.Nodes()]
    for i, AIId in enumerate(NIds[:-1]):
        for BIId in NIds[i+1:]:
            B, W = ut.PMI(IRG, AIId, BIId, NumRecipes, RecipeThreshold)
            if B:
                OCN.AddEdge(AIId, BIId)
                Weights[(AIId, BIId)] = W
    
    # Save Graph
    save_graph(OCN, ocn_graph_file)
    # Save Weights
    save_weights(Weights, ocn_weights_file)

##############################################
###### Food Pairing Hypothesis Network #######
##############################################

fph_graph_file = '../data/graphs/fph.graph'
fph_weights_file = '../data/weights/fph_weights.pkl'
def build_food_pairing_hypothesis_network(RecipeThreshold=20):
    # Get original Bipartite Graphs
    IFG, IRG, Mappings = ut.load_basic_graphs()
    IIds = Mappings['IID_to_Ingredient_Mapping'].keys()
    RIds = Mappings['RID_to_List_of_Ingredients_Mapping'].keys()
    
    FPH = snap.TUNGraph.New()
    Weights = {}
    
    # Add Nodes to Complement Network
    for IId in IIds:
        # Skip nodes that were pruned
        if not IFG.IsNode(IId) or not IRG.IsNode(IId): continue
        # Otherwise Add Node
        FPH.AddNode(IId)
        
    # Go through all pairs of nodes and add an edge weighted by PMI
    NIds = [NI.GetId() for NI in FPH.Nodes()]
    MedFF = ut.MedFF(IFG, NIds)
    for i, AIId in enumerate(NIds[:-1]):
        for BIId in NIds[i+1:]:
            B, W = ut.FPHF(IFG, IRG, AIId, BIId, MedFF, RecipeThreshold)
            if B:
                FPH.AddEdge(AIId, BIId)
                Weights[(AIId, BIId)] = W
    
    # Save Graph
    save_graph(FPH, fph_graph_file)
    # Save Weights
    save_weights(Weights, fph_weights_file)

##############################################
######### Updated Complement Network #########
##############################################

ucn_graph_file = '../data/graphs/ucn.graph'
ucn_weights_file = '../data/weights/ucn_weights.pkl'
def build_updated_complement_network(RecipeThreshold=20):
    # Get original Bipartite Graphs
    IFG, IRG, Mappings = ut.load_basic_graphs()
    IIds = Mappings['IID_to_Ingredient_Mapping'].keys()
    RIds = Mappings['RID_to_List_of_Ingredients_Mapping'].keys()
    
    UCN = snap.TUNGraph.New()
    Weights = {}
    
    # Add Nodes to Complement Network
    for IId in IIds:
        # Skip nodes that were pruned
        if not IFG.IsNode(IId) or not IRG.IsNode(IId): continue
        # Otherwise Add Node
        UCN.AddNode(IId)
        
    # Go through all pairs of nodes and add an edge weighted by PMI
    NumRecipes = len(RIds)
    NIds = [NI.GetId() for NI in UCN.Nodes()]
    MedFF = ut.MedFF(IFG, NIds)
    for i, AIId in enumerate(NIds[:-1]):
        for BIId in NIds[i+1:]:
            B, W = ut.COF(IFG, IRG, AIId, BIId, NumRecipes, MedFF, RecipeThreshold)
            if B:
                UCN.AddEdge(AIId, BIId)
                Weights[(AIId, BIId)] = W
    
    # Save Graph
    save_graph(UCN, ucn_graph_file)
    # Save Weights
    save_weights(Weights, ucn_weights_file)

##############################################
####### Inferred Substitution Netowork #######
##############################################

sn_graph_file = '../data/graphs/sn.graph'
sn_weights_file = '../data/weights/sn_weights.pkl'
def build_substitution_network():
    # Get original Bipartite Graphs
    IFG, IRG, Mappings = ut.load_basic_graphs()
    IIds = Mappings['IID_to_Ingredient_Mapping'].keys()
    RIds = Mappings['RID_to_List_of_Ingredients_Mapping'].keys()
    
    SN = snap.TUNGraph.New()
    Weights = {}
    
    # Add Nodes to Complement Network
    for IId in IIds:
        # Skip nodes that were pruned
        if not IFG.IsNode(IId) or not IRG.IsNode(IId): continue
        # Otherwise Add Node
        SN.AddNode(IId)
        
    # Go through all pairs of nodes and add an edge weighted by PMI
    NIds = [NI.GetId() for NI in SN.Nodes()]
    FlavorThreshold = ut.MeanCommonFlavors(IFG, NIds)
    print 'Mean Common Flavors:', FlavorThreshold
    for i, AIId in enumerate(NIds[:-1]):
        for BIId in NIds[i+1:]:
            B, W = ut.SF(IFG, IRG, AIId, BIId, FlavorThreshold)
            if B:
                SN.AddEdge(AIId, BIId)
                Weights[(AIId, BIId)] = W
    
    # Save Graph
    save_graph(SN, sn_graph_file)
    # Save Weights
    save_weights(Weights, sn_weights_file)

##############################################
############### Build Graphs #################
##############################################

# Build The Base Graphs
def build_basic_graphs():
    print 'Building Basic Graphs...'
    # Get all of the necessary information from our data files
    ingredient_mappings = get_ingredient_info()
    flavor_mappings = get_flavor_info()
    iid_to_fid = get_ingredient_flavor_info()
    recipe_mappings = get_recipes_info()
    region_mappings = get_region_info()
    
    # Build the Ingredient Flavor Bipartite Graph (retrieve new iids and fids)
    IFGraph, iid_to_ingredient, fid_to_flavor = build_ingredient_flavor_graph(ingredient_mappings[0], flavor_mappings[0], iid_to_fid)
    
    # Build the Ingredient Recipe Bipartite Graph (retrieve new rids)
    IRGraph, rid_to_ingredients, old_to_new_rids = build_ingredient_recipe_graph(iid_to_ingredient, recipe_mappings[0])
    
    print 'IFGraph Num Nodes:', IFGraph.GetNodes()
    print 'IRGraph Num Nodes:', IRGraph.GetNodes()
    
    # Prune the graphs of ingredients not contained in recipes
    prune_graphs(IFGraph, IRGraph, iid_to_ingredient.keys())
    
    print '(POST PRUNE) IFGraph Num Nodes:', IFGraph.GetNodes()
    print '(POST PRUNE) IRGraph Num Nodes:', IRGraph.GetNodes()
    
    # Save the graphs as binary files for future quick loading
    save_graph(IFGraph, ingredient_flavor_graph_file)
    save_graph(IRGraph, ingredient_recipe_graph_file)
    
    # Update RID mappings with new RIDs
    rid_to_cuisine, cuisine_to_rids = new_cuisine_mappings(old_to_new_rids, recipe_mappings[3], recipe_mappings[4])
    
    # Save all of the final mappings
    mappings = [
        iid_to_ingredient,           # IID to Ingredient Mapping
        ingredient_mappings[1],      # Ingredient to Category Mapping
        ingredient_mappings[2],      # Category to Ingredient Mapping
        fid_to_flavor,               # FID to Flavor Mapping
        flavor_mappings[1],          # Flavor to CAS Mapping
        flavor_mappings[2],          # CAS to List of Flavors Mapping
        rid_to_ingredients,          # RID to List of Ingredients Mapping
        recipe_mappings[1],          # Cuisine to List of Ingredients Mapping
        recipe_mappings[2],          # Ingredient to List of Cuisines Mapping
        rid_to_cuisine,              # RID to Cuisine Mappings
        cuisine_to_rids,             # Cuisine to List of RIDs Mapping
        region_mappings[0],          # Cuisine to Regions
        region_mappings[1]           # Region to Cuisines
    ]
    save_mappings(mappings)
    
# Build Networks
def build_networks():
    RecipeThreshold = 25
    print 'Building Original Complement Network...'
    build_original_complement_network(RecipeThreshold)
    print 'Building Food Pairing Hypothesis Network...'
    build_food_pairing_hypothesis_network(RecipeThreshold)
    print 'Building Updated Complement Network...'
    build_updated_complement_network(RecipeThreshold)
    print 'Building Substitution Network...'
    build_substitution_network()

##############################################
########### Main Program Execution ###########
##############################################

def main():
    print 'Build Process:'
    build_basic_graphs()
    build_networks()
    print 'Done!'

if __name__ == '__main__':
    main()

