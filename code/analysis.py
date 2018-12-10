#!/usr/bin/env python
# coding: utf-8

# In[1]:


import snap
import numpy as np
from matplotlib import pyplot as plt
import utils as ut
import networkx as nx
from math import ceil
from wordcloud import WordCloud, ImageColorGenerator
import collections
from PIL import Image

np.random.seed(42)


# In[2]:


# Load graphs and edge weights
OCN, OW = ut.load_ocn()
FPH, FW = ut.load_fph()
UCN, UW = ut.load_ucn()
SN, SW = ut.load_sn()
IFG, IRG, Mappings = ut.load_basic_graphs()
IIds = [NI.GetId() for NI in UCN.Nodes()]
iid_to_ingredient = Mappings['IID_to_Ingredient_Mapping']
ingredient_to_iid = {ingredient:iid for iid, ingredient in iid_to_ingredient.iteritems()}
RawCounts = sorted([(len(ut.get_common_neighbors(IRG, AIId, BIId)[0]),(AIId, BIId)) for i, AIId in enumerate(IIds[:-1]) for BIId in IIds[i+1:]], reverse=True)
RawCountsDict = {Edge:Count for Count, Edge in RawCounts}


# In[47]:


##############################################
########## Network Analysis Tools ############
##############################################

# Plot a degree distribution histogram
def hist_degree_distribution(G, name):
    filename = '../analysis/' + name + '_DegDistrHist'
    description = name + ': Degree Distribution Histogram'
    Histogram = [NI.GetDeg() for NI in G.Nodes()]
    plt.xlabel('Node Degree')
    plt.ylabel('Number of Nodes with a Given Degree')
    plt.hist(Histogram)
    plt.savefig(filename)
    plt.show()
    
def weighted_hist_degree_distribution(G, W, name):
    iid_to_weight = collections.defaultdict(int)
    for Edge, Weight in W.iteritems():
        iid_to_weight[Edge[0]] += Weight
        iid_to_weight[Edge[1]] += Weight
    filename = '../analysis/' + name + '_WeightedDegDistrHist'
    Histogram = [weight for iid, weight in iid_to_weight.iteritems()]
    plt.xlabel('Node Degree')
    plt.ylabel('Number of Nodes with a Given Degree')
    plt.hist(Histogram)
    plt.savefig(filename)
    plt.show()

# Plot the degree distribution of a complement network
def plot_degree_distribution(G, name):
    filename = '../analysis/' + name + '_DegDistr'
    description = name + ': Degree Distribution'
    X, Y = [], []
    DegToCntV = snap.TIntPrV()
    snap.GetOutDegCnt(G, DegToCntV)
    for item in DegToCntV:
        X.append(item.GetVal1())
        Y.append(item.GetVal2())
    plt.xlabel('Node Degree')
    plt.ylabel('Number of Nodes with a Given Degree')
    plt.title(description)
    plt.plot(X, Y, 'ro')
    plt.savefig(filename)
    plt.show()
    
# Print the clustering coefficient of a complement network
def clustering_coefficient(G):
    DegToCCfV = snap.TFltPrV()
    Result = snap.GetClustCfAll(G, DegToCCfV, -1)
    print 'Average Clustering Coefficient:', Result[0]

# Get the Top K Edge Weights of a complement network
def print_top_w(W, K, iid_to_ingredient, name=None, Reverse=True):
    print '\\begin{subtable}[b]{.23\linewidth}'
    print '% {} Table'.format(name)
    print '\centering'
    print '\\begin{tabular}{c|c|c}'
    print '\\textbf{Ingredient 1} & \\textbf{Ingredient 2} & \\textbf{Score}\\\\ \hline'
    
    
    OrderedWeights = sorted([(Weight, Edge) for Edge, Weight in W.iteritems()                                      if Edge[0] in iid_to_ingredient and Edge[1] in iid_to_ingredient],                                             reverse=Reverse)[:K]
    for i, (Weight, Edge) in enumerate(OrderedWeights):
        pairing = '{} & {} & {:.3f}\\\\'.format(iid_to_ingredient[Edge[0]], iid_to_ingredient[Edge[1]], Weight)
        if i < len(OrderedWeights) - 1:
            print pairing, '\hline'
        else:
            print pairing
        
    print '\end{tabular}'
    print '\caption{\\footnotesize {', name, '}}'
    print '\end{subtable}'
        
# Get random set of node pairs lacking an edge
def print_no_edge_pairs(G, K):
    NoEdges = []
    NIds = [NI.GetId() for NI in G.Nodes()]
    for i, ANId in enumerate(NIds[:-1]):
        for BNId in NIds[i+1:]:
            if not G.IsEdge(ANId, BNId):
                NoEdges.append((ANId, BNId))
    NoEdges = [NoEdges[i] for i in np.random.choice(range(len(NoEdges)), K)]
    for Edge in NoEdges:
        pair = '{},{}'.format(iid_to_ingredient[Edge[0]], iid_to_ingredient[Edge[1]])
        print pair

# Get the top K ingredients by page rank
def print_top_pr(G, K, iid_to_ingredient, Reverse=True):
    print 'PageRank:'
    PRankH = snap.TIntFltH()
    snap.GetPageRank(G, PRankH)
    PageRank = sorted([(PRankH[item], item) for item in PRankH], reverse=Reverse)
    for Rank, IId in PageRank[:K]:
        print '{}: {:.5f}'.format(iid_to_ingredient[IId], Rank)

# Get the communities in the graph
def get_communities(G):
    CmtyV = snap.TCnComV()
    modularity = snap.CommunityCNM(G, CmtyV)
    return [[IId for IId in Community] for Community in CmtyV], modularity

# Get a community for a given ingredient
def get_community_of_ingredient(Communities, IId):
    for C in Communities:
        if IId in C: return C
    return None
    
# Draw a full graph
def draw_graph(G, Communities, iid_to_ingredient):
    NG = nx.Graph()
    for NI in G.Nodes(): NG.add_node(NI.GetId())
    for EI in G.Edges(): NG.add_edge(EI.GetSrcNId(), EI.GetDstNId())
    print 'Kamada Kawai Graph Drawing'
    plt.figure(1)
#     nx.draw_kamada_kawai(NG)
    nx.draw(NG)
    plt.show()
#     print 'Spectral Graph Drawing'
#     plt.figure(2)
#     nx.draw_spectral(NG)
#     plt.show()
    pass

# Graph a word cloud for each community based on edge weight
def community_word_cloud(Communities, W, key_ingredient, name):
    # Get the community containing the key ingredient
    Community = []
    for C in Communities:
        if ingredient_to_iid[key_ingredient] in C:
            Community = C
            break
    # Create dictionary of words to sum of edge weights
    MinWeight = min([Weight for Edge, Weight in W.iteritems()])
    PowerDict = collections.defaultdict(int)
    for IId in Community:
        for Edge, Weight in W.iteritems():
            if Edge[0] == IId or Edge[1] == IId:
                PowerDict[IId] += (Weight - MinWeight)
    # Create a word list by dividing the power by the min power, taking the ceil, and adding that many words to the list
    WordList = []
    for IId, Power in PowerDict.iteritems():
        NumWords = int(ceil(Power))
        NewWords = [iid_to_ingredient[IId]] * NumWords
        WordList = WordList + NewWords
    # Graph the word cloud
    text = ' '.join(WordList)
    mask = np.array(Image.open("../analysis/color_wheel_mask.png"))
    wordcloud = WordCloud(background_color="rgba(255, 255, 255, 0)", mode="RGBA", mask=mask, max_words=2000, max_font_size=75, stopwords=[], collocations = False, width=1000, height=1000, margin=0).generate(text)
    # Create coloring from image
    image_colors = ImageColorGenerator(mask)
    wordcloud.recolor(color_func=image_colors)
    wordcloud.to_file('../analysis/{}_wordcloud.png'.format(name))
    # Display the generated image:
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.show()

# Network Analysis of G with weights W
def analyze_network(G, W, name):
    hist_degree_distribution(G, name)
    weighted_hist_degree_distribution(G, W, name)
#     plot_degree_distribution(G, name)
    clustering_coefficient(G)
    print_no_edge_pairs(G, 10)
    print_top_pr(G, 10, Mappings['IID_to_Ingredient_Mapping'])
    Communities, Modularity = get_communities(G)
#     print 'Communities:'
#     for C in Communities: print [iid_to_ingredient[IId] for IId in C]
#     print '\n'
#     draw_graph(G, Communities, Mappings['IID_to_Ingredient_Mapping'])
    if name == 'SN':
        community_word_cloud(Communities, W, 'egg', name + '_egg')
        community_word_cloud(Communities, W, 'black_pepper', name + '_black_pepper')
    else:
        community_word_cloud(Communities, W, 'egg', name)

# Analyze the ingredients in terms of 1. Common Recipes 2. FPHF 3.  4. PMI
def analyze_ingredients(TopK, Cuisine=None):
    iid_to_ingredient = Mappings['IID_to_Ingredient_Mapping']
    # Analyze Raw Counts
    # print 'Raw Counts:'
    # for Count, Edge in RawCounts[:TopK]:
    #     print '{}, {}: {}'.format(iid_to_ingredient[Edge[0]], iid_to_ingredient[Edge[1]], Count)
    # print '\n'
    # Analyze FPHF
    if Cuisine is not None:
        # Filter iid_to_ingredient using only ingredients from given cuisine
#         print 'Cuisine: {}'.format(Cuisine)
        ingredients = Mappings['Cuisine_to_List_of_Ingredients_Mapping'][Cuisine]
        ingredient_to_iid = {ingredient:iid for iid, ingredient in iid_to_ingredient.iteritems()}
        iti = {ingredient_to_iid[ingredient]:ingredient for ingredient in ingredients}
    else:
        iti = iid_to_ingredient.copy()
        
    print '\\begin{table*}[!ht]'
    print '\centering'
    print '\\fontsize{6}{6}\selectfont'

    print_top_w(FW, TopK, iti, 'FPHF', Reverse=True)
    print_top_w(UW, TopK, iti, 'COF', Reverse=True)
    print_top_w(SW, TopK, iti, 'SN', Reverse=True)
    print_top_w(OW, TopK, iti, 'PMI', Reverse=True)
    
    if '_' in Cuisine:
        Cuisine = ' '.join(Cuisine.split('_'))
    print '\caption{Cuisine:', Cuisine, '}'
    if ' ' in Cuisine:
        Cuisine = '-'.join(Cuisine.split(' '))
    print '\label{', '{}-top-scores'.format(Cuisine), '}'
    print '\end{table*}'


# In[48]:


###############################################
########## Main Analysis Execution ############
###############################################

def analysis():
    print '###############################################\n################# Analysis ####################\n###############################################'
    # Original Complement Network Analysis
    print 'Original Complement Network Analysis:\n'
    analyze_network(OCN, OW, 'OCN')
    print '\n'
    # Food Pairing Hypothesis Network Analysis
    print 'Food Pairing Hypothesis Network Analysis:\n'
    analyze_network(FPH, FW, 'FPH')
    print '\n'
    # Updated Complement Network Analysis
    print 'Updated Complement Network Analysis:\n'
    analyze_network(UCN, UW, 'UCN')
    print '\n'
    # Substitution Network Analysis
    print 'Substitution Network Analysis:\n'
    analyze_network(SN, SW, 'SN')
    print '\n'
    # Ingredient Analysis
    cuisines = Mappings['Cuisine_to_List_of_Ingredients_Mapping'].keys()
    for cuisine in sorted(cuisines):
#         print 'Ingredient Analysis:\n'
        analyze_ingredients(5, cuisine)


# In[49]:


if __name__ == '__main__':
    analysis()


# In[ ]:


###############################################
###### Ingredient Comparison Functions ########
###############################################

rid_to_ingredients = Mappings['RID_to_List_of_Ingredients_Mapping']
RIds = rid_to_ingredients.keys()

NIds = [NI.GetId() for NI in IFG.Nodes() if NI.GetId() in IIds]
MedFF = ut.MedFF(IFG, NIds)

def compare(Ingredient1, Ingredient2):
    print 'Comparing {} to {}'.format(Ingredient1, Ingredient2)
    # IIds to test
    AIId = ingredient_to_iid[Ingredient1]
    BIId = ingredient_to_iid[Ingredient2]
    print 'Median Flavor Factor:', MedFF
    FF = ut.JI(IFG, AIId, BIId)
    print 'Flavor Factor:', FF
    RF = ut.JI(IRG, AIId, BIId)
    print 'Recipe Factor:', RF
    FPHF = ut.FPHF(IFG, IRG, AIId, BIId, MedFF)
    print 'FPHF:', FPHF
    COF = ut.COF(IFG, IRG, AIId, BIId, len(RIds), MedFF)
    print 'COF:', COF
    PMI = ut.PMI(IRG, AIId, BIId, len(RIds))
    print 'PMI:', PMI


# In[ ]:


compare('walnut', 'cashew')
print '\n'


# In[ ]:




