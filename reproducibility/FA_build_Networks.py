
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict
from igraph import Graph
from nltk.corpus import wordnet as wn
import pickle
import random
from FA_Functions import *


###############################################################################

# Load data
models = ['Humans', 'Mistral', 'Llama3', 'Haiku']
FA_clean_dfs = {}
for model in models:
    FA_clean_dfs[model] = pd.read_csv('./data/processed_datasets/FA_' + model + '.csv')

# Create graphs
# All graphs are weighted and directed
FA_graphs = {}
for model in models:
    edges = FA_edgeList(FA_clean_dfs[model])
    g = graphFromEdgeList(edges, directed = True, weighted = True)
    FA_graphs[model] = g

# Make graphs undirected
FA_graphs_full = {}
for model, g in FA_graphs.items():
    g = makeUndirected(g)
    FA_graphs_full[model] = g

# Create the filtered graphs by:
# keeping only WN nodes
# keeping only non-idiosyncratic edges
# taking the LCC
FA_graphs_filt = {}
for model in models:
    g = FA_graphs_full.copy()[model]
    g = WNfilter(g)
    g = idiosynfilter(g)
    g = CC(g)
    FA_graphs_filt[model] = g


# Get graph summaries
graph_summary(FA_graphs_full)
graph_summary(FA_graphs_filt)
graph_summary(FA_graphs_full).to_csv('./data/summary_tables/FA_graphs_full_summary.csv')
graph_summary(FA_graphs_filt).to_csv('./data/summary_tables/FA_graphs_filtered_summary.csv')

# Get a sample of nodes that were removed from the graphs to see the types of nodes removed
removedNodes = {}
for model in models:
    random.seed(30)
    rem = set(FA_graphs_full[model].nodes()) - set(FA_graphs_filt[model].nodes())
    removedNodes[model] = random.sample(rem, 20)
removedNodesDF = pd.DataFrame(removedNodes)
removedNodesDF

# Graph comparisons
compDict = {}
for model, graph in FA_graphs_filt.items():
    df = netComparison(FA_graphs_filt['Humans'], graph)
    compDict[model] = df

compDict2 = {'Comparison with Humans':
['Percentage of Human nodes not in LLM network',
'Percentage of all nodes common to both networks',
'Percentage of LLM nodes not in Human network',
'Percentage of Human edges not in LLM network',
'Percentage of all edges common to both networks',
'Percentage of LLM edges not in Human network']}

for model in ['Mistral', 'Llama3', 'Haiku']:
    compDict2[model] = list(compDict[model].values())
compDF = pd.DataFrame(compDict2)
compDF.to_csv('./data/summary_tables/FA_graph_comparisons.csv', index = False)

# Save edge lists
for model, g in FA_graphs_filt.items():
    graph2csv(g, model)

# Convert graphs to igraphs to do the spreading activation
for model, g in FA_graphs_filt.items():
    nxGraph2igraph(g, name = model + '_ig', directed = False, weighted = True)        
        