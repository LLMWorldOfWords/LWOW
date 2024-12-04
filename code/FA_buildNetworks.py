
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict
from igraph import Graph
from nltk.corpus import wordnet as wn
import pickle
import random
import os
from FA_Functions import *

base_dir = os.path.dirname(os.path.dirname(__file__))

###############################################################################

# Load data
models = ['Humans', 'Mistral', 'Llama3', 'Haiku', 'GPT-3.5 Turbo', 'GPT-4o', 'GPT-4 Turbo']
FA_clean_dfs = {}
for model in models:
    FA_clean_dfs[model] = pd.read_csv(os.path.join(base_dir, 'data/processed_datasets/FA_datasets/FA_' + model + '.csv'))

# Create graphs
# All graphs are weighted and directed, except GPT graphs are unweighted
FA_graphs = {}
for model in models:
    edges = FA_edgeList(FA_clean_dfs[model])
    if 'GPT' in model:
        g = graphFromEdgeList(edges, directed = True, weighted = False)
    else:
        g = graphFromEdgeList(edges, directed = True, weighted = True)
    FA_graphs[model] = g

# save graphs
pickle.dump(FA_graphs, open(os.path.join(base_dir, 'data/graphs/FA_all_graphs.pickle', 'wb')))

# Make graphs undirected
FA_graphs_full = {}
for model, g in FA_graphs.items():
    if 'GPT' not in model:
        g = makeUndirected(g)
    if 'GPT' in model:
        g = g.to_undirected()
    FA_graphs_full[model] = g

# Create the filtered graphs by:
# keeping only WN nodes
# keeping only non-idiosyncratic edges
# taking the LCC
FA_graphs_filt = {}
for model in models:
    g = FA_graphs_full.copy()[model]
    if 'GPT' not in model:
        g = WNfilter(g)
        g = idiosynfilter(g)
        g = CC(g)
        FA_graphs_filt[model] = g
    if 'GPT' in model:
        g = CC(g)
        FA_graphs_filt[model] = g

# Get graph summaries
graph_summary(FA_graphs_full)
graph_summary(FA_graphs_filt)
graph_summary(FA_graphs_full).to_csv(os.path.join(base_dir, 'data/summary_tables/FA_graphs_full_summary.csv'))
graph_summary(FA_graphs_filt).to_csv(os.path.join(base_dir, 'data/summary_tables/FA_graphs_filtered_summary.csv'))

# Get a sample of nodes that were removed from the graphs to see the types of nodes removed
removedNodes = {}
for model in models:
    random.seed(30)
    if 'GPT' not in model:
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
compDF.to_csv(os.path.join(base_dir, 'data/summary_tables/FA_graph_comparisons.csv', index = False))

# Save edge lists
for model, g in FA_graphs_filt.items():
    graph2csv(g, model)

# Convert graphs to igraphs to do the spreading activation
for model, g in FA_graphs_filt.items():
    if 'GPT' not in model:
        nxGraph2igraph(g, name = model + '_ig', directed = False, weighted = True)
    if 'GPT' in model:
        nxGraph2igraph(g, name = model + '_ig', directed = False, weighted = False)
        
        
        