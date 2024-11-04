
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict
import igraph as ig
from igraph import Graph
from nltk.corpus import wordnet as wn
import pickle
import random

########################################################################

# Convert the dataframe to an edgelist
def FA_edgeList(df):
    for column in ['cue', 'R1', 'R2', 'R3']:
        if column == 'cue':
            col = [str(x) for x in df[column]]
        else:
            col = ['' if pd.isna(x) else x for x in df[column]]
        df[column] = col
    # Remove blanks
    df = df[df['cue'] != '']
    df_new = df.copy()
    Edges = list(zip(df_new.cue.values, df_new.R1.values)) +\
        list(zip(df_new.cue.values, df_new.R2.values)) +\
        list(zip(df_new.cue.values, df_new.R3.values))
    Edges = [edge for edge in Edges if '' not in edge]
    Edges = [edge for edge in Edges if edge[0] != edge[1]] # avoid self-loops
    return Edges

# Convert the edgelist to a graph
def graphFromEdgeList(EdgeList, directed = True, weighted = True):
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    W_EdgeDict = defaultdict(int)
    for t in EdgeList:
        if not(pd.isnull(t[1])): # skip edges that have nan nodes
            W_EdgeDict[t] += 1     
    W_EdgeList = [(a, b, c) for (a, b), c in W_EdgeDict.items()]
    if weighted:
        g.add_weighted_edges_from(W_EdgeList)
    else:
        EdgeList = list(set(EdgeList))
        g.add_edges_from(EdgeList)
    return g

# Keep only edges that have a weight of at least 2
def idiosynfilter(g):
    keepEdges = []
    for edge in g.edges():
        if g[edge[0]][edge[1]]['weight'] > 1:
            keepEdges.append(edge)
    g = g.edge_subgraph(keepEdges)
    return g

# Keep only nodes that are words in WordNet
def WNfilter(g):
    keepNodes = []
    for node in g.nodes():
        WNnode = str(node).replace(' ', '_')
        if len(wn.synsets(WNnode)) >= 1:
            keepNodes.append(node)
    g = g.subgraph(keepNodes)
    return g

# Get largest connected component
def CC(g):
    if nx.is_directed(g):
        nodes = list(max(nx.weakly_connected_components(g), key=len))
    else:
        nodes = list(max(nx.connected_components(g), key=len))
    return g.subgraph(nodes)

# Get summary of graphs
def graph_summary(FA_graphs_dictionary):
    df_dict = {}
    for model, g in FA_graphs_dictionary.items():
        d = {'Nodes': len(g.nodes()),
             'Edges': len(g.edges()),
             'Density': nx.density(g),
             'Average degree': np.mean(list(dict(nx.degree(g)).values()))}
        df_dict[model] = d
    df = pd.DataFrame(df_dict).T
    return df

# Save the edges in a CSV
def graph2csv(g, name):
    src = []
    tgt = []
    wt = []
    if nx.is_weighted(g):
        for e in g.edges():
            src.append(e[0])
            tgt.append(e[1])
            wt.append(g.get_edge_data(e[0],e[1])['weight'])
        df = pd.DataFrame({'src': src, 'tgt': tgt, 'wt': wt})
    else:
        for e in g.edges():
            src.append(e[0])
            tgt.append(e[1])
        df = pd.DataFrame({'src': src, 'tgt': tgt})
    df.to_csv('./data/graphs/edge_lists/FA_' + name + '_edgelist.csv', index = False)

# Convert networkx graph to igraph and save
def nxGraph2igraph(g, name, directed = False, weighted = False):
    if directed:
        G_ig = Graph(directed=True)
    else:
        G_ig = Graph(directed=False)

    G_ig.add_vertices(list(g.nodes()))
    G_ig.add_edges([e for e in list(g.edges())])
    G_ig.vs['name'] = list(g.nodes())

    if weighted:
        weights = [g[u][v]['weight'] for u, v in g.edges()]
        G_ig.es['weight'] = weights

    G_ig.write_graphml('./data/graphs/igraphs/FA_' + name + '.graphml')

# Make the graph undirected, taking the max weight at the edge weight
def makeUndirected(g):
    ug = g.to_undirected()
    for node in g:
        for ngbr in nx.neighbors(g, node):
            if node in nx.neighbors(g, ngbr):
                ug.edges[node, ngbr]['weight'] = max(g.edges[node, ngbr]['weight'], g.edges[ngbr, node]['weight'])
    ug.edges.data('weight')
    return ug

# Order the elements of the edges alphabetically
def orderedEdges(edgeList):
    newEdges = []
    for e in edgeList:
        newEdge = tuple(sorted([e[0], e[1]]))
        newEdges.append(newEdge)
    return newEdges

# Compare two graphs, nodes and edges
def netComparison(g1, g2):

    # Node intersection and union
    n1 = set(g1.nodes())
    n2 = set(g2.nodes())
    n1ANDn2 = n1.intersection(n2)
    n1ORn2 = n1.union(n2)
    
    # Subgraphs of the node intersections
    sg1 = g1.subgraph(list(n1ANDn2))
    sg2 = g2.subgraph(list(n1ANDn2))

    # Edges of the subgraphs
    e1 = list(set(sg1.edges()))
    e2 = list(set(sg2.edges()))

    # Ordered edges of the subgraphs
    e1 = set(orderedEdges(e1))
    e2 = set(orderedEdges(e2))
    
    # Edge intersection and union
    e1ANDe2 = e1.intersection(e2)
    e1ORe2 = e1.union(e2)
    
    stats = {}
    stats['(A-B)/A Nodes'] = len(n1 - n2)/len(n1)
    stats['Jaccard Nodes'] = len(n1ANDn2)/len(n1ORn2)
    stats['(B-A)/B Nodes'] = len(n2 - n1)/len(n2)
    
    stats['(A-B)/A Edges'] = len(e1 - e2)/len(e1)
    stats['Jaccard Edges'] = len(e1ANDe2)/len(e1ORe2)
    stats['(B-A)/B Edges'] = len(e2 - e1)/len(e2)
    
    return stats

###############################################################################

# Load data
models = ['Humans', 'Mistral', 'Llama3', 'Haiku', 'GPT-3.5 Turbo', 'GPT-4o', 'GPT-4 Turbo']
FA_clean_dfs = {}
for model in models:
    FA_clean_dfs[model] = pd.read_csv('./data/output/FA_cleaned_dfs/FA_' + model + '.csv')

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
pickle.dump(FA_graphs, open('./data/graphs/FA_graphs.pickle', 'wb'))

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
graph_summary(FA_graphs_full).to_csv('./data/summary_tables/FA_graphs_full_summary.csv')
graph_summary(FA_graphs_filt).to_csv('./data/summary_tables/FA_graphs_filtered_summary.csv')

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

compDict2 = {'Comparison':
['Percentage of Human nodes not in LLM',
'Percentage of all nodes common to both',
'Percentage of LLM nodes not in Human',
'Percentage of Human edges not in LLM',
'Percentage of all edges common to both',
'Percentage of LLM edges not in Human']}

for model in ['Mistral', 'Llama3', 'Haiku']:
    compDict2[model] = list(compDict[model].values())
compDF = pd.DataFrame(compDict2)
compDF.to_csv('./data/summary_tables/FA_graph_comparisons.csv', index = False)

# Save edge lists
for model, g in FA_graphs_filt.items():
    graph2csv(g, model)

# Convert graphs to igraphs to do the spreading activation
for model, g in FA_graphs_filt.items():
    if 'GPT' not in model:
        nxGraph2igraph(g, name = model + '_ig', directed = False, weighted = True)
    if 'GPT' in model:
        nxGraph2igraph(g, name = model + '_ig', directed = False, weighted = False)