
import numpy as np
import pandas as pd
import json
import random
import collections
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from tqdm import tqdm
import networkx as nx
from collections import defaultdict
from igraph import Graph
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
import seaborn as sns
from matplotlib.lines import Line2D

###############################################################################
# FA_dataGeneration.py

def loadSimplifiedSWOW():
    df_orig = pd.read_csv('./data/original_datasets/SWOW-EN.R100.csv')
    countries = pd.read_csv('./data/mapping_tables/country.csv')
    df = df_orig.copy()
    # Gender
    df = df.replace({'gender': {'Fe': 'Female', 'Ma': 'Male', 'X': 'Unknown'}})
    # Education
    df.education = df.education.fillna('Unknown')
    df = df.replace({'education':
                        {1.0: 'No education',
                         2.0: 'Elementary school',
                         3.0: 'High school',
                         4.0: 'Bachelor degree',
                         5.0: 'Master degree'}})
    # Native Language
    eng = ['Australia',
           'Canada',
           'Ireland',
           'New Zealand',
           'United Kingdom',
           'United States',
           'Other_English']
    df['nativeLanguage'] = np.where(df['nativeLanguage'].isin(eng), 'English', 'Not English')
    # Country
    df['country'] = np.where(df['country'].isin(countries.value.values), df['country'], 'Unknown')
    # Cue (convert to strings)
    df['cue'] = [str(x) for x in df.cue.values]
    # Responses (convert NANs to blanks)
    df.R1 = df.R1.fillna('')
    df.R2 = df.R2.fillna('')
    df.R3 = df.R3.fillna('')
    # Choose only the variables we need
    df = df[['age','gender','nativeLanguage','country','education','cue','R1','R2','R3']]
    # Sort
    df_sorted = df.sort_values(by = ['cue','age','gender','nativeLanguage','country','education'])
    return df_sorted

# FA_dataGeneration.py
# FA_dataCleaning.py 
def getProfilesCues(df):
    # Group by age, native language, gender, education, country
    df_agg = df.groupby(['age', 'nativeLanguage', 'gender', 'education', 'country'])
    # Make a list profiles and cues as input for the LLMs
    profiles_cues = []
    for group, data in df_agg:
        # cues
        cue_list = [str(x) for x in list(data.cue.values)]
        # dictionary with profile data
        prof_dict = {'age': str(group[0]),
                        'native language': str(group[1]),
                        'gender': str(group[2]),
                        'highest level of education': str(group[3]),
                        'country': str(group[4])}
        # list of tuples (profile dictionary, cue list)
        profiles_cues.append((prof_dict, cue_list))
    return profiles_cues

###############################################################################
# FA_dataCleaning.py

def loadData(filename):
    with open(filename, 'r') as json_file:
        json_list = list(json_file)
    output = []
    for json_str in json_list:
        result = json.loads(json_str)
        output.append(result)
    return output

# FA_dataCleaning.py
def loadTextFile(filename):
    file = open(filename, 'r')
    content = file.readlines()
    output = []
    for line in content:
        line = line.strip()
        if '\t' in line:
            line = line.split('\t')
        output.append(line)
    file.close()
    return output

# Put FA data into dataframe
def FA_df(list_of_dictionaries, profiles = False):
    if profiles:
        dataDict = dict(zip(['age','gender','nativeLanguage','country','education','cue','R1','R2','R3'], [[] for _ in range(9)]))
    else:
        dataDict = dict(zip(['cue','R1','R2','R3'], [[] for _ in range(4)]))
    respCols = {'R1': 0,'R2': 1,'R3': 2}
    for d in tqdm(list_of_dictionaries):
        try:
            dataDict['cue'].append(d['cue'].lower())
        except:
            dataDict['cue'].append('')
        for resp in respCols.keys():
            try:
                dataDict[resp].append(d['response'][respCols.get(resp)])
            except:
                dataDict[resp].append('')
        if profiles:    
            dataDict['age'].append(d['age'])
            dataDict['gender'].append(d['gender'])
            dataDict['nativeLanguage'].append(d['native language'])
            dataDict['country'].append(d['country'])
            dataDict['education'].append(d['education'])
    df = pd.DataFrame(dataDict)
    df['cue'] = [str(x) for x in df['cue']]
    return df

# Get 100 rows per cue
def cue100(df1, unqCues):
    random.seed(30)
    df1 = df1[df1['cue'].isin(list(unqCues))] # Only keep cues in orignal
    c = list(df1['cue'])
    cCount = collections.Counter(c)
    over100 = {}
    under100 = {}
    for key, value in cCount.items():
        if value > 100:
            over100[key] = value
        if value < 100:
            under100[key] = value
    df = df1.copy()
    # Remove rows for cues that appear more than 100 times
    if len(over100) > 0:
        rows_to_remove = []
        for c, count in tqdm(over100.items()):
            dfCue = df1[df1['cue'] == c]
            surplus = count - 100
            rows_to_remove.append(random.sample(dfCue.index.tolist(), surplus))
        rows_to_remove = [item for sublist in rows_to_remove for item in sublist]
        df = df.drop(rows_to_remove)

    # Add rows for cues that appear less than 100 times
    if len(under100) > 0:
        for c, count in tqdm(under100.items()):
            deficit = 100 - count
            rows_to_add = pd.DataFrame(zip([c] * deficit,
                                            [''] * deficit,
                                            [''] * deficit,
                                            [''] * deficit),
                                        columns = ['cue', 'R1', 'R2', 'R3'])
            df = pd.concat([df, rows_to_add], ignore_index=True)

    # Add rows for cues that are completely missing from the dataset
    missingCues = list(set(unqCues) - set(df['cue']))
    if len(missingCues) > 0:
        for c in missingCues:
            rows_to_add = pd.DataFrame(zip([c] * 100,
                                            [''] * 100,
                                            [''] * 100,
                                            [''] * 100),
                                        columns = ['cue', 'R1', 'R2', 'R3'])
            df = pd.concat([df, rows_to_add], ignore_index=True)
    return df

# Converts NAs to blanks
def NA2Blank(df1):
    df = df1.copy()
    for col in ['cue', 'R1', 'R2', 'R3']:
        df[col] = [x if isinstance(x, str) else '' for x in df[col]]
    return df

# Makes everything lowercase
def Lowercase(df1):
    df = df1.copy()
    for col in ['cue', 'R1', 'R2', 'R3']:
        df[col] = [x.lower() for x in df[col]]
    return df

# Removes underscores
def RemoveUnderscore(df1):
    df = df1.copy()
    for col in ['cue', 'R1', 'R2', 'R3']:
        df[col] = [x.replace('_', ' ') for x in df[col]]
    return df

# Removes: [a, an, the, to] unless it is present in the cue words (a lot)
def RemoveRespArticles(df1, unqCues):
    df = df1.copy()
    for col in ['R1', 'R2', 'R3']:
        for prefix in ['a ', 'an ', 'the ', 'to ']:
            mask = (df[col].str.startswith(prefix)) & (~df[col].isin(unqCues))
            df.loc[mask, col] = df.loc[mask, col].str[len(prefix):]
    return df

# Add spaces or hyphens when one is missing
def AddSpaceOrHyphen(df1, missingDict):
    df = df1.copy()
    for col in ['cue', 'R1', 'R2', 'R3']:
        df[col] = df[col].map(missingDict).fillna(df[col])
    return df

# Correct spelling
def Spelling(df1, spelling_dict):
    df = df1.copy()
    for col in ['cue', 'R1', 'R2', 'R3']:
        df[col] = df[col].map(spelling_dict).fillna(df[col])
    return df

# Lemmatize and make some manual corrections
def Lemmatization(df1):
    df = df1.copy()
    for col in ['cue', 'R1', 'R2', 'R3']:
        df[col] = [lemmatizer.lemmatize(x) for x in df[col]]
        df[col] = [x.replace('men', 'man') for x in df[col]]
        df[col] = [x.replace('hands', 'hand') for x in df[col]]
    return df

# Remove responses that are equal to their cues
def RemoveCueResp(df1):
    df = df1.copy()
    for col in ['R1', 'R2', 'R3']:
        df[col] = np.where(df[col] == df['cue'], '', df[col])
    return df

# Remove duplicate responses
def RemoveDupeResp(df1):
    df = df1.copy()
    # if R3 is equal to R1 or R2, remove it
    df['R3'] = np.where((df['R3'] == df['R1']) | (df['R3'] == df['R2']), '', df['R3'])
    # if R2 is equal to R1, remove it
    df['R2'] = np.where(df['R2'] == df['R1'], '', df['R2'])
    return df

# Change the order of the responses so responses are on the left and blanks on the right
def ShiftResp(df1):
    df = df1.copy()
    
    # _ _ X becomes X _ _
    df['R1'] = np.where((df['R1'] == '') & (df['R2'] == '') & (df['R3'] != ''), df['R3'], df['R1'])
    df['R3'] = np.where(df['R1'] == df['R3'], '', df['R3'])
    
    # _ X _ becomes X _ _
    df['R1'] = np.where((df['R1'] == '') & (df['R2'] != '') & (df['R3'] == ''), df['R2'], df['R1'])
    df['R2'] = np.where(df['R1'] == df['R2'], '', df['R2'])
    
    # _ X X becomes X _ X
    df['R1'] = np.where((df['R1'] == '') & (df['R2'] != '') & (df['R3'] != ''), df['R2'], df['R1'])
    df['R2'] = np.where(df['R1'] == df['R2'], '', df['R2'])
    
    # X _ X becomes X X _
    df['R2'] = np.where((df['R1'] != '') & (df['R2'] == '') & (df['R3'] != ''), df['R3'], df['R2'])
    df['R3'] = np.where(df['R2'] == df['R3'], '', df['R3'])
    
    return df

# Sort the columns alphabetically
def SortColumns(df1):
    df = df1.copy()
    df = df[['cue', 'R1', 'R2', 'R3']]
    df = df.sort_values(by = ['cue','R1','R2','R3'])
    return df

# Put the whole cleaning pipeline together
def cleaningPipeline(df1, unqCues, missingDict, spelling_dict, name):
    df = df1.copy()
    df = NA2Blank(df) # make the NA responses into blanks    
    df = Lowercase(df) # make everything lowercase
    df = RemoveUnderscore(df)
    df = RemoveRespArticles(df, unqCues) # remove articles from responses if not a cue
    df = AddSpaceOrHyphen(df, missingDict) # Add spaces or hyphen when one is missing
    df = Spelling(df, spelling_dict) # correct spelling of cues and responses if in spelling dictionary
    df = Lemmatization(df) # lemmatize all cues and responses
    df = cue100(df, unqCues) # align data to get 100 sets of responses per cue
    df = RemoveCueResp(df) # remove responses that are equal to cues
    df = RemoveDupeResp(df) # remove duplicate responses
    df = ShiftResp(df) # align dataframe so all response are to the left
    df = SortColumns(df)
    return df

###############################################################################
# FA_buildNetworks.py

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
# FA_analyses_LDT_Gender.py

def edgelist2graph(path, directed = False):
    edge_list = pd.read_csv(path)
    edge_tuples = zip(edge_list['src'], edge_list['tgt'], edge_list['wt'])
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    g.add_weighted_edges_from(edge_tuples)
    return g

def normalizeDF(df, normalize_rows = False):
    cols = list(df.columns)[1:]
    df_norm = df.copy().drop(['node'], axis = 1)
    df_norm = preprocessing.normalize(df_norm, axis = 0)
    if normalize_rows:
        df_norm = preprocessing.normalize(df_norm, axis = 1)
    df_norm = pd.DataFrame(df_norm)
    df_norm.columns = cols
    df_norm.index = list(df['node'])
    return df_norm

def activationDict(df_norm, triplets):
    tgt_rel_un_act = {}
    for trip in triplets:
        rel = float(df_norm[trip[1]].loc[trip[0]])
        un = float(df_norm[trip[2]].loc[trip[0]])
        diff = rel - un
        tgt_rel_un_act[trip] = (rel, un, diff)
    return tgt_rel_un_act

def matricesGender(df_norm, rows, primes_F, primes_M):
    mat = df_norm[primes_F + primes_M].loc[rows]
    return mat

def boxplotRTs(dictionary):
    data1 = [x[0] for x in list(dictionary.values())]
    data2 =  [x[1] for x in list(dictionary.values())]
    data = [data1, data2]
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, patch_artist=True, notch=True, 
                boxprops=dict(facecolor='lightgrey', color='black'),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(markerfacecolor='black', markeredgecolor='black', marker='o'),
                showfliers = False)
    plt.xticks([1, 2], ['Related', 'Unrelated'], fontsize = 14)
    plt.ylabel('Mean z-scored reaction time', fontsize=16)
    plt.xlabel('Prime type', fontsize=16)
    plt.title('Reaction times of 50 targets from the LDT dataset by prime type', ha='center', fontsize=18, weight='bold')
    plt.show()
    print(stats.wilcoxon(data1, data2, alternative = 'less'))

def boxplotsLDT(LDT_AL_dicts, modelColors):
    actData = []
    modelData = []
    primeData = []
    for model, d in LDT_AL_dicts.items():
        ALdict = d
        data1 = [x[0] for x in list(ALdict.values())]
        data2 =  [x[1] for x in list(ALdict.values())]
        actData.append(data1)
        actData.append(data2)
        modelData.append([model] * 100)
        primeData.append([model + 'RP'] * 50)
        primeData.append([model + 'UP'] * 50)
    data = {
        'Model': np.concatenate(modelData),
        'Prime Type': np.concatenate(primeData),
        'Normalized activation level': np.concatenate(actData)}
    df = pd.DataFrame(data)
    group_colors = {}
    for model, color in modelColors.items():
        group_colors[model] = color
    df['Color'] = df['Model'].map(group_colors)
    plt.figure(figsize=(12, 4))
    sns.boxplot(
        x='Prime Type',
        y='Normalized activation level',
        hue='Model',
        data=df,
        notch = True,
        palette=df['Color'].unique(),
        dodge=True,
        showfliers=False)
    plt.title('Activation levels of 50 targets from the LDT dataset by prime type ', ha='center', fontsize=18, weight='bold')
    plt.ylabel('Normalized activation level', fontsize=16)
    plt.xlabel('Prime type', fontsize=16,)
    plt.legend(title='Prime Type')
    custom_xtick_labels = [
        'Related', 'Unrelated',
        'Related', 'Unrelated',
        'Related', 'Unrelated',
        'Related', 'Unrelated']
    plt.xlim(-1, len(custom_xtick_labels))
    tick_positions = [-.3,.7,1.9,2.9,4.1,5.1,6.3,7.3]
    plt.xticks(ticks=tick_positions, labels=custom_xtick_labels, rotation=45, fontsize = 14)
    plt.axvline(1.25, color='gray', linewidth=1)
    plt.axvline(3.5, color='gray', linewidth=1)
    plt.axvline(5.75, color='gray', linewidth=1)
    legend_elements = [
        Line2D([0], [0], color=color, lw=4, label=group) for group, color in group_colors.items()]
    plt.legend(handles=legend_elements, title='Network', loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

def boxplotsGender(matDict, targetType, primes_F, primes_M, modelColors, main = False,):
    actData = []
    modelData = []
    primeData = []
    for model, mat in matDict.items():
        matF = mat[targetType]
        data1 = matF[primes_F].values.flatten()
        data2 = matF[primes_M].values.flatten()
        actData.append(data1)
        actData.append(data2)
        modelData.append([model] * 250)
        primeData.append([model + 'P1'] * 125)
        primeData.append([model + 'P2'] * 125)
    data = {
        'Model': np.concatenate(modelData),
        'Prime Type': np.concatenate(primeData),
        'Normalized activation level': np.concatenate(actData)}
    df = pd.DataFrame(data)
    group_colors = {}
    for model, color in modelColors.items():
        group_colors[model] = color
    df['Color'] = df['Model'].map(group_colors)
    plt.figure(figsize=(12, 4))
    if main:
        plt.figtext(0.5, 1, 'Activation levels of 50 gender-related targets by prime type', ha='center', fontsize=20, weight='bold')
    sns.boxplot(
        x='Prime Type',
        y='Normalized activation level',
        hue='Model',
        data=df,
        notch = True,
        palette=df['Color'].unique(),
        dodge=True,
        showfliers=False)
    plt.title(targetType, fontsize=18)
    plt.ylabel('Normalized activation level', fontsize=16)
    plt.xlabel('Prime type', fontsize = 16)
    plt.legend(title='Prime type')
    custom_xtick_labels = [
        'Female-related', 'Male-related',
        'Female-related', 'Male-related',
        'Female-related', 'Male-related',
        'Female-related', 'Male-related']
    plt.xlim(-1, len(custom_xtick_labels))
    tick_positions = [-.3,.7,1.9,2.9,4.1,5.1,6.3,7.3]
    plt.xticks(ticks=tick_positions, labels=custom_xtick_labels, rotation=45, fontsize = 14)
    plt.axvline(1.25, color='gray', linewidth=1)
    plt.axvline(3.5, color='gray', linewidth=1)
    plt.axvline(5.75, color='gray', linewidth=1)
    legend_elements = [
        Line2D([0], [0], color=color, lw=4, label=group) for group, color in group_colors.items()]
    plt.legend(handles=legend_elements, title='Network', loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

def heatmapsGender(dfF, dfM, model, modelColorPalettes):
    fig, axes = plt.subplots(1,2, figsize=(18, 10))
    plt.suptitle(model, fontsize=24, y = 1)
    plt.figtext(0.5, .93, 'Activation levels of 50 gender-related targets by female-related and male-related primes', ha='center', fontsize=20, weight='bold')
    ylab = 'Targets'
    xlab = 'Primes'
    heatF = dfF
    cols = list(heatF.columns)
    rows = list(heatF.index)
    plt.figure(figsize=(6, 3))
    sns.heatmap(heatF, annot=False, cmap=modelColorPalettes[model], ax=axes[0], vmin=0, vmax=1)
    axes[0].set_title('Female-related targets', fontsize=18)
    axes[0].set_xticklabels(cols, fontsize=16, rotation = 90)
    axes[0].set_yticklabels(rows, fontsize=16)
    axes[0].set_ylabel(ylab, fontsize=18)
    axes[0].set_xlabel(xlab, fontsize=18)

    heatM = dfM
    cols = list(heatM.columns)
    rows = list(heatM.index)
    plt.figure(figsize=(6, 3))
    sns.heatmap(heatM, annot=False, cmap=modelColorPalettes[model], ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title('Male-related targets', fontsize=18)
    axes[1].set_xticklabels(cols, fontsize=16, rotation = 90)
    axes[1].set_yticklabels(rows, fontsize=16)
    axes[1].set_xlabel(xlab, fontsize=18)
    plt.subplots_adjust(wspace=2)
    plt.show()
