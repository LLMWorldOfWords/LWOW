
import json
import networkx as nx
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy import stats
from tqdm import tqdm
from sklearn import preprocessing
from nltk.corpus import wordnet as wn
import itertools
import seaborn as sns
from functools import reduce
from matplotlib.ticker import ScalarFormatter

###################################################################

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

def makeUndirected(g):
    ug = g.to_undirected()
    for node in g:
        for ngbr in nx.neighbors(g, node):
            if node in nx.neighbors(g, ngbr):
                ug.edges[node, ngbr]['weight'] = max(g.edges[node, ngbr]['weight'], g.edges[ngbr, node]['weight'])
    ug.edges.data('weight')
    return ug

def genderTriplets(df, words, genderPrimePairs):
    triplets = list(itertools.product(words, genderPrimePairs))
    triplets = [(x[0], x[1][0], x[1][1]) for x in triplets if ((x[0] != x[1][0]) and (x[0] != x[1][1]))]
    return triplets

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

def boxplotRTs(dictionary):
    data1 = [x[0] for x in list(dictionary.values())]
    data2 =  [x[1] for x in list(dictionary.values())]
    data = [data1, data2]
    xlab = ''
    plt.figure(figsize=(8, 7))
    plt.figtext(0.5, .95, 'Reaction times of 50 targets from the LDT dataset', ha='center', fontsize=20, weight='bold')
    plt.boxplot(data, patch_artist=True, notch=True, 
                boxprops=dict(facecolor='lightgrey', color='lightgrey'),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(markerfacecolor='black', markeredgecolor='black', marker='o'),
                showfliers = False)
    plt.xticks([1, 2], ['Related prime', 'Unrelated prime'], fontsize=16)
    plt.ylabel('Mean z-scored reaction time', fontsize=16)
    plt.title('Reaction time by prime type', fontsize=16)
    plt.show()
    print(stats.wilcoxon(data1, data2, alternative = 'less'))

def boxplotsLDT(ALdicts):

    fig, axes = plt.subplots(2,2, figsize=(14, 12))
    plt.figtext(0.5, .95, 'Activation levels by prime type of 50 targets from the LDT dataset', ha='center', fontsize=20, weight='bold')
    for i, (model, d) in enumerate(ALdicts.items()):
        ALdict = d
        categories = ['Related prime', 'Unrelated prime']
        data1 = [x[0] for x in list(ALdict.values())]
        data2 =  [x[1] for x in list(ALdict.values())]
        data = [data1, data2]
        plt.figure(figsize=(4, 5))
        if i == 0:
            a = 0
            b = 0
        if i == 1:
            a = 0
            b = 1
        if i == 2:
            a = 1
            b = 0
        if i == 3:
            a = 1
            b = 1
        boxplot = axes[a,b].boxplot(data, patch_artist=True, notch=True, 
                    boxprops=dict(facecolor=modelColors[model], color=modelColors[model]),
                    medianprops=dict(color='black'),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    flierprops=dict(markerfacecolor='black', markeredgecolor='black', marker='o'),
                    showfliers = False)
        axes[a,b].set_title(model, fontsize=24)
        axes[a,b].set_ylim(0, 1)
        axes[a,b].set_xticks([1, 2])
        axes[a,b].set_xticklabels(categories, fontsize=16)
        axes[a,b].set_ylabel('Normalized activation level', fontsize=16)
        print(stats.wilcoxon(data1, data2, alternative = 'greater'))

def boxplotsGender(dictionaryF, dictionaryM):

    fig, axes = plt.subplots(1,2, figsize=(16, 6))
    categories = ['Female-related prime', 'Male-related prime']
    ylab = 'Normalized activation level'
    dataF1 = [x[0] for x in list(dictionaryF.values())]
    dataF2 =  [x[1] for x in list(dictionaryF.values())]
    dataF = [dataF1, dataF2]
    plt.figure(figsize=(6, 3))
    boxplot = axes[0].boxplot(dataF, patch_artist=True, notch=True, 
                boxprops=dict(facecolor=modelColors[model], color=modelColors[model]),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(markerfacecolor='black', markeredgecolor='black', marker='o'),
                showfliers = False)
    axes[0].set_title('Activation of female-related targets by prime type', fontsize=18)
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(categories, fontsize=16)
    axes[0].set_ylabel(ylab, fontsize=16)
    print(stats.wilcoxon(dataF1, dataF2, alternative = 'greater'))

    dataM1 = [x[0] for x in list(dictionaryM.values())]
    dataM2 =  [x[1] for x in list(dictionaryM.values())]
    dataM = [dataM1, dataM2]
    plt.figure(figsize=(6, 3))
    boxplot = axes[1].boxplot(dataM, patch_artist=True, notch=True, 
                boxprops=dict(facecolor=modelColors[model], color=modelColors[model]),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(markerfacecolor='black', markeredgecolor='black', marker='o'),
                showfliers = False)
    axes[1].set_title('Activation of male-related targets by prime type', fontsize=18)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(categories, fontsize=16)
    print(stats.wilcoxon(dataM1, dataM2, alternative = 'less'))
    plt.subplots_adjust(wspace=2)
    plt.show()

def heatmapsGender(df_norm, rowsF, rowsM):

    fig, axes = plt.subplots(1,2, figsize=(18, 10))
    plt.suptitle(model, fontsize=24, y = 1)
    plt.figtext(0.5, .93, 'Activation levels of 50 stereotypical targets by female-related and male-related primes', ha='center', fontsize=20, weight='bold')
    ylab = 'Targets'
    xlab = 'Primes'
    heatF = df_norm[primes_F + primes_M].loc[rowsF]
    plt.figure(figsize=(6, 3))
    sns.heatmap(heatF, annot=False, cmap=modelColorPalettes[model], ax=axes[0], vmin=0, vmax=1)
    axes[0].set_title('Female-related targets', fontsize=18)
    axes[0].set_xticklabels(primes_F + primes_M, fontsize=16, rotation = 90)
    axes[0].set_yticklabels(rowsF, fontsize=16)
    axes[0].set_ylabel(ylab, fontsize=16)
    axes[0].set_xlabel(xlab, fontsize=16)

    heatM = df_norm[primes_F + primes_M].loc[rowsM]
    plt.figure(figsize=(6, 3))
    sns.heatmap(heatM, annot=False, cmap=modelColorPalettes[model], ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title('Male-related targets', fontsize=18)
    axes[1].set_xticklabels(primes_F + primes_M, fontsize=16, rotation = 90)
    axes[1].set_yticklabels(rowsM, fontsize=16)
    axes[1].set_xlabel(xlab, fontsize=16)
    plt.subplots_adjust(wspace=2)
    plt.show()

modelColors = {'Humans': '#3269AF',
               'Mistral': '#F19542',
               'Llama3': '#73BC6B',
               'Haiku': '#6A54A6',
               'GPT-3.5 Turbo': '#DC3425',
               'GPT-4o': '#DC3425',
               'GPT-4 Turbo': '#DC3425'}

modelColorPalettes = {'Humans': 'Blues',
                      'Mistral': 'Oranges',
                      'Llama3': 'Greens',
                      'Haiku': 'Purples',
                      'GPT-3.5 Turbo': 'Reds',
                      'GPT-4o': 'Reds',
                      'GPT-4 Turbo': 'Reds'}

########################################################################################

# These graphs are weighted and directed, and they are NOT the CCs
FA_graphs = pickle.load(open('./data/graphs/FA_graphs.pickle', 'rb'))
models = ['Humans', 'Mistral', 'Llama3', 'Haiku', 'GPT-3.5 Turbo', 'GPT-4o', 'GPT-4 Turbo']

# Create full graphs but undirected versions
FA_graphs_full = {}
for model, g in FA_graphs.items():
    if 'GPT' not in model:
        g = makeUndirected(g)
    if 'GPT' in model:
        g = g.to_undirected()
    FA_graphs_full[model] = g

# Create the filtered graphs by applying the WN nodes and non-idiosyncratic edge filters
# Finally take the LCC
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

##################################################################

# Load and clean LDT dataset
LDT = pd.read_csv('./data/LDT_analyses/primingLDT_data.csv')
LDT = LDT[LDT['rel'].isin(['un', 'rel'])]
LDT = LDT[LDT['keep'] == 1]
LDT = LDT[['prime', 'target', 'rel', 'target.RT', 'Ztarget.RT']]
LDT['prime'] = [str(x.lower()) for x in LDT['prime']]
LDT['target'] = [str(x.lower()) for x in LDT['target']]

# Get all triplets
grouped = LDT.groupby(['target'])
tgt_rel_un = {}
for tgt, data in grouped:
    rel_prime = data[data['rel'] == 'rel']
    rel_primes = {}
    for prime, d in rel_prime.groupby(['prime']):
        rel_z = np.average([float(x) for x in list(d['Ztarget.RT'])])
        rel_primes[prime[0]] = rel_z
    un_primes = {}
    un_prime = data[data['rel'] == 'un']
    for prime, d in un_prime.groupby(['prime']):
        un_z =  np.average([float(x) for x in list(d['Ztarget.RT'])])
        un_primes[prime[0]] = un_z
    combos = itertools.product(list(rel_primes.keys()), list(un_primes.keys()))
    for combo in combos:
        tgt_rel_un[tgt[0], combo[0], combo[1]] = [rel_primes[combo[0]], un_primes[combo[1]], un_primes[combo[1]] - rel_primes[combo[0]]]
tgt_rel_un_UNQ = {}
for trip, dat in tgt_rel_un.items():
    if trip[0] not in tgt_rel_un_UNQ.keys():
        tgt_rel_un_UNQ[trip[0]] = (trip, dat)
    else:
        if dat[2] > tgt_rel_un_UNQ[trip[0]][1][2]:
            tgt_rel_un_UNQ[trip[0]] = (trip, dat)
tgt_rel_un_UNQ = {x[0]:x[1] for x in list(tgt_rel_un_UNQ.values())}

# Find node intersection of graphs
all_nodes = []
for g in FA_graphs_filt.values():
    all_nodes.append(list(g.nodes()))
nodeIntAll = list(reduce(lambda x, y: set(x) & set(y), all_nodes))

# Find node intersection of graphs excluding GPT graphs
all_nodes_full = []
for model, g in FA_graphs_filt.items():
    if 'GPT' not in model:
        all_nodes_full.append(list(g.nodes()))
nodeIntFull = list(reduce(lambda x, y: set(x) & set(y), all_nodes_full))

# Find the LDT triplets in all graphs
keepKeys = []
for trip in tgt_rel_un_UNQ.keys():
    if (trip[0] in nodeIntAll and trip[1] in nodeIntAll and trip[2] in nodeIntAll):
        keepKeys.append(trip)
tgt_rel_un_keep = {k:tgt_rel_un_UNQ[k] for k in keepKeys}

# Of those triplets in the graphs, find those with largest effects
diffs = dict(sorted(tgt_rel_un_keep.items(), key=lambda item: item[1][2], reverse=True))
LDT_50_triplets = list(diffs.keys())[:50]
LDT_RT_dict = {k:tgt_rel_un_keep[k] for k in LDT_50_triplets}

# Get the primes and save them
tgts = []
rels = []
uns = []
for triplet in LDT_RT_dict:
    tgts.append(triplet[0])
    rels.append(triplet[1])
    uns.append(triplet[2])
primes = list(set(rels).union(set(uns)))
pd.DataFrame({'prime':primes}).to_csv('./data/LDT_analyses/primes.csv', index = False)

# Get the 50 triplets and save them
LDT_50_triplets_DF = pd.DataFrame({'Target': [x[0] for x in LDT_50_triplets],
'Related Prime': [x[1] for x in LDT_50_triplets],
'Unrelated Primes': [x[2] for x in LDT_50_triplets],
'Target-Related RT': [x[0] for x in LDT_RT_dict.values()],
'Target-Unelated RT': [x[1] for x in LDT_RT_dict.values()]})
LDT_50_triplets_DF.to_csv('./data/LDT_analyses/LDT_50_triplets.csv', index = False)

# Make list of bias related words
genderWords = ['woman', 'man',
             'girl', 'boy',
             'mother', 'father',
             'female', 'male',
             'feminine', 'masculine'
             ]
for word in genderWords:
    if word not in nodeIntAll:
        print(word)
pd.DataFrame({'genderWord':genderWords}).to_csv('./data/LDT_analyses/genderWords.csv', index = False)

###########################################################################

# Load results
primeDFs = {}
genderDFs = {}
for model in models:
    if 'GPT' not in model:
        primeDFs[model] = pd.read_csv('./data/LDT_analyses/FA_matrices/' + model + '_primes.csv')
        genderDFs[model] = pd.read_csv('./data/LDT_analyses/FA_matrices/' + model + '_gender.csv')

# Get the prime pairs
genderPrimePairs = [('woman', 'man'),
                    ('female', 'male'),
                    ('mother', 'father'),
                    ('girl', 'boy'),
                    ('feminine', 'masculine')
                    ]   
primes_F = [x[0] for x in genderPrimePairs]
primes_M = [x[1] for x in genderPrimePairs]

# Load the gender targets
targets = pd.read_csv('./data/LDT_analyses/gender_adj.csv')
tgts_F = [x.lower() for x in list(targets['Female']) if x.lower() in nodeIntFull]
tgts_M = [x.lower() for x in list(targets['Male'])if x.lower() in nodeIntFull]


#####################################

# For original LDT data analyses
# Reaction time boxplots
boxplotRTs(LDT_RT_dict)

# Activation level boxplots
LDT_AL_dicts = {}
for model, g in FA_graphs_filt.items():
    if 'GPT' not in model:
        prime_df = primeDFs[model]
        LDT_df_norm = normalizeDF(prime_df, normalize_rows = True)
        LDT_AL_dict = activationDict(LDT_df_norm, LDT_50_triplets)
        LDT_AL_dicts[model] = LDT_AL_dict
boxplotsLDT(LDT_AL_dicts)

# For gender data analyses
mat = {}
mat_f1 = {}
mat_m1 = {}
mat_f2f = {}
mat_m2f = {}
mat_f2m = {}
mat_m2m = {}
mat2_diffF = {}
mat2_diffM = {}
for model, g in FA_graphs_filt.items():
    if 'GPT' not in model:
        gender_df = genderDFs[model]
        gender_df_norm = normalizeDF(gender_df, normalize_rows = True)
        # All adjectives
        mat[model] = gender_df_norm.loc[tgts_F + tgts_M].to_numpy()
        # Female adjectives
        mat_f1[model] = gender_df_norm.loc[tgts_F].to_numpy()
        # Male adjectives
        mat_m1[model] = gender_df_norm.loc[tgts_M].to_numpy()
        # Female primes female adjectives
        mat_f2f[model] = gender_df_norm[primes_F].loc[tgts_F].to_numpy()
        # Male primes female adjectives
        mat_m2f[model] = gender_df_norm[primes_M].loc[tgts_F].to_numpy()
        # Female primes male adjectives
        mat_f2m[model] = gender_df_norm[primes_F].loc[tgts_M].to_numpy()
        # Male primes male adjectives
        mat_m2m[model] = gender_df_norm[primes_M].loc[tgts_M].to_numpy()

        # Female primes vs. male primes 
        from numpy.linalg import norm
        A = mat_f2f[model]
        B = mat_m2f[model]
        A_flat = A.flatten()
        B_flat = B.flatten()
        cosine_similarity = np.dot(A_flat, B_flat) / (norm(A_flat) * norm(B_flat))
        mat2_diffF[model] = cosine_similarity

        # Female primes vs. male primes 
        from numpy.linalg import norm
        A = mat_f2m[model]
        B = mat_m2m[model]
        A_flat = A.flatten()
        B_flat = B.flatten()
        cosine_similarity = np.dot(A_flat, B_flat) / (norm(A_flat) * norm(B_flat))
        mat2_diffM[model] = cosine_similarity

        # For the boxplots
        genderTriplets_F = genderTriplets(gender_df, tgts_F, genderPrimePairs)
        genderTriplets_M = genderTriplets(gender_df, tgts_M, genderPrimePairs)
        gender_dict_F = activationDict(gender_df_norm, genderTriplets_F)
        gender_dict_M = activationDict(gender_df_norm, genderTriplets_M)
        
        # Plots
        heatmapsGender(gender_df_norm, tgts_F, tgts_M)
        boxplotsGender(gender_dict_F, gender_dict_M)

list(itertools.combinations(mat.values(), 2))
modelCombos = list(itertools.combinations(mat.keys(), 2))
mat1_diffF = {}
mat1_diffM = {}
mat1_diff = {}
for combo in modelCombos:

    A = mat_f1[combo[0]]
    B = mat_f1[combo[1]]
    A_flat = A.flatten()
    B_flat = B.flatten()
    cosine_similarity = np.dot(A_flat, B_flat) / (norm(A_flat) * norm(B_flat))
    mat1_diffF[combo] = cosine_similarity

    A = mat_m1[combo[0]]
    B = mat_m1[combo[1]]
    A_flat = A.flatten()
    B_flat = B.flatten()
    cosine_similarity = np.dot(A_flat, B_flat) / (norm(A_flat) * norm(B_flat))
    mat1_diffM[combo] = cosine_similarity

    A = mat[combo[0]]
    B = mat[combo[1]]
    A_flat = A.flatten()
    B_flat = B.flatten()
    cosine_similarity = np.dot(A_flat, B_flat) / (norm(A_flat) * norm(B_flat))
    mat1_diff[combo] = cosine_similarity

# Between model comparisons
mat1_diffs = pd.DataFrame([list(mat1_diffF.values()), list(mat1_diffM.values())]).T
mat1_diffs.columns = ['Female-related targets', 'Male-related targets']
mat1_diffs.index = [combo[0] + ' vs. ' + combo[1] for combo in modelCombos]

# Within model comparisons
mat2_diffs = pd.DataFrame([list(mat2_diffF.values()), list(mat2_diffM.values())]).T
mat2_diffs.columns = ['Female-related targets', 'Male-related targets']
mat2_diffs.index = models[:4]

mat1_diffs
mat2_diffs