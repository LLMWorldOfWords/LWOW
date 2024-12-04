

import networkx as nx
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy import stats
from tqdm import tqdm
from sklearn import preprocessing
from nltk.corpus import wordnet as wn
import itertools
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from functools import reduce
import os
from FA_Functions import *
base_dir = os.path.dirname(os.path.dirname(__file__))

########################################################################################

modelColors = {'Humans': '#3269AF',
               'Mistral': '#F19542',
               'Llama3': '#73BC6B',
               'Haiku': '#6A54A6'}

modelColorPalettes = {'Humans': 'Blues',
                      'Mistral': 'Oranges',
                      'Llama3': 'Greens',
                      'Haiku': 'Purples'}

# These graphs are weighted and directed, and they are NOT the CCs
FA_graphs = pickle.load(open(os.path.join(base_dir, 'data/graphs/FA_all_graphs.pickle', 'rb')))
models = ['Humans', 'Mistral', 'Llama3', 'Haiku']

# Create full graphs but undirected versions
FA_graphs_full = {}
for model in models:
    g = FA_graphs[model]
    g = makeUndirected(g)
    FA_graphs_full[model] = g

# Create the filtered graphs by applying the WN nodes and non-idiosyncratic edge filters
# Finally take the LCC
FA_graphs_filt = {}
for model in models:
    g = FA_graphs_full.copy()[model]
    g = WNfilter(g)
    g = idiosynfilter(g)
    g = CC(g)
    FA_graphs_filt[model] = g

##################################################################

# Load and clean LDT dataset
LDT = pd.read_csv(os.path.join(base_dir, 'data/LDT_analyses/primingLDT_data.csv'))
LDT = LDT[LDT['rel'].isin(['un', 'rel'])]
LDT = LDT[LDT['keep'] == 1]
LDT = LDT[['prime', 'target', 'rel', 'target.RT', 'Ztarget.RT']]
LDT['prime'] = [str(x.lower()) for x in LDT['prime']]
LDT['target'] = [str(x.lower()) for x in LDT['target']]

# Get all triplets
grouped = LDT.groupby(by = 'target')
tgt_rel_un = {}
for tgt, data in grouped:
    rel_prime = data[data['rel'] == 'rel']
    rel_primes = {}
    for prime, d in rel_prime.groupby(by = 'prime'):
        rel_z = np.average([float(x) for x in list(d['Ztarget.RT'])])
        rel_primes[prime] = rel_z
    un_primes = {}
    un_prime = data[data['rel'] == 'un']
    for prime, d in un_prime.groupby(by = 'prime'):
        un_z =  np.average([float(x) for x in list(d['Ztarget.RT'])])
        un_primes[prime] = un_z
    combos = itertools.product(list(rel_primes.keys()), list(un_primes.keys()))
    for combo in combos:
        tgt_rel_un[tgt, combo[0], combo[1]] = [rel_primes[combo[0]], un_primes[combo[1]], un_primes[combo[1]] - rel_primes[combo[0]]]
tgt_rel_un_UNQ = {}
for trip, dat in tgt_rel_un.items():
    if trip[0] not in tgt_rel_un_UNQ.keys():
        tgt_rel_un_UNQ[trip] = (trip, dat)
    else:
        if dat[2] > tgt_rel_un_UNQ[trip[0]][1][2]:
            tgt_rel_un_UNQ[trip] = (trip, dat)
tgt_rel_un_UNQ = {x[0]:x[1] for x in list(tgt_rel_un_UNQ.values())}

# Find node intersection of graphs
all_nodes_full = []
for model, g in FA_graphs_filt.items():
    all_nodes_full.append(list(g.nodes()))
nodeIntFull = list(reduce(lambda x, y: set(x) & set(y), all_nodes_full))

# Find the LDT triplets in all graphs
keepKeys = []
for trip in tgt_rel_un_UNQ.keys():
    if (trip[0] in nodeIntFull and trip[1] in nodeIntFull and trip[2] in nodeIntFull):
        keepKeys.append(trip)
tgt_rel_un_keep = {k:tgt_rel_un_UNQ[k] for k in keepKeys}

# Of those triplets in the graphs, find those with largest effects
diffs = dict(sorted(tgt_rel_un_keep.items(), key=lambda item: item[1][2], reverse=True))
LDT_RT_dict = {}
for trip, acts in tqdm(diffs.items()):
    if len(list(LDT_RT_dict.keys())) == 0:
        LDT_RT_dict[trip] = acts
    else:
        primes = list(np.concatenate([[k[1], k[2]] for k in LDT_RT_dict.keys()]))
        if (trip[1] not in primes) and (trip[2] not in primes):
            LDT_RT_dict[trip] = acts
LDT_50_triplets = list(LDT_RT_dict.keys())[:50]
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
pd.DataFrame({'prime':primes}).to_csv(os.path.join(base_dir, 'data/LDT_analyses/LDT_primes.csv', index = False))

# Get the 50 triplets and save them
LDT_50_triplets_DF = pd.DataFrame({'Target': [x[0] for x in LDT_50_triplets],
'Related Prime': [x[1] for x in LDT_50_triplets],
'Unrelated Prime': [x[2] for x in LDT_50_triplets],
'Target-Related RT': [x[0] for x in LDT_RT_dict.values()],
'Target-Unrelated RT': [x[1] for x in LDT_RT_dict.values()]})
LDT_50_triplets_DF.to_csv(os.path.join(base_dir, 'data/LDT_analyses/LDT_50_triplets.csv', index = False))

# Make list of bias related words
genderWords = ['woman', 'man',
             'girl', 'boy',
             'mother', 'father',
             'female', 'male',
             'feminine', 'masculine'
             ]
for word in genderWords:
    if word not in nodeIntFull:
        print(word)
pd.DataFrame({'genderWord':genderWords}).to_csv(os.path.join(base_dir, 'data/LDT_analyses/gender_primes.csv', index = False))

###########################################################################

# Load results
LDT_dfs = {}
gender_dfs = {}
for model in models:
    LDT_dfs [model] = pd.read_csv(os.path.join(base_dir, 'data/LDT_analyses/FA_matrices/' + model + '_LDT.csv'))
    gender_dfs[model] = pd.read_csv(os.path.join(base_dir, 'data/LDT_analyses/FA_matrices/' + model + '_gender.csv'))

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
targets = pd.read_csv(os.path.join(base_dir, 'data/LDT_analyses/gender_targets.csv'))
tgts_F = [x for x in list(targets['Female']) if x in nodeIntFull]
tgts_M = [x for x in list(targets['Male'])if x in nodeIntFull]

#####################################

# Reaction time boxplots
boxplotRTs(LDT_RT_dict)
# Wilcoxon test
data1 = [x[0] for x in list(LDT_RT_dict.values())]
data2 =  [x[1] for x in list(LDT_RT_dict.values())]
data = [a - b for (a,b) in zip(data1, data2)]
res = stats.wilcoxon(data, alternative = 'less', method = 'approx')
p = res.pvalue
z = res.zstatistic
effect = z / np.sqrt(len(data))
wilxoconRT = {'p':p, 'effect': effect}
wilxoconRT

# Activation level boxplots for LDT analyses
LDT_AL_dicts = {}
for model, g in FA_graphs_filt.items():
    LDT_df = LDT_dfs[model]
    LDT_df_norm = normalizeDF(LDT_df, normalize_rows = True)
    LDT_AL_dict = activationDict(LDT_df_norm, LDT_50_triplets)
    LDT_AL_dicts[model] = LDT_AL_dict
boxplotsLDT(LDT_AL_dicts, modelColors)

# Wilcoxon test for LDT analyses
wilxoconLDT = {}    
for model, d in LDT_AL_dicts.items():
    data1 = [x[0] for x in list(d.values())]
    data2 =  [x[1] for x in list(d.values())]
    data = [a - b for (a,b) in zip(data1, data2)]
    res = stats.wilcoxon(data, alternative = 'greater', method = 'approx')
    p = res.pvalue
    z = res.zstatistic
    effect = z / np.sqrt(len(data))
    wilxoconLDT[model] = {'p':p, 'effect': effect}
pd.DataFrame(wilxoconLDT).to_csv(os.path.join(base_dir, 'data/LDT_analyses/Statistical_tests/wilcoxonLDT.csv', index = True))

# Correlations for LDT analyses
corrsLDT = {}  
RT1 = [x[0] for x in list(LDT_RT_dict.values())]
RT2 = [x[1] for x in list(LDT_RT_dict.values())]  
for model, d in LDT_AL_dicts.items():
    AL1 = [x[0] for x in list(d.values())]
    AL2 =  [x[1] for x in list(d.values())]
    r, p = spearmanr(np.concatenate([AL1, AL2]), np.concatenate([RT1, RT2]))
    corrsLDT[model] = {'r': r, 'p': p}
pd.DataFrame(corrsLDT).to_csv(os.path.join(base_dir, 'data/LDT_analyses/Statistical_tests/corrLDT.csv', index = True))


# For gender data analyses
matGender = {}
for model, g in FA_graphs_filt.items():
    gender_df = gender_dfs[model]
    gender_df_norm = normalizeDF(gender_df, normalize_rows = True)
    matF = matricesGender(gender_df_norm, tgts_F, primes_F, primes_M)
    matM = matricesGender(gender_df_norm, tgts_M, primes_F, primes_M)
    matGender[model] = {'Female-related targets': matF,
                        'Male-related targets': matM}

# Boxplots for Gender
boxplotsGender(matGender, 'Female-related targets', primes_F, primes_M, modelColors, main = True)
boxplotsGender(matGender, 'Male-related targets', primes_F, primes_M, modelColors, main = False)

# Activation level heatmaps for Gender
for model in models:
    heatmapsGender(matGender[model]['Female-related targets'], matGender[model]['Male-related targets'], model, modelColorPalettes)
 
# AL differences wilcoxon test
wilxoconF = {}
wilxoconM = {}
AL_differences = {}
for model in models:
    testRes ={}
    AL_differences[model] = {}
    for t in [('Female-related targets', 'greater'), ('Male-related targets', 'less')]:
        data1 = matGender[model][t[0]][primes_F].values.flatten()
        data2 = matGender[model][t[0]][primes_M].values.flatten()
        data = [a - b for (a,b) in zip(data1, data2)]
        AL_differences[model][t[0]] = data
        res = stats.wilcoxon(data, alternative = t[1], method = 'approx')
        p = res.pvalue
        z = res.zstatistic
        effect = z / np.sqrt(len(data))
        testRes[t[0]] = {'p':p, 'effect': effect}
    wilxoconF[model] = testRes['Female-related targets']
    wilxoconM[model] = testRes['Male-related targets']
pd.DataFrame(wilxoconF).to_csv(os.path.join(base_dir, 'data/LDT_analyses/Statistical_tests/wilxoconGenderF.csv', index = True))
pd.DataFrame(wilxoconM).to_csv(os.path.join(base_dir, 'data/LDT_analyses/Statistical_tests/wilxoconGenderM.csv', index = True))


# AL differences histograms
fig, axes = plt.subplots(4, 2, figsize=(12, 16), sharey=True)
fig.suptitle('Difference in activation levels', fontsize=24, y=.97, weight='bold')
plt.figtext(0.5, .93, 'Female-related primes - Male-related primes', ha='center', fontsize=20)
legend_patches = [mpatches.Patch(color=modelColors[model], label=model) for model in models]
for i, model in enumerate(models):
    # Female-related targets
    axes[i, 0].hist(AL_differences[model]['Female-related targets'], bins=30, color=modelColors[model], edgecolor='black')
    axes[i, 0].set_ylabel('Frequency', fontsize=12)
    if i == 0:
        axes[i, 0].set_title('Female-related targets', fontsize=18)
    else:
        axes[i, 0].set_title('')
    if i == 3:
        axes[i, 0].set_xlabel('Difference in activation level', fontsize=18)
    else:
        axes[i, 0].set_xlabel('')
    # Male-related targets
    axes[i, 1].hist(AL_differences[model]['Male-related targets'], bins=30, color=modelColors[model], edgecolor='black')
    if i == 0:
        axes[i, 1].set_title('Male-related targets', fontsize=18)
    else:
        axes[i, 1].set_title('')
    if i == 3:
        axes[i, 1].set_xlabel('Difference in activation level', fontsize=18)
    else:
        axes[i, 1].set_xlabel('', fontsize=12)
fig.legend(handles=legend_patches, loc='upper right', ncol=1, fontsize=12, title='Network', title_fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# Correlations humans vs. LLMs
modelsCorrF = {}
modelsCorrM = {}
for model in models[1:]:
    
    mat1F = np.concatenate(matGender['Humans']['Female-related targets'].values)
    mat2F = np.concatenate(matGender[model]['Female-related targets'].values)
    rF, pF = spearmanr(mat1F, mat2F)
    modelsCorrF[model] = {'r': rF, 'p': pF}
    
    mat1M = np.concatenate(matGender['Humans']['Male-related targets'].values)
    mat2M = np.concatenate(matGender[model]['Male-related targets'].values)
    rM, pM = spearmanr(mat1M, mat2M)
    modelsCorrM[model] = {'r': rM, 'p': pM}

pd.DataFrame(modelsCorrF).to_csv(os.path.join(base_dir, 'data/LDT_analyses/Statistical_tests/corrModelsF.csv', index = True))
pd.DataFrame(modelsCorrM).to_csv(os.path.join(base_dir, 'data/LDT_analyses/Statistical_tests/corrModelsM.csv', index = True))

