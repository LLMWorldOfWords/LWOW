

import numpy as np
import pandas as pd
import json
import random
import collections
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from tqdm import tqdm
from FA_Functions import *


###############################################################################
# Free Associations

# Load spelling lookup list for cleaning
data_path = './data/mapping_tables/EnglishCustomDict.txt'
spelling = loadTextFile(data_path)
spelling_dict = {a.lower():b.lower() for [a, b] in spelling}

# Load SWOW data
FA_SWOW = loadSimplifiedSWOW()
FA_SWOW = FA_SWOW[FA_SWOW['cue'] != 'nan']

# Get cues
OrigCues = FA_SWOW['cue']
 # Remove NAs
OrigCues  = [x for x in OrigCues if isinstance(x, str)]
# Make lowercase
OrigCues = [x.lower() for x in OrigCues]
# Correct spelling
OrigCues = [spelling_dict.get(x) if x in spelling_dict.keys() else x for x in OrigCues]
# Lemmatize and manually change men to man
OrigCues = [lemmatizer.lemmatize(x) if x != 'men' else 'man' for x in OrigCues]
# Final set of unique cues
unqCues = list(set(OrigCues))

# Creates mapping dictionary to map words with no spaces to words with hyphens or spaces
wnWords = []
for w in wn.all_synsets():
    wnWords.append([str(lemma.name()) for lemma in w.lemmas()])
wnWordsFlat = [item for sublist in wnWords for item in sublist]
wnWordsLower = [x.lower() for x in wnWordsFlat]
noSpacesDict = {x.replace('_', ''): x.replace('_', ' ') for x in wnWordsLower}
noHyphensDict = {x.replace('-', ''): x for x in wnWordsLower}
onlyNoSpaces = list(set(list(noSpacesDict.keys())) - set(wnWordsLower))
onlyNoHyphens = list(set(list(noHyphensDict.keys())) - set(wnWordsLower))
onlyNoSpacesDict = {x:noSpacesDict[x] for x in onlyNoSpaces}
onlyNoHyphensDict = {x:noHyphensDict[x] for x in onlyNoHyphens}
missingDict = onlyNoSpacesDict.copy()
missingDict.update(onlyNoHyphensDict)

# All models
models = ['Humans', 'Mistral', 'Llama3', 'Haiku']

# Load LLM FA data and put it into a dataframe
FA_data = {}
FA_data['Humans'] = FA_SWOW
FA_data['Mistral'] = FA_df(loadData('./data/original_datasets/mistral_free_associations.jsonl'))
FA_data['Llama3'] = FA_df(loadData('./data/original_datasets/llama3-8b_free_associations.jsonl'))
FA_data['Haiku'] = FA_df(loadData('./data/original_datasets/haiku_free_associations.jsonl'))

# Apply the cleaning procedure to the dataframes
FA_clean_dfs = {}
for model, data in FA_data.items():
    clean = cleaningPipeline(data, unqCues, missingDict, spelling_dict, model)
    FA_clean_dfs[model] = clean

# Save datasets
for model, data in FA_clean_dfs.items():
    data.to_csv('./data/processed_datasets/FA_' + model + '.csv', index = False)

# Get summary of the cleaned data
FA_summary_dict = {}
for model, data in FA_clean_dfs.items():
    all_resp = list(data['R1']) + list(data['R2']) + list(data['R3'])
    all_resp = [x for x in all_resp if x != '']
    d = {'Unique cues': len(set(data['cue'])),
         'Total responses': len(all_resp),
         'Unique responses': len(set(all_resp)),
         'Percent missing responses': ((len(data)*3) - len(all_resp))/(len(data)*3)}
    FA_summary_dict[model] = d
df = pd.DataFrame.from_dict(FA_summary_dict).T
df

# Save the pre-processed data as a csv
df.to_csv('./data/summary_tables/FA_summary_stats.csv', index = True)

