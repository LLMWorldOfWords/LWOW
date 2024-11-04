

import numpy as np
import pandas as pd
import json
import random
import networkx as nx
import collections
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from tqdm import tqdm

###############################################################################

def loadData(filename):
    with open(filename, 'r') as json_file:
        json_list = list(json_file)
    output = []
    for json_str in json_list:
        result = json.loads(json_str)
        output.append(result)
    return output

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
def cue100(df1):
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

# Get one row per cue (for GPT data)
def cue1(df1):
    df = df1.copy()
    df = df.drop_duplicates(['cue'])
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
def RemoveRespArticles(df1):
    df = df1.copy()
    for col in ['R1', 'R2', 'R3']:
        for prefix in ['a ', 'an ', 'the ', 'to ']:
            mask = (df[col].str.startswith(prefix)) & (~df[col].isin(unqCues))
            df.loc[mask, col] = df.loc[mask, col].str[len(prefix):]
    return df

# Add spaces or hyphens when one is missing
def AddSpaceOrHyphen(df1):
    df = df1.copy()
    for col in ['cue', 'R1', 'R2', 'R3']:
        df[col] = df[col].map(missingDict).fillna(df[col])
    return df

# Correct spelling
def Spelling(df1):
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
def cleaningPipeline(df1, name):
    df = df1.copy()
    df = NA2Blank(df) # make the NA responses into blanks    
    df = Lowercase(df) # make everything lowercase
    df = RemoveUnderscore(df)
    df = RemoveRespArticles(df) # remove articles from responses if not a cue
    df = AddSpaceOrHyphen(df) # Add spaces or hyphen when one is missing
    df = Spelling(df) # correct spelling of cues and responses if in spelling dictionary
    df = Lemmatization(df) # lemmatize all cues and responses
    if 'GPT' in name:
        df = cue1(df) # remove duplicate cues
    else:
        df = cue100(df) # align data to get 100 sets of responses per cue
    df = RemoveCueResp(df) # remove responses that are equal to cues
    df = RemoveDupeResp(df) # remove duplicate responses
    df = ShiftResp(df) # align dataframe so all response are to the left
    df = SortColumns(df)
    return df

###############################################################################
# Free Associations

# Load spelling lookup list for cleaning
filename = './data/mapping_tables/EnglishCustomDict.txt'
spelling = loadTextFile(filename)
spelling_dict = {a.lower():b.lower() for [a, b] in spelling}

# Load SWOW data
FA_SWOW = pd.read_csv('./data/output/FA_SWOW.csv')
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
models = ['Humans', 'Mistral', 'Llama3', 'Haiku', 'GPT-3.5 Turbo', 'GPT-4o', 'GPT-4 Turbo']

# Load LLM FA data and put it into a dataframe
FA_data = {}
FA_data['Humans'] = FA_SWOW
FA_data['Mistral'] = FA_df(loadData('./data/original_datasets/Mistral/mistral_free_associations.jsonl'))
FA_data['Llama3'] = FA_df(loadData('./data/original_datasets/Llama3/llama3-8b_free_associations.jsonl'))
FA_data['Haiku'] = FA_df(loadData('./data/original_datasets/Haiku/haiku_free_associations.jsonl'))
FA_data['GPT-3.5 Turbo'] = FA_df(loadData('./data/original_datasets/GPT/gpt-3.5_turbo125_male-female.jsonl'))
FA_data['GPT-4o'] = FA_df(loadData('./data/original_datasets/GPT/gpt-4o-2024-05-13_male-female.jsonl'))
FA_data['GPT-4 Turbo'] = FA_df(loadData('./data/original_datasets/GPT/gpt-4-turbo-2024-04-09_male-female.jsonl'))

# Apply the cleaning procedure to the dataframes
FA_clean_dfs = {}
for model, data in FA_data.items():
    clean = cleaningPipeline(data, model)
    FA_clean_dfs[model] = clean

# Save datasets
for model, data in FA_clean_dfs.items():
    data.to_csv('./data/output/FA_cleaned_dfs/FA_' + model + '.csv')

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
df.to_csv('./data/summary_tables/FA_summary_stats.csv', index = False)

