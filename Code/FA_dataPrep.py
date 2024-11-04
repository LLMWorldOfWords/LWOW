
import numpy as np
import pandas as pd
import json

# pre-processed SWOW data
df_orig = pd.read_csv('./data/original_datasets/SWOW/SWOW-EN.R100.csv')

# countries
countries = pd.read_csv('./data/mapping_tables/country.csv')

# Creates input data for LLMs
df = df_orig

# Gender
df = df.replace({'gender':
                    {'Fe': 'Female',
                    'Ma': 'Male',
                    'X': 'Unknown'}})

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
df['nativeLanguage'] = np.where(df['nativeLanguage'].isin(eng),
                                'English', 'Not English')
# Country
df['country'] = np.where(df['country'].isin(countries.value.values),
                            df['country'], 'Unknown')

# Cue (convert to strings)
df['cue'] = [str(x) for x in df.cue.values]

# Responses (convert NANs to blanks)
df.R1 = df.R1.fillna('')
df.R2 = df.R2.fillna('')
df.R3 = df.R3.fillna('')

# Choose only the variables we need
df = df[['age','gender','nativeLanguage','country','education','cue','R1','R2','R3']]

# Save sorted df
df_sorted = df.sort_values(by = ['cue','age','gender','nativeLanguage','country','education'])
df_sorted.to_csv('./data/output/FA_SWOW.csv', index = False)
  
# Group by age, native language, gender, education, country
df_agg = df.groupby(['age',
                        'nativeLanguage',
                        'gender',
                        'education',
                        'country'])

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

with open('./data/output/profiles_cues.json', 'w') as f:
    json.dump(profiles_cues, f)

