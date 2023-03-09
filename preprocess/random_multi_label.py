import ujson
import os
from random import shuffle


with open('../embedding/mimic4_icd9/mimic4_icd9_max_sort.json', 'r') as f:
    df = ujson.load(f)
    
new_df = {}
for key, value in df.items():
    shuffle(value)
    shuffle(value)
    new_df[key] = value
    
with open('../embedding/mimic4_icd9/mimic4_icd9_random_sort.json', 'w') as f:
    ujson.dump(new_df, f, indent=2)


with open('../embedding/mimic4_icd10/mimic4_icd10_max_sort.json', 'r') as f:
    df = ujson.load(f)
    
new_df = {}
for key, value in df.items():
    shuffle(value)
    shuffle(value)
    new_df[key] = value
    
with open('../embedding/mimic4_icd10/mimic4_icd10_random_sort.json', 'w') as f:
    ujson.dump(new_df, f, indent=2)