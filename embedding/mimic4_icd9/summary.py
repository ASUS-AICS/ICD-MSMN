import json


with open('mimic4_icd9_random_sort.json', 'r') as f:
    df = json.load(f)
    
count = []
for x in df:
    count.append(len(x))
    
import numpy as np

count = np.array(count)
print(np.mean(count),np.percentile(count,50))