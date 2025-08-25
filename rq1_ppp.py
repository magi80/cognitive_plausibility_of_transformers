import pandas as pd
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


gpt2_file = sys.argv[1]
bert_file = sys.argv[2]

gpt2_csv = pd.read_csv(gpt2_file, delimiter='\t')
bert_csv = pd.read_csv(bert_file, delimiter='\t')

##gpt2_csv.dropna()
#bert_csv.dropna()

combined = gpt2_csv.join(bert_csv, lsuffix='_gpt2', rsuffix='_bert')
#combined.to_csv('fuck.csv', sep='\t')

comb = pd.DataFrame(combined, columns=['tokens_gpt2', 'surprisal_gpt2', 'tokens_bert', 'surprisal_bert'])
#print(comb)
comb.to_csv('COMB1.csv', sep='\t')

co = comb[comb['tokens_gpt2'] == comb['tokens_bert']]
co.to_csv('COMB2.csv', sep='\t')

co['prob_gpt2'] = np.exp2(-co['surprisal_gpt2'])
co['prob_bert'] = np.exp2(-co['surprisal_bert'])
#co.to_csv('ARGHt.csv', sep='\t')
#diff1 = gpt2_csv[~gpt2_csv.apply(tuple, axis=1).isin(bert_csv.apply(tuple, axis=1))]
#diff1.to_csv('diomerdoso.csv', sep='\t')
#read_csv['prob'] = np.exp2(-read_csv['surprisal'])
#read_csv['p_norm'] = read_csv['prob'] / np.sum(read_csv['prob'])

##print(read_csv['p_norm'])
#print(np.sum(read_csv['p_norm']))

#sns.histplot(data=read_csv, x=-np.log2(read_csv['p_norm']), bins=100)
#plt.show()
