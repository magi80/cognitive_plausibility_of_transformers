import pandas as pd
import numpy as np
from scipy.spatial import distance
import seaborn as sns
import matplotlib.pyplot as plt
import difflib
from matplotlib.patches import Patch


def match_tokens(tokens_gpt2, tokens_bert, surprisal_gpt2, surprisal_bert, probs_gpt2, probs_bert):
    matcher = difflib.SequenceMatcher(None, tokens_gpt2, tokens_bert)

    aligned_rows = []
    for block in matcher.get_matching_blocks():
        i, j, size = block
        # Each block is a matching continuous run of tokens
        for k in range(size):
            aligned_rows.append({
                'token': tokens_gpt2[i+k],
                'token2': tokens_bert[j+k],
                'gpt2_surprisal': surprisal_gpt2[i+k],
                'bert_surprisal': surprisal_bert[j+k],
                'probs_gpt2': probs_gpt2[i+k],
                'probs_bert': probs_bert[j+k]
            })

    # Create a DataFrame of aligned tokens and surprisals
    aligned_df = pd.DataFrame(aligned_rows)

    print(f"Aligned tokens count: {len(aligned_df)}")
    print(aligned_df)
    print(len(aligned_df))
    return aligned_df

# Debugged CSVs. Change with current CVS in FOLDER
csv_file = {'gpt2': {512: 'CSV_DEBAGGATI/ENGKJV_surprisal_GPT2_512_DEBUG_PREPROCESS_new.csv',
                     256: 'CSV_DEBAGGATI/ENGKJV_surprisal_GPT2_256_DEBUG_PREPROCESS_new.csv',
                     128: 'CSV_DEBAGGATI/ENGKJV_surprisal_GPT2_128_DEBUG_PREPROCESS_new.csv'},
            'bert': {512: 'CSV_DEBAGGATI/ENGKJV_surprisal_BERT_512_DEBUG_CW.csv',
                     256: 'CSV_DEBAGGATI/ENGKJV_surprisal_BERT_256_DEBUG_CW.csv',
                     128: 'CSV_DEBAGGATI/ENGKJV_surprisal_BERT_128_DEBUG_CW.csv'}
            }

gpt2_512 = pd.read_csv(csv_file.get('gpt2').get(512), sep='\t').dropna()
gpt2_256 = pd.read_csv(csv_file.get('gpt2').get(256), sep='\t').dropna()
gpt2_128 = pd.read_csv(csv_file.get('gpt2').get(128), sep='\t').dropna()

bert_512 = pd.read_csv(csv_file.get('bert').get(512), sep='\t').dropna()
bert_256 = pd.read_csv(csv_file.get('bert').get(256), sep='\t').dropna()
bert_128 = pd.read_csv(csv_file.get('bert').get(128), sep='\t').dropna()


gpt2_512_surp = gpt2_512['surprisal'].to_list()
gpt2_256_surp = gpt2_256['surprisal'].to_list()
gpt2_128_surp = gpt2_128['surprisal'].to_list()

bert_512_surp = bert_512['surprisal'].to_list()
bert_256_surp = bert_256['surprisal'].to_list()
bert_128_surp = bert_128['surprisal'].to_list()

gpt2_512_tokens = gpt2_512['tokens'].to_list()
gpt2_256_tokens = gpt2_256['tokens'].to_list()
gpt2_128_tokens = gpt2_128['tokens'].to_list()

bert_512_tokens = bert_512['tokens'].to_list()
bert_256_tokens = bert_256['tokens'].to_list()
bert_128_tokens = bert_128['tokens'].to_list()

print(f'--Total GPT2 512 tokens: {len(gpt2_512_tokens)}')
print(f'--Total GPT2 256 tokens: {len(gpt2_256_tokens)}')
print(f'--Total GPT2 128 tokens: {len(gpt2_128_tokens)}')

print(f'--Total BERT 512 tokens: {len(bert_512_tokens)}')
print(f'--Total BERT 256 tokens: {len(bert_256_tokens)}')
print(f'--Total BERT 128 tokens: {len(bert_128_tokens)}')

gpt2_512_prob = [np.exp2(-s) for s in gpt2_512_surp]
gpt2_256_prob = [np.exp2(-s) for s in gpt2_256_surp]
gpt2_128_prob = [np.exp2(-s) for s in gpt2_128_surp]

bert_512_prob = [np.exp2(-s) for s in bert_512_surp]
bert_256_prob = [np.exp2(-s) for s in bert_256_surp]
bert_128_prob = [np.exp2(-s) for s in bert_128_surp]

#assert len(gpt2_surp) == len(gpt2_prob), 'BOOM'
#print(gpt2_tokens)

GPT2_512 = {'tokens': gpt2_512['tokens'], 'surprisal': gpt2_512['surprisal'], 'probs': gpt2_512_prob}
GPT2_256 = {'tokens': gpt2_256['tokens'], 'surprisal': gpt2_256['surprisal'], 'probs': gpt2_256_prob}
GPT2_128 = {'tokens': gpt2_128['tokens'], 'surprisal': gpt2_128['surprisal'], 'probs': gpt2_128_prob}
        
BERT_512 = {'tokens': bert_512['tokens'], 'surprisal': bert_512['surprisal'], 'probs': bert_512_prob}
BERT_256 = {'tokens': bert_256['tokens'], 'surprisal': bert_256['surprisal'], 'probs': bert_256_prob}
BERT_128 = {'tokens': bert_128['tokens'], 'surprisal': bert_128['surprisal'], 'probs': bert_128_prob}

GPT2 = pd.DataFrame(GPT2_256, columns=['tokens', 'surprisal', 'probs'])
BERT = pd.DataFrame(BERT_256, columns=['tokens', 'surprisal', 'probs'])

#print('--GPT-2 dataframe:') #debug
#print(GPT2)
#print('-'*20)
#print('--BERT dataframe:')
#print(BERT)
#print('-'*20)

comb512 = match_tokens(gpt2_512_tokens, bert_512_tokens, gpt2_512_surp, bert_512_surp, gpt2_512_prob, bert_512_prob)
comb256 = match_tokens(gpt2_256_tokens, bert_256_tokens, gpt2_256_surp, bert_256_surp, gpt2_256_prob, bert_256_prob)
comb128 = match_tokens(gpt2_128_tokens, bert_128_tokens, gpt2_128_surp, bert_128_surp, gpt2_128_prob, bert_128_prob)

#print(comb512) #debug
combined = GPT2.join(BERT, lsuffix='_gpt2', rsuffix='_bert')
print('--Combined Pandas:')
print(combined.reset_index(drop=True))
print('-'*20)
matches = combined['tokens_gpt2'] == combined['tokens_bert']
print(f'Matching tokens: {matches.sum()} out of {len(combined)}')
print('-'*20)

#comb = combined.dropna()
comb = comb128.dropna() #change to comb512 and comb256
print(comb.reset_index(drop=True))
match = comb['token'] == comb['token2']
print(f'Matching tokens: {match.sum()} out of {len(comb)}') 
#combined.to_csv('combined_gpt2_bert.csv', sep='\t')

probs_gpt2_128 = comb['probs_gpt2'].to_list()
probs_bert_128 = comb['probs_bert'].to_list()
print('--P and Q distribution lengths:')
print(f'--P(gpt2): {len(probs_bert_128)}')
print(f'--Q(bert): {len(probs_bert_128)}')

p = np.array(probs_gpt2_128) #change to probs_gpt2_512 and robs_gpt2_256
q = np.array(probs_bert_128) #change to probs_gpt2_512 and robs_gpt2_256

#normalize probs
p /= p.sum()
q /= q.sum()
print('--JSD:')
jsd = distance.jensenshannon(p, q, 2.0)
print(round(jsd, 3))

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
sns.histplot(data=comb512, x='gpt2_surprisal', bins=70, label='512', alpha=0.6, ax=axes[0])
sns.histplot(data=comb256, x='gpt2_surprisal', bins=70, label='256', alpha=0.6, ax=axes[0])
sns.histplot(data=comb128, x='gpt2_surprisal', bins=70, label='128', alpha=0.6, ax=axes[0])
axes[0].set_title(f'Surprisal Distribution for GPT-2')
axes[0].legend()

sns.histplot(data=comb512, x='bert_surprisal', bins=70, label='512', multiple='stack', ax=axes[1])
sns.histplot(data=comb256, x='bert_surprisal', bins=70, label='256', multiple='stack', ax=axes[1])
sns.histplot(data=comb128, x='bert_surprisal', bins=70, label='128', multiple='stack', ax=axes[1])
axes[1].set_title(f'Surprisal Distribution for BERT')
axes[1].legend()

axes[0].set_xlabel('Surprisal')
axes[0].set_ylabel('Frequency')
axes[1].set_xlabel('Surprisal')

jsd_512 = 0.788
jsd_256 = 0.803
jsd_128 = 0.768

legend_elements = [
    Patch(color='C0', label=f'512, JSD={jsd_512:.3f})'),
    Patch(color='C1', label=f'256, JSD={jsd_256:.3f})'),
    Patch(color='C2', label=f'128, JSD={jsd_128:.3f})')
]

axes[0].legend(handles=legend_elements, title='Context Lengths', loc='upper right')
axes[1].legend(handles=legend_elements, title='Context Lengths', loc='upper right')

print('-- Mean Surprisal (GPT-2) --')
print(f'128: {comb128["gpt2_surprisal"].mean():.3f}')
print(f'256: {comb256["gpt2_surprisal"].mean():.3f}')
print(f'512: {comb512["gpt2_surprisal"].mean():.3f}')

totsurp = (comb128['gpt2_surprisal'].sum() + comb256['gpt2_surprisal'].sum() + comb512['gpt2_surprisal'].sum())
tottokens = len(comb128['token']) + len(comb256['token']) + len(comb512['token'])
mean_val_gpt = totsurp/tottokens
print('--Mean GPT-2:',totsurp/tottokens )

print('-- Mean Surprisal (BERT) --')
print(f'128: {comb128["bert_surprisal"].mean():.3f}')
print(f'256: {comb256["bert_surprisal"].mean():.3f}')
print(f'512: {comb512["bert_surprisal"].mean():.3f}')

totsurpbert = (comb128['bert_surprisal'].sum() + comb256['bert_surprisal'].sum() + comb512['bert_surprisal'].sum())
tottokensbert = len(comb128['token2']) + len(comb256['token2']) + len(comb512['token2'])
mean_val_bert = totsurpbert/tottokens
print('--Mean GPT-2:',totsurpbert/tottokensbert )

axes[0].axvline(mean_val_gpt, linestyle='--', label=f'GPT-2 Mean = {mean_val_gpt:.2f}')
axes[0].legend()

axes[1].axvline(mean_val_bert, linestyle='--', label=f'BERT Mean = {mean_val_bert:.2f}')
axes[1].legend()

plt.tight_layout()
plt.subplots_adjust(wspace=0.2)
plt.show()
#plt.savefig('RQ1.png', dpi=300)