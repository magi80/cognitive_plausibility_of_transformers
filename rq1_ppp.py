import pandas as pd
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Log-likelihood scores obtained from rthe r2_glms.py scripts
data = {'models': ['GPT2-512', 'GPT2-256', 'GPT2-128',
                   'BERT-512', 'BERT-256', 'BERT-128'],
        'perplexity': [31.84, 33.08, 47.71,
                       8.00, 7.53, 10.87],
        'log_lik': [-114256.77915877078, -59207.6548053951, -29407.999814581315, 
                    -108690.28349505928, -52312.96132379862, -25641.41919112911], 
        'delta_ll': [728.679, 113.073, 635.813, 2161.859, 4421.912, 2815.702]}

# Convert to DataFrame
df = pd.DataFrame(data)

# Add a new column to tag model type
df['model_type'] = df['models'].apply(lambda x: 'GPT-2' if 'GPT2' in x else 'BERT')
df['size'] = df['models'].str.extract(r'-(\d+)').astype(int)
gpt_df = df[df['models'].str.contains("GPT2")]
bert_df = df[df['models'].str.contains("BERT")]
print(df[['models', 'size']])

palette = {128: "#2ca02c", 256: "#ff7f0e", 512: "#1f77b4"}

#plt.figure(figsize=(10, 6))

sns.scatterplot(data=gpt_df, x='perplexity', y='delta_ll', hue='size',
                s=50, edgecolor='black', alpha=0.6, marker='o', #style='size',
                palette=palette)
sns.scatterplot(data=bert_df, x='perplexity', y='delta_ll', hue='size',
                s=50, edgecolor='black', alpha=0.6, marker='s', #style='size',
                palette=palette)

# Converting the x-axis to the log scale
plt.xscale('log', base=2)
ticks = [2**i for i in range(2, 7)] 
plt.xticks(ticks)
plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda val, pos: f"$2^{int(np.log2(val))}$"
))

# Annotate datapoints
for i, row in df.iterrows():
    plt.annotate(row['models'], 
                 xy=(row['perplexity'], row['delta_ll']),
                 xycoords='data',
                 xytext=(2, 11),  # offset label 5 points to the right (4, 10)
                 textcoords='offset points',
                 fontsize=8,
                 arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                 ha='right', va='top')


plt.xlabel("Perplexity")
plt.ylabel("ΔLL")
plt.title("PPP of GPT-2 and BERT Models (ΔLL vs. PPL)")
plt.grid(True)

# Combined legend with size and model type
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))  
plt.legend(by_label.values(), by_label.keys(), title="Size")
plt.tight_layout()
plt.savefig('PPP_plot.png', dpi=300)
plt.show()
