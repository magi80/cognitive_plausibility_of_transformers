import sys
from read_tg import read_textgrid
from extract_duration import extract_word_duration
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def get_duration(path):
    """This function implements the get_duration
    function by extraxting the SECOND SAMPA tier with syllable
    marks."""
    data = read_textgrid(path)
    sentences = []
    i = 0
    while i < len(data):
        # Finds the start index of the second tier
        if 'item [2]:' in data[i]:
            start_sentence = i
            for j in range(i+1, len(data)):
                # End the end index of the second tier
                if 'item [3]:' in data[j]:
                    end_sentence = j
                    # Extracts the data for each sentence
                    sen = data[start_sentence:end_sentence]
                    # Function call for extracting the word and its duration
                    word_dur = extract_word_duration(sen)
                    sentences.append(word_dur)
                    # Avoids extracting unrelated strings
                    break
        i += 1
    # Function call to get_bigram_dur for creating 
    # bigrams as dictionary keys. This ensures that
    # the duration for words with freq > 1 is correctly
    # estimated, such as 'the': ('and', 'the') ('of', 'the')
    #bigrams_dur = get_bigrams_duration(sentences)
    #return bigrams_dur
    return sentences
              

def get_orth_tier(path):
    """This function implements the get_duration
    function by extraxting the FIRST ORTH tier with syllable
    marks."""
    data = read_textgrid(path)
    sentences = []
    i = 0
    while i < len(data):
        # Finds the start index of the second tier
        if 'item [1]:' in data[i]:
            start_sentence = i
            for j in range(i+1, len(data)):
                # End the end index of the second tier
                if 'item [2]:' in data[j]:
                    end_sentence = j
                    # Extracts the data for each sentence
                    sen = data[start_sentence:end_sentence]
                    # Function call for extracting the word and its duration
                    word_dur = extract_word_duration(sen)
                    sentences.append(word_dur)
                    # Avoids extracting unrelated strings
                    break
        i += 1
    return sentences


def get_articulation_rates(tuples_data):
    art = []
    for bible in tuples_data:
        for tpl in bible:
            syllables = tpl[0].count('.')
            #print(syllables)
            #print((tpl[0], syllables+1))
            #if syllables == 0:
            #    syllables = 1
            syllables = syllables + 1
            #ar = (tpl[1]/syllables)
            ar = (syllables/ tpl[1]) * 1000 
            art.append((tpl[0], ar))
    return art
            
                       
              
if __name__ == '__main__':
    # sys.argv takes a TextGrid file
    data = get_duration(sys.argv[1])
    print(data)
    ar = get_articulation_rates(data)
    print(ar)
    #orth = get_orth_tier(sys.argv[1])
    print(len(ar))

    file_csv = sys.argv[1]
    df = pd.DataFrame(ar, columns=['sampa', 'ar'])
    #df = df[(df['ar'] < 10)]

    df['log_ar'] = np.log2(df['ar'])
#df['z_ar'] = zscore(df['ar'])
#df_clean = df[(df['z_ar'] > -2.75) & (df['z_ar'] < 2.75)]

    plt.figure(figsize=(6, 4))
    sns.histplot(df['log_ar'], bins=70, kde=True)   
    plt.xlabel('Articulation Rate (syll/sec)')
    plt.title('Histogram of Articulation Rate')
    plt.tight_layout()
    plt.show()
    #print(len(orth[0]))
    #data = {'sampa': [], 'orth': []}
    #for tupls in ar:
    #    data['sampa'].append(tupls)
    
    #for tupls in orth[0]:
    #    data['orth'].append(tupls)

    #for sampa, orth in zip(data['sampa'], data['orth']):
    #    print((orth[0], sampa[0]))
   