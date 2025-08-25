from transformers import AutoTokenizer, GPT2LMHeadModel
import math
import torch
import random
import numpy as np
import pandas as pd
#from feature_class_pipe import ExtractFeatures
from feature_class_pipe_DEBUG import ExtractFeatures
from nltk import sent_tokenize
import sys
from tqdm import tqdm
import string


def merge_subtokens(tok_surp_pairs):
    recon = []
    current_word = ""
    current_surprisal = []

    for i, (token, surprisal) in enumerate(tok_surp_pairs):
        if token.startswith(" "):  # New word starts (space at the beginning)
            #print('--Token:', (token, surprisal))
            #print(surprisal)
            #print('-'*20)
            if current_word:  # Store previous word
                #print('Current Word:', current_word)
                #print('Current Surprisal:', current_surprisal)
                #print('-'*20)
                recon.append((current_word, sum(current_surprisal))) #Averaging?
            current_word = token.strip()  # Remove leading space
            current_surprisal = [surprisal]
            #print('Current Surprisal:', current_surprisal)
            #print(current_surprisal)
            #print('-'*20)
        else:  # Subword continuation (no leading space)
            #print('SubToken:', token)
            current_word += token  # Append subword
            current_surprisal.append(surprisal)

    # Append the last word
    if current_word:
        recon.append((current_word, sum(current_surprisal)))
    return recon


def get_surprisal(inputs, probs, tokenizer):
    surp =[]
    for i, token_id in enumerate(inputs['input_ids'][0]):
        token_prob = probs[0, i, token_id].item()  # Get the probability of token_i
        #print(token_prob) #debug
        surprisal = -math.log2(token_prob)  # Surprisal formula
        #token = tokenizer.convert_ids_to_tokens(token_id.item(), skip_special_tokens=False)
        token = tokenizer.decode(token_id, skip_special_tokens=True)
        #print((token, surprisal)) #debug
        #print('-'*20) #debug
        surp.append((token, surprisal))
    return surp


def merge_and_pad(result):
    SURP = {}
    for i in range(len(result)):
        #print('Input to merge_and_pad:')
        #print(result[i])
        merged_tok = merge_subtokens(result[i])
        #print(merged_tok)
        count = len(merged_tok)
        SURP[f'chapter_{i}'] = {'tokens': [], 'surprisal': [], 'length': ''}
        for tupl in merged_tok:
            SURP[f'chapter_{i}']['tokens'].append(tupl[0])
            SURP[f'chapter_{i}']['surprisal'].append(tupl[1])
        SURP[f'chapter_{i}']['length'] = count
    
    for i, (chap, (chap_name, chap_data)) in enumerate(zip(chapter_length, SURP.items())):
        #chap_len = chap
        model_len = chap_data.get('length')
        diff = chap - model_len
        if diff != 0:
            SURP[chap_name]['tokens'].extend(['None']*diff)
            SURP[chap_name]['surprisal'].extend(['None']*diff)
        #print((chap, model_len))
        #print('-'*50)

    surp_data = []
    for chap, lex in SURP.items():
        print('Updated Model Count:') #debug
        tok = lex.get('tokens') #debug
        sur = lex.get('surprisal') #debug
        print((len(tok), len(sur))) #debug
        for token, surp in zip(tok, sur):
            surp_data.append([token, surp])
        #surp_data['tokens'].append(tok)
        #surp_data['surprisal'].append(sur)
    return surp_data
    #print(surp_data)
    #basta = {'tokens': [], 'surprisal': []}
    #SURP_df = pd.DataFrame(surp_data, columns=['tokens', 'surprisal'])


def get_sentences_dct(sentences):
    sent = {}
    sent_length = {}
    for i in range(len(sentences)):
        #print(sentences[i])
        sen = ' '.join(sentences[i])
        sent[f'chapter_{i}'] = sen
        sent_length[f'chapter_{i}'] = len(' '.join(sentences[i]).split())
    return sent, sent_length


def eval_surp(saved_model, txt, ctx=512):
    """
    Takes a list of RAW sentences (not tokenized),
    including punctuation. Each chapter is passed as
    a long string, using all the 512 context window.
    """
    tokenizer = AutoTokenizer.from_pretrained(saved_model)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #config = AutoConfig.from_pretrained(saved_model)
    model = GPT2LMHeadModel.from_pretrained(saved_model) #load weights + config
    #print(model.state_dict()) #deb
    #print('-'*30) #deb 
    #print(model.config) #deb
    #print('-'*30) #deb
    model.eval()

    surp = []
    ctx = ctx
    print('Extracting Surprisal...')
    with torch.no_grad(): # Enter eval mode
        for chapter, txt in tqdm(txt.items(), desc='Processing', unit='sentence'):
            #sents = sent_tokenize(txt)
            #for sent in sents:
                #print('--Passed text:')
                
                inputs = tokenizer(txt, return_tensors='pt', # txt
                                   max_length=ctx, truncation=True, 
                                   padding=True)
                outputs = model(**inputs) # extract the logits
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=-1)
                #print('--Logits:')
                #print(logits)
                #print(logits.shape)
                #print('--Probs:')
                #print(probs)
                #print(probs.shape)
                #print('--Inputs:')
                #print([tokenizer.decode(inp, skip_special_tokens=False) for inp in inputs['input_ids']])
                #print([tokenizer.convert_ids_to_tokens(inp) for inp in inputs['input_ids']])
                #print(inputs['input_ids'].shape)
                sprsl = get_surprisal(inputs, probs, tokenizer)
                #print('--Length Chapter:', len(txt.split()))
                #print('--Lenght Surprisal:', len(sprsl))
                surp.append(sprsl)
    return surp 


def filter_subtokens(result, punctuations=set(string.punctuation)):
    sents = []
    for sentence in result:
        filtered_sent = []
        for subtok_surp_tupl in sentence:
            tok = subtok_surp_tupl[0]
            print('-'*20)
            print(f'--START_subtokens_tuples: [{subtok_surp_tupl}] --first token: [{tok}]')
            #print(tok)
            if "'" in tok:
                print(f'--True: current token contains elision: [{tok}]')
                filtered_sent.append(subtok_surp_tupl)
            #elif tok not in punctuations:
            else:
                print(f'--False: current token is subtoken: [{tok}]')
                filtered_tok = [char for char in tok if char not in punctuations]
                filtered = ''.join(filtered_tok)
                print(f'--Filtered tok: [{filtered}]')  
                if len(filtered) != 0:
                    if filtered == ' ':
                        print(f'--Current filtered tok is empty: {filtered}')
                        subtoktupl = (filtered, 0.0)
                        filtered_sent.append(subtoktupl)
                    else:
                        print(f'--Filtered + surprisal: [{filtered}, {subtok_surp_tupl[1]}]')
                        subtoktupl = (filtered, subtok_surp_tupl[1])
                        filtered_sent.append(subtoktupl)
        sents.append(filtered_sent)
    return sents


if __name__ == '__main__':
    models = {'gpt2_512': "/Volumes/One Touch/MacAir-2025/5LN709/models_updated_gpt2/model-512/best_model", # new folder: models_update
              'gpt2_256': "/Volumes/One Touch/MacAir-2025/5LN709/models_updated_gpt2/model-256/best_model",
              'gpt2_128': "/Volumes/One Touch/MacAir-2025/5LN709/models_updated_gpt2/model-128/best_model"
              }

    LANG_ID = sys.argv[1] #ENGWWH
    mod = sys.argv[2] #gpt2
    context_size = sys.argv[3] #512, 256, 128
    write = sys.argv[4]

    saved_model = ''
    for model, filepath in models.items():
        if context_size in model and mod in model:
            saved_model = filepath
    print(f'--Using Model: {mod.upper()} cw: {context_size}')
    print('-'*30)

    features = ExtractFeatures()
    sentences = features.tok_sent # NORM TEXTS as input to model
    raw_sentences = [bible.get('orth').get('txt') for bible in features.raw_orth] # RAW texts

    #print('--Normalized Sentences:') #debug
    #print(sentences[:1]) #debug
    #print('-'*20)
    #print(raw_sentences[:1])
    #print('--Total Chapters:', len(sentences)) #debug
    #assert len(sentences) == 28, 'The number of sentences do not match.'
    #print('-'*30)

    norm_sent, norm_sent_length = get_sentences_dct(sentences) # NORM TEXTS as dcts
    norm_chapter_length = [length for chap, length in norm_sent_length.items()] #deb
    #print('--Chapter Lengths:') #deb 
    #print(sum(chapter_length)) #deb 
    #print(sent_length)
    #print('-'*30)

    # UPDATED pass raw Bible texts instead
    # Coherence between Wiki + Bible
    raw, raw_length = get_sentences_dct(raw_sentences) # NORM TEXTS as dcts
    chapter_length = [length for chap, length in raw_length.items()] #deb
    #print('--Chapter Lengths:') #deb 
    #print(f' {sum(chapter_length)} tokens') #deb 
    #print(raw_length)
    #print('-'*30)

    #DEBUG with less files
    file = {'chap1': ["God's is fake and (Well, I don't know.) Maybe another one? (what?)"],
            'chap2': ['Hello, (Get out! My God.)'],
            'chap3': ["I'll fight for the houses' sake. The kings' houses. Jesus' feet. Moses' disciples. Jews' religion. Religion, disciples, velvet."],
            'chap4': ["Paul a servant of God and God's death an apostle of Jesus Christ according to the faith of God's elect and the acknowledging of the truth which is after godliness in hope of eternal life which God that cannot lie promised before the world began."]}
    
    result = eval_surp(saved_model, raw, ctx=int(context_size)) #change "raw" to "file" for debug
    #print(result[:3])
    res = filter_subtokens(result, punctuations=set(string.punctuation))
    #print('--Total Chapter Length:') #debug
    #print(sum(norm_chapter_length))
    #print('-'*20)
    #print('--Chapter Lenght:')
    #print(len([d for s in res for d in s]))
    #assert sum(norm_chapter_length) == len(res)
    #for r in res:
    #    for c in r:
    #        print(c)
    surp_data = merge_and_pad(res)
    #print(surp_data) #debug
    surp_df = pd.DataFrame(surp_data, columns=['tokens', 'surprisal'])
    #print(surp_df) #debug

    #write to file
    if write:
        surp_df.to_csv(f'{LANG_ID}_surprisal_{mod.upper()}_{context_size}_DEBUG_PREPROCESS.csv', sep="\t")