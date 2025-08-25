from transformers import AutoTokenizer, BertForMaskedLM
import math
import torch
import numpy as np
import random
#from feature_class_pipe import ExtractFeatures
from feature_class_pipe import ExtractFeatures
import sys
from tqdm import tqdm
import string
from extract_surprisal_gpt2_pipe import filter_subtokens
import pandas as pd
import os


def merge_subtokens(tok_surp_pairs):
    """Input: [('a', 23), ('b', 22),...]"""
    parsed_toks = []
    for tupl in tok_surp_pairs:
        if tupl[0].startswith('##'):
            sdreh = tupl[0].replace('##', '')
            parsed_toks.append((sdreh, tupl[1]))
        else:
            sdreh = ' '+tupl[0]
            parsed_toks.append((sdreh, tupl[1]))
    #print('--Parsed Tokens without "##":') #DEBUG
    #print(parsed_toks) #DEBUG
    #print('-'*20) #DEBUG

    recon = []
    current_word = ""
    current_surprisal = []

    for i, (token, surprisal) in enumerate(parsed_toks):
        if token.startswith(" "):  # New word starts (space at the beginning)
            #print('Token:', (token, surprisal))
            #print(surprisal)
            #print('-'*20)
            if current_word:  # Store previous word
                #print('Current Word:', current_word)
                #print('Current Surprisal:', current_surprisal)
                #print('-'*20)
                recon.append((current_word, sum(current_surprisal))) #Averaging?
            current_word = token.strip() # Remove leading space
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


def merge_and_pad(result):
    SURP = {}
    for i in range(len(result)):
        #print('--Merge and Pad Input:') #DEBUG
        #print(result[i]) #DEBUG
        merged_tok = merge_subtokens(result[i]) #
        count = len(merged_tok)
        SURP[f'chapter_{i}'] = {'tokens': [], 'surprisal': [], 'length': ''}
        for tupl in merged_tok:
            SURP[f'chapter_{i}']['tokens'].append(tupl[0])
            SURP[f'chapter_{i}']['surprisal'].append(tupl[1])
        SURP[f'chapter_{i}']['length'] = count

    
    for i, (chap, (chap_name, chap_data)) in enumerate(zip(chapter_length, SURP.items())):
        chap_len = chap
        model_len = chap_data.get('length')
        diff = chap - model_len
        if diff != 0:
            SURP[chap_name]['tokens'].extend(['None']*diff)
            SURP[chap_name]['surprisal'].extend(['None']*diff)
        print((chap, model_len))
        print('-'*50)


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
        sent[f'chapter_{i}'] = sen #ADDED sentence[i]
        sent_length[f'chapter_{i}'] = len(' '.join(sentences[i]).split())
    return sent, sent_length


def eval_surp(saved_model, sent, ctx=512):
    """
    Takes a list of normalized sentences (not tokenized).
    """
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #added
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--Using device: {device}") #added
    
    tokenizer = AutoTokenizer.from_pretrained(saved_model)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = BertForMaskedLM.from_pretrained(saved_model) #load weights + config
    #print(tokenizer.convert_ids_to_tokens(0))
    #print(model.state_dict()) #deb
    #print('-'*30) #deb 
    #print(model.config) #deb
    #print('-'*30) #deb
    model.to(device) #added
    model.eval()
    surprisal_scores = []
    ctx = ctx
    print('Extracting Surprisal...')
    with torch.no_grad():
        for chapter, txt in tqdm(sent.items(), desc='Processing', unit='chapter'):
            chap = [] # CHAPTER-LEVEL SURP
            #for sent in txt: #ADDED for SENTENCE-LEVEL SURP
                #chap = [] #ADDED for SENTENCE-LEVEL SURP
                #print('--Sent:')
                #print(sent)
            inputs = tokenizer(txt, return_tensors='pt', #ADDED "sent" for SENTENCE-LEVEL_SURP
                               max_length=ctx, truncation=True, 
                               padding=True)
            
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs) # extract the logits
            logits = outputs['logits'] 
            inputs_ids = inputs['input_ids'][0]
            att = inputs['attention_mask'][0]
            #print('--Input shape:', inputs_ids.shape)
            #dec = tokenizer.convert_ids_to_tokens(inputs_ids, skip_special_tokens=False)
            #print(f'[INFO] total sentence tokens: {len(dec)}' )
            #print('--Decoded:')
            #print(dec)

            # masking tokens iteratively
            #for i in tqdm(range(len(inputs_ids)), desc='Processing', unit='tokens'):
            for i in range(len(inputs_ids)):
                masked_ids = inputs_ids.clone()
                masked_ids[i] = tokenizer.mask_token_id
                #print(masked_ids.shape)
                #decoded = tokenizer.convert_ids_to_tokens(masked_ids, skip_special_tokens=True) #debug
                #print('--Decoded:')
                #print(decoded) #debug
                #print('-'*20) ##debug
                masked_input = {"input_ids": masked_ids.unsqueeze(0).to(device),
                                "attention_mask": att.unsqueeze(0).to(device)}
                out = model(**masked_input)
                logits = out['logits']
                probs = torch.softmax(logits[0, i], dim=-1) # masked tokens
                token_id = inputs_ids[i]
                token_prob = probs[token_id].item() 
                surp = -math.log2(token_prob)
                token = tokenizer.convert_ids_to_tokens([token_id])[0]
                #print(token)
                #surprisal_scores.append((token, surp)) # changed
                chap.append((token, surp)) # add
                #print('--Chapter tuples:') # DEBUG
                #print((token, surp)) # DEBUG
            surprisal_scores.append(chap)
    #print('--Total Sentences:', len(surprisal_scores)) #DEBUG
    #print('--Total Subtokens:', sum([len(sen) for sen in surprisal_scores])) #DEBUG
    return surprisal_scores


def recontruct_contractions(result):
    """Recontruct contractions like don+'+t. Takes lists of tuples:
    [[(word, surp), (word, surp), ...], [(word, surp), ...]]
    Returns a list of tuples: [(tok, surp), (tok, surp),...]
    """
    filtered_data = []
    suffixes = ('s', 't', 'd', 're', 've', 'll')
    #skip_index = []
    for chapter in result:
        reconstructed = []
        i = 0
        while i < len(chapter):
            if i+2 < len(chapter):
                first, second, third = chapter[i], chapter[i+1], chapter[i+2]
                if "'" in second and third[0] in suffixes:
                #if (i+2 < len(chapter) and chapter[i+1] == "'" and chapter[i+2] in suffixes):
                    print(f'--subtoken: [{first[0]}] index: [{i}] surp: [{first[1]}]')
                    print(f'--elision: [{second[0]}] index: [{i+1}] surp: [{second[1]}]')
                    print(f'--endtoken: [{third[0]}] index: [{i+2}] surp: [{third[1]}]')
                    combined_token = first[0] + second[0] + third[0]
                    combined_surp = first[1] + second[1] + third[1]
                    print(f'--Reconstructed Token: [{combined_token}] surp: [{combined_surp}]')
                    reconstructed.append((combined_token, combined_surp))
                    i += 3  # Skip the next two because they're part of the contraction
                    continue
                elif "'" in second and third[0] not in suffixes:
                    print(f'--subtoken: [{first[0]}] index: [{i}] surp: [{first[1]}]')
                    print(f'--elision: [{second[0]}] index: [{i+1}] surp: [{second[1]}]')
                    print(f'--endtoken: [{third[0]}] index: [{i+2}] surp: [{third[1]}]')
                    combined_token = first[0] + second[0]
                    combined_surp = first[1] + second[1]
                    print(f'--Reconstructed Token: [{combined_token}] surp: [{combined_surp}]')
                    reconstructed.append((combined_token, combined_surp))
                    i += 2
                    continue
            reconstructed.append(chapter[i])
            i += 1
        get_rid_of_stuff = [tok for tok in reconstructed if tok[0] != "[SEP]" and tok[0] != "[CLS]"] #and tok[0] != "[PAD]"]
        filtered_data.append(get_rid_of_stuff)
    return filtered_data


if __name__ == '__main__':   
    models = {'bert': {512: "/Volumes/One Touch/MacAir-2025/5LN709/models_updated/bert-512/best_model", # I dont remeber if the final models are here or mot.
                       256: "/Volumes/One Touch/MacAir-2025/5LN709/models_updated/bert-256/model-256/best_model",
                       128: "/Volumes/One Touch/MacAir-2025/5LN709/models_updated/bert-128/model-128/best_model"
              }}

    LANG_ID = sys.argv[1] #ENGWWH
    mod = sys.argv[2] #gpt2 or bert
    context_size = int(sys.argv[3]) #512, 256, 128
    write = sys.argv[4] # bool
    saved_model = models.get(mod).get(context_size)
    print(f'Using Model: {mod.upper()} cw: {context_size}')
    print('-'*30)

    features = ExtractFeatures()
    #sentences = features.tok_sent # Normalized texts
    #norm_sent, norm_sent_length = get_sentences_dct(sentences) #input as dictsu 
    #norm_chapter_length = [length for chap, length in norm_sent_length.items()] #deb
    #print(norm_sent)
    #print('Chapter Lengths:') #deb 
    #print(sum(norm_chapter_length)) #deb 
    #print(norm_sent_length)
    #print('-'*30)

    raw_sentences = [bible.get('orth').get('txt') for bible in features.raw_orth] # RAW texts
    raw, raw_length = get_sentences_dct(raw_sentences) # RAW TEXTS as dcts
    chapter_length = [length for chap, length in raw_length.items()] #deb
    #print('--Chapter Lengths:') #debug
    #print(f' {sum(chapter_length)} tokens') #debug
    #print(raw_length)
    #print('-'*30)


    # DEBUG LESS SENTENCES
    file = {'chap1': ["I shouldn't: love this! Iscariot, you're undoubtely transparent people from my country don't hasn't"],
            #'chap2': ["This thing doesn't make god lucre wherefore art thou!!"],
            #'chap3': ['Not given to filthy lucre.',  'What the fuck you are talking about'],
            'chap4': ["I'll fight for the houses' sake. The kings' houses. Jesus' feet. Moses' disciples. Jews' religion. Religion, disciples, velvet."],
            'chap5': ["Paul a servant of God and God's death an apostle of Jesus Christ according to the faith of God's elect and the acknowledging of the truth which is after godliness in hope of eternal life which God that cannot lie promised before the world began."]}
    
    result = eval_surp(saved_model, raw, ctx=int(context_size)) # file= debug
    #print('--Result:') #debug
    #print(result) #debug
    #print('-'*30) #debug
    result_contr = recontruct_contractions(result)
    #print('--Reconstructed:') #debug
    #for i, s in enumerate(result_contr): #debug
    #    print(f'--Chapter {i}:')
    #    for ss in s: 
    #        print(ss)
    #    print('-'*30)

    final = filter_subtokens(result_contr, punctuations=set(string.punctuation))
    #print('--Filtered Punctuation:') #debug
    #print(final) #debug
    #print('-'*20) #debug

    surp_data = merge_and_pad(final)
    #print(sum([len(s) for s in surp_data]))
    #print('--Merged and padded:')
    #for pad in surp_data:
    #    print(pad)

    if write:
        print('--Writing CSV...')
        file_name = f'{LANG_ID}_surprisal_{mod.upper()}_{context_size}_DEBUG_CW.csv' #Change title
        root = "/Volumes/One Touch/MacAir-2025/5LN709/surprisal_debug_preprocess"
        surp_df = pd.DataFrame(surp_data, columns=['tokens', 'surprisal'])
        #print(surp_df) #debug
        
        #write to file
        if os.path.exists(os.path.join(root, file_name)):
            message = input(f'Filename "{file_name}" already exists. Overrride (y/n)? ')
            if message.lower() == 'y':
                surp_df.to_csv(os.path.join(root, file_name), sep='\t')
            elif message.lower() == 'n':
                print('Try again.')
        else:
            surp_df.to_csv(os.path.join(root, file_name), sep='\t')
        print(f'--File "{file_name}" saved to path.')

