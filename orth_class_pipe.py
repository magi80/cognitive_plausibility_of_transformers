from bs4 import BeautifulSoup
import requests
import json
import sys
import os
import time
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize
import string
import re
import copy

#PROVA_WRAPPER 2
#implement normalization

class OrthographicText:

    def __init__(self, write=False, test=False, open_json=True):
        self.language = sys.argv[1] # Bible ID-code: ENGKJV
        self.audio_url = f"https://live.bible.is/bible/{self.language}/MAT/1"
        self.root_path = "/Users/matteo/Desktop/HT2024/5LN709_master_thesis/PIPELINE_REV" 
        self.audio_data = {'ENGKJV': "/Volumes/One Touch/audio/audio/English_eng_KJV_NT_Non-Drama" # change the audio folder
                           #'RUSDPI': "/Volumes/One Touch/audio/audio/Russian_rus_DPI_NT_Non-Drama", # optional language
                           #'SPAWTC': "/Volumes/One Touch/audio/audio/Spanish_spa_WTC_NT_Non-Drama", # optional language
                           }
        self.write = write
        self.test = test
        self.open_json = open_json
        self.audio_names, self.norm_orth, self.raw_orth = self.get_orth_texts_dict()
       
        if self.write:
            self.write_to_path()


    def make_soup(self, path):
        """
        Make a Beautiful soup from Bible.is orthographies. 
        Returns a parsed html webpage conaining all information.
        """
        try:
            txt = requests.get(path)
            txt.raise_for_status() # CHANGED TRY/EXCEPT
        except requests.exceptions.RequestException as err:
            print(f"[ERROR] Failed to fetch URL: {path}\n{err}")
            return None

        txt  = txt.content
        soup = BeautifulSoup(txt, 'html.parser')
        script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
        if script_tag:
            try:
                json_data = json.loads(script_tag.string) # Parse JSON
                return json_data
            except json.JSONDecodeError as error:
                print(f"[ERROR] JSON decode failed for URL: {path}\n{error}")
                return None
        else:
            print("No script tag with id='__NEXT_DATA__' found.") 
            return None


    def get_bible_ids(self, json_data):
        """
        Filters the HTML data by extracting the dictionaries 
        related to the New Testament books and their chapter indices. 
        Returns a list of dictionaries.
        """
        nt = []
        data = json_data.get('props').get("pageProps").get('books') # Access the nested dictionary 'books'
        new_test = [book for book in data if book['testament'] == 'NT']
        for book in new_test:
            bible = {'bible': book['book_id'], # CHANGE
                     'name': book['name'],
                     'chapters': book['chapters'], 
                     'book_order': book['book_order'], 
                     'nt_order': book['testament_order']
                     }
            #print(bible) #debug
            nt.append(bible)
        return nt


    def get_bible_urls(self, nt, lan):
        """
        This wonderful function will exptract all the urls of the
        format /LANG_CODE/BIBLE_BOOK/chapter number.
        """
        bible_urls = []
        for bible in nt:
            bible_order = bible['book_order'] # CHANGE
            bible_name = ''.join(bible['name'].split()) # BO1 ...
            bible_id = bible['bible'] # MAT, 1CO, REV ...
            chapters = bible['chapters']
            for chap in chapters:
                url = f"{self.audio_url}/{bible_id}/{chap}"
                bible_urls.append((bible_order, bible_name, chap, url))
        return bible_urls


    def extract_orthographies(self, json_data):
        """
        Returns a tuple with 1) a list of file names withput extension and
        2) a nested dictionary wit the format:
        [{'language_code': CODE, 'bible_name': BIBLE; orthographic_text: {'chapter': NUM, 'text': TEXT}}]
        """
        orths = []
        start_time = time.time()
        mess = f'Making dictionary entries for "{self.language}"...'
        print(mess)
        print('-'*len(mess))
        for tupl_data in tqdm(json_data, desc='Processing', unit='chapter'):
            json_content = self.read_json(tupl_data[0])
            nt_split = tupl_data[1].split('_')
            lan, bib, chap, order = nt_split[0], nt_split[-2], nt_split[2], nt_split[1] # change

            entry = {'lan': lan, 
                     'bible': bib, 
                     'order': order, 
                     'orth': {'chapter': chap, 'txt': []}}

            verses = json_content.get("props", {}).get("pageProps", {}).get("chapterText", [])
            print(f"Chapter {chap}: {len(verses)} verses from {tupl_data[0]}") # Added

            for verse in verses:
                entry['orth']['txt'].append(verse['verse_text'])
            orths.append(entry)
        end_time = time.time()
        print('-'*len(mess))
        print(f"Done. Execution time: {end_time - start_time:.5f} seconds")
        return orths


    def file_format(self, path):
        """
        Makes a list of filenames with format B01_01_Matthew_CODE.
        Assuring same file format for MAUS.
        """
        if not os.path.isdir(path):  # Check if path is a directory
            raise Exception(f'Error: invalid path "{path}".')
        files = os.listdir(path)
        audio_files = [f for f in files if f.startswith('B') and f.endswith('A.mp3') if os.path.isfile(os.path.join(path, f))]
        audio_files = [file.split('.')[0] for file in audio_files]
        return sorted(audio_files)
    

    def write_to_file(self, filepath, txt):
        """
        Write to folder.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(' '.join(txt))
  

    def get_orth_texts_dict(self):
        #soup_dict = self.make_soup(self.audio_url)
        #new_testament_dict = self.get_bible_ids(soup_dict)
        #new_testament_urls = self.get_bible_urls(new_testament_dict, self.language)
        
        audio_path = self.audio_data.get(self.language)
        audio_names = self.file_format(audio_path)
        #print(f'--Audio names: {len(audio_names)}') #DEBUG
        #print(f'--Testament Urls: {len(new_testament_urls)}') #DEBUG
        assert (len(audio_names)) == (len(new_testament_urls)), f"Warning: the number of audio files ({len(audio_names)}) and urls ({len(new_testament_urls)}) mismatches."
        
        if self.open_json:
            folder = f'{self.language}-JSON'
            folder_path = os.path.join(self.root_path, folder)
            print(f'--Folder "{self.language}-JSON" Exist. Opening json...')
            json_file_list = os.listdir(folder_path)
            #print(sorted(json_list))
            new_testament_urls = []
            for js_name in sorted(json_file_list):
                js_path = os.path.join(folder, js_name)
                #print(js_path)
                new_testament_urls.append((js_path, js_name))
            #assert len(new_testament_urls) == 260, "Doesn't match 260 audio files."

        print(f'--Audio names: {len(audio_names)}')
        print(f'--Testament Urls: {len(new_testament_urls)}')
        #assert (len(audio_names)) == (len(new_testament_urls)), f"Warning: the number of audio files ({len(audio_names)}) and urls ({len(new_testament_urls)}) mismatch."

        # Try less files
        if self.test:
            less_urls = [url for url in new_testament_urls if 'REV' in url[0]]
            raw_texts = self.extract_orthographies(less_urls) # less_urls for less data
        else:
            raw_texts = self.extract_orthographies(new_testament_urls) # less_urls for less data
        raw_copy = copy.deepcopy(raw_texts)
        norm_texts = self._normalize(raw_copy)    
        return audio_names, norm_texts, raw_texts


    def _normalize(self, orth_dct):
        regex = r"\b(\w+'(s|t|d|ve|re|ll))\b" # match contractions 'don't', 'hasn't' etc
        contr = []
        #normalized = []
        for dct in orth_dct:
            normalized = []
            d = dct.get('orth').get('txt')
            for txt in d:
                txt = ' '.join(txt.split())
                txt = txt.replace('“', "").replace("”", "").replace("’", "'").replace("‘", "") #remove quotation marks

                # Spanish and Russian need these extra steps before tokenizing
                #if self.language == 'SPAWTC':
                #    txt = txt.replace('¡', "").replace("¿", "").replace("»", "'").replace("«", "").replace("—", "").replace("(", "").replace(")", "")
                #elif self.language == 'RUSDPI':
                #    txt = txt.replace("—", "") 
                #print('--Text with replaced characters:')
                #print(txt)
                #print('_'*20)
                
                txt = sent_tokenize(txt)
                #print('--Text using sentence tokenizer:') #DEBUG
                #print(txt) #DEBUG
                for sent in txt:
                    match = re.search(regex, sent)
                    if match:
                        contr.append(match.group(1))
    
                norm_sent = []
                for sent in txt:
                    tokenized = sent.split()
                    for word in tokenized:
                        if word in contr:
                            norm_sent.append(word.lower()) # Add lowercase
                        else:
                            #print('Word Before:', word) #Debug
                            w = [w.lower() for w in word if w not in string.punctuation]
                            #print('Word Whoa:', ''.join(w)) #Debug
                            norm_sent.append(''.join(w)) # Add lowercase
                normalized.append(norm_sent) 
            dct['orth']['txt'] = [' '.join(sent) for sent in normalized]
        return orth_dct


    def write_to_path(self):
        """
        Write .txt files in path "ROOT/LANG-ID/text".
        """
        destination_path = "/Volumes/One Touch/MacAir-2025/5LN709/audio" # change to ROOT/LANG_ID/text

        folder = f'{self.language}' # ENGKJV, ENGWWH...
        folder_name = os.path.join(destination_path, folder)
        
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        else:
            pass

        text_folder = os.path.join(folder_name, 'text')
        if not os.path.exists(text_folder):
            os.mkdir(text_folder)
        else:
            pass

        print('Writing to file...')
        #for audio, txt in zip(self.orth[0], self.orth[1]):
        for audio, txt in zip(self.audio_names, self.norm_orth):
            #chapter = txt.get('orth').get('chapter')
            file_name = f'{audio}'+'.txt'
            file = os.path.join(text_folder, file_name)
            orth = txt.get('orth').get('txt')
            self.write_to_file(file, orth)
        print('Writing Done.')


    def read_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return d


if __name__ == "__main__":
    model = OrthographicText(write=False, test=False, open_json=True) # write=TRUE for writing 260 .txt files
    orth = model.norm_orth
    raw = model.raw_orth
    print('--Raw Texts:'+'\n')
    print(raw[0])
    print('-'*20)
    print('--Normalized Texts:'+'\n')
    print(orth[0])
    print('-'*20)
    print(model.audio_names[:1])

    #DEBUG
    #for an, txt in zip(model.audio_names, raw):
    #    print(an+'\n', txt.get('orth').get('txt'))
