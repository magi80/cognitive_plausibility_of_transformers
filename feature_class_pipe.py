import sys
from articulation_rate_pipe import get_duration
from orth_class_pipe import OrthographicText
import pandas as pd
import os
from collections import defaultdict
#from nltk import sent_tokenize


class ExtractFeatures(OrthographicText):

    def __init__(self):
        super().__init__()
        self.language = sys.argv[1] # ENGKJV
        self.tg = self._get_tg_files()
        self.word_duration = self._get_duration()
        self.ar = self._get_articulation_rate()
        self.word_length = self._get_word_length()
        self.dummies, self.tok_sent, self.chapters = self._get_word_position()


    def _get_articulation_rate(self):
        """
        Extract articulation rate (AR).
        """
        tuple_pairs = self.word_duration
        ar_rates = defaultdict(list)
        #for bible in tuple_pairs:
        for filename, word_dur_tpl in tuple_pairs.items():
            for tpl in word_dur_tpl[0]: # [0] because is nested
                syllables = tpl[0].count('.')
                syllables = syllables + 1 
                ar = (syllables/tpl[1]) * 1000 # total n_syllables / word_dur per seconds
                ar_rates[filename].append((tpl[0], ar))
        return ar_rates
    
    
    def _get_word_length(self):
        """
        Count the periods and get WL.
        """
        tuple_pairs = self.word_duration
        w_length = defaultdict(list)
        for filename, word_dur_tupl in tuple_pairs.items():
            for tpl in word_dur_tupl[0]:
                syllables = tpl[0].count('.')
                syllables = syllables + 1
                w_length[filename].append((tpl[0], syllables))
        return w_length
 

    def _get_word_position(self):
        """
        Assign a variable between 0 and 1 across sentences.
        First word=0, last word=1.
        """
        dummy_variables = []
        tok_sent = []
        chapters = []
        norm_texts = self.norm_orth # get normalized texts
        for bible in norm_texts:
            bible_chapter = bible.get('orth').get('txt')
            bible_chap_num = bible.get('orth').get('chapter') #added chapter number
            bible_name = bible.get('bible') # ADDED 2025-6-8 
            #print('--Bible Name:')
            #print(bible_name)
            #print('Bible Chapter:') #debug
            #print(bible_chapter) #debug
            #print('_'*20) #debug
            split_sent = [sen.split() for sen in bible_chapter] #[[tok, tok, ...], [tok, tok,...]]
            tok_sent.append(bible_chapter)
            for sent in split_sent:
                n = len(sent)
                #print(sent, len(sent)) #debug
                variables = [round((i / (n - 1)), 3) for i in range(n)]
                dummy_variables.append(list(zip(sent, variables))) 
                chapters.append((bible_chap_num, sent, bible_name)) #added
        #print(chapters) #debug
        return dummy_variables, tok_sent, chapters


    def _get_tg_files(self):
        """
        Loop over TextGrids in the specified folder.
        """
        if self.language == 'ENGKJV':
            tg_path = f"/Volumes/One Touch/MacAir-2025/5LN709/audio/{self.language}/text_grid_final" # For hard drive text_grid_fixed = RELAXMINDUR, text_grid_final=RELAXMINDURTHREE
        else: 
            tg_path = f"/Volumes/One Touch/MacAir-2025/5LN709/audio/{self.language}/text_grid" # For hard drive text_grid_fixed = RELAXMINDUR, text_grid_final=RELAXMINDURTHREE

        tg_filename = [os.path.join(tg_path, fil) for fil in os.listdir(tg_path) if fil.endswith('.TextGrid')]
        #print('TextGrid Files:') #debug
        #print(tg_filename) #debug
        #print(len(tg_filename)) #debugœ
        return sorted(tg_filename)
    

    def _get_duration(self):
        word_dur = defaultdict(dict)
        file_lst = self._get_tg_files()
        assert len(file_lst) == 260, 'The number of audio files and TextGrids mismatches.'
        for tg in file_lst:
            dur = get_duration(tg)
            word_dur[tg.split('/')[-1]] = dur
        return word_dur


    def _get_statistic(self):
        """Print stuff to debug stuff."""
        articulation_rate = self.ar
        word_len = self.word_length
        word_pos = self.dummies
        tokenized_sent = self.tok_sent

        #########################################################
        print('#'*27)
        mess = f'# Statistics for "{self.language}" #'
        print(mess)
        print('#'*len(mess))
        print('-'*50)
        print('--Total Number Chapters:', len(tokenized_sent))
        print('-'*50)
        sentenz = []
        for i in range(len(tokenized_sent)):
            chapter = tokenized_sent[i]
            print(chapter)
            count = 0
            for sentence in chapter:
                count += len(sentence.split())
            #n_tokens = len(' '.join([word for word in chapter.split()]))
            counts = {'chapter': i+1, 'tokens': chapter, 'length': count, 'n_sent': len(chapter)}
            sentenz.append(counts)    
        print('--Total TOKENS per chapter:')
        counts = 0
        for dicts in sentenz:
            print(f"Chapter: {dicts['chapter']} Total length: {dicts['length']} Total Sentences: {dicts['n_sent']}")
            counts += dicts['length']
        print('-'*50)
        print('--Total Token Count:', counts)
        
        #########################################################

        print('-'*50)
        print('--Total SAMPA tokens (AR):')
        for file, data1 in articulation_rate.items():
            print(f'File: {file} Total SAMPA token: {len(data1)}')
        
        #########################################################

        print('-'*50)
        print('--Total LENGTH tokens (WL):')
        count = 0
        for file, data1 in word_len.items():
            for tupl in data1:
                count += int(tupl[1])
            #count += int(data1)
            print(f'File: {file} Total LENGTH token: {len(data1)}')
        print(f'Total SYLLABLES: {count}')
        #########################################################

        print('-'*50)
        print('--WORD POSITION (WP) (=23684 tokens for Matthew; =180384 for the whole NT):')
        #for i in range(len(word_pos)):
        count = 0
        for d in word_pos:
            count += len(d)
        print(f'Total TOKENS: {count}')


    def _get_chapters(self, chap_and_sent):
        chap_data = []
        for sentence in chap_and_sent:
            numch = sentence[0]
            bib = sentence[2] 
            for sen in sentence[1]:
                chap_data.append((numch, sen, bib))
        
        num_chapter = {'bible': [], 'chapter': [], 'tokens': []}
        for data in chap_data:
            num_chapter['bible'].append(data[2]) # Added
            num_chapter['chapter'].append(data[0])
            num_chapter['tokens'].append(data[1])
        #print(num_chapter)
        return chap_data
    

    def main(self, write_to_file=False, statistics=False):
        if statistics:
            self._get_statistic()
        articulation_rate = self.ar
        word_len = self.word_length
        word_pos = self.dummies

        #articulation rate
        ar_data = {'sampa': [], 'ar': []}
        for file, data in articulation_rate.items():
            for tupl in data:
                ar_data['sampa'].append(tupl[0])
                ar_data['ar'].append(tupl[1])
        AR = pd.DataFrame(ar_data)
        #assert AR.shape == (180384, ), 'Error: should be 180384 tokens for the whole NT.'
        print('Articulation Rate Dataset:')
        print(AR)
        print('-'*20)
        
        #word length
        wl_data = {'sampa': [], 'wl': []}
        for file, data in word_len.items():
            for tupl in data:
                wl_data['sampa'].append(tupl[0])
                wl_data['wl'].append(tupl[1])
        WL = pd.DataFrame(wl_data)
        #assert WL.shape  == (180384, ), 'Error: should be 180384 tokens for the whole NT.'

        # WORD POSITION
        # UPDATED 2025-4-15 with columns first and last
        wp_data = {'tokens': [], 'first': [], 'last': []}
        for sentence in word_pos:
            for tupl in sentence:
                wp_data['tokens'].append(tupl[0])
                if tupl[1] == 0.0:
                    #print(tupl) #deb
                    wp_data['first'].append(1)
                    wp_data['last'].append(0)
                elif tupl[1] == 1.0:
                    #print(tupl) #deb
                    wp_data['first'].append(0)
                    wp_data['last'].append(1)
                else:
                    wp_data['first'].append(0)
                    wp_data['last'].append(0)
        #print(len(wp_data['tokens'])) #deb
        #print(len(wp_data['first'])) #deb
        #print(len(wp_data['last'])) #deb
        WP = pd.DataFrame(wp_data, columns=['tokens', 'first', 'last'])
        
        print('Word Position Dataset:')
        print(WP)
        print('-'*20)
        print('Word Length Dataset:')
        print(WL)
        print('-'*20)

        #Chapters
        ch = self.chapters
        num_chapter = self._get_chapters(ch)
        #print(num_chapter) #debug
        CH = pd.DataFrame(num_chapter, columns=['chapter', 'tokens', 'bible']) # added bible
        #print('--Chapters:') #debug
        #print(CH) #debug

        print(f"Lengths - AR: {len(AR)}, WL: {len(WL)}, WP: {len(WP)}, CH: {len(CH)}")
        print('-'*20)
        assert len(AR) == len(WL), "Mismatch: AR and WL length."
        assert len(WP) == len(AR), "Mismatch: WP and AR length."
        assert len(CH) == len(WP), "Mismatch: CH and WP length."
        if len(AR) == len(WL) and len(AR) == len(WP) and len(WP) == len(CH):
            print(f'--Data for "{self.language}" is aligned. Saving...')
            print('-'*20)
        #print("Sample WP tokens:", WP['tokens'][:10])
        #print("Sample AR sampa:", AR['sampa'][:10])


        all_data = {'tokens': [], 'sampa': [], 'ar': [], 
                    'first': [], 'last': [], 'wl': [], 'bib': [], 'chap': []
                    }
        for tokens, sampa, ar, first, last, wl, bib, chap in zip(WP['tokens'], 
                                                            AR['sampa'], 
                                                            AR['ar'], 
                                                            WP['first'], 
                                                            WP['last'],
                                                            WL['wl'], 
                                                            CH['bible'],
                                                            CH['chapter']
                                                            ):
            all_data['tokens'].append(tokens)
            all_data['sampa'].append(sampa)
            all_data['ar'].append(ar)
            all_data['first'].append(first)
            all_data['last'].append(last)
            all_data['wl'].append(wl) 
            all_data['bib'].append(bib) #
            all_data['chap'].append(chap)

        ALL_DATA = pd.DataFrame(all_data)
        print(ALL_DATA)

        if write_to_file:
            mess = input(f'--Write data for "{self.language}" to file (y/n)? ') # extra safe
            if mess.lower() == 'y':

                storage_folder = f'{self.language}_csv_storage'
                filepath = os.path.join(self.root_path, storage_folder)
                if not os.path.exists(filepath):
                    os.mkdir(filepath)

                AR.to_csv(os.path.join(filepath, f'{self.language}_AR_RATE-rmd.csv'), sep='\t', columns=['sampa', 'ar'])
                WP.to_csv(os.path.join(filepath, f'{self.language}_WPOS-rmd.csv'), sep='\t', columns=['tokens', 'first', 'last'])
                WL.to_csv(os.path.join(filepath, f'{self.language}_WLEN-rmd.csv'), sep='\t', columns=['sampa', 'wl'])
                CH.to_csv(os.path.join(filepath, f'{self.language}_CHAP-rmd.csv'), sep='\t', columns=['chapter', 'tokens']) #added
                ALL_DATA.to_csv(os.path.join(filepath, f'{self.language}_ALL_DATA-rmd.csv'), sep='\t', columns=['tokens', 'sampa', 'ar', 'first', 'last', 'wl', 'bib', 'chap'])
                print('--Saving completed.')

            elif mess.lower() == 'n':
                print(f'--Skipping writing data for "{self.language}".')
            else:
                print('--Invalid input. Please enter "y" or "n".')
        #debug = pd.DataFrame({'tokens': all_data['tokens'],
        #                      'sampa': all_data['sampa'],
        #                      'wl': all_data['wl']})
        #debug.to_csv('DEBUG_WL.csv', sep='\t')
        return ALL_DATA

 
if __name__ == '__main__':
    model = ExtractFeatures()
    model.main(write_to_file=False, 
               statistics=False)