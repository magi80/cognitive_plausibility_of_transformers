import subprocess
import requests
import os
import sys
from read_maus_output import read_output, get_url, download_par_file


def get_par_files(txt): #txt = /Users/matteo/Desktop/ENGWWH/text/B01___08_Matthew_____EN1WEBN2DA.txt
    """
    Takes an file name as argument: <<B01_1_Matthew_LANG_ID.txt>>
    """
    command = ['curl', '-v', 
           '-X', 'POST', 
           '-H', 'content-type: multipart/form-data',
           '-F', 'com=no',
           #'-F', 'tgrate=16000', 
           '-F', 'stress=no',
           #'-F', 'imap=@<filename>', 
           '-F', 'lng=eng-US', #spa-ES, rus-RU...
           '-F', 'lowercase=yes',
           '-F', 'syl=yes',
           '-F', 'outsym=sampa',
           '-F', 'nrm=yes', 
           '-F', f'i=@{txt}', 
           #'-F', 'tgitem=ort',
           '-F', 'align=no', 
           '-F', 'featset=standard',
           '-F', 'iform=txt', 
           #'-F', 'except=@<filename>', 
           '-F', 'embed=no', 
           '-F', 'oform=bpf', 
           'https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runG2P']

    process = subprocess.run(command, capture_output=True, text=True, encoding='latin-1')

    # Save the response of the GP2 service 
    # to a file "FILE_TXT_NAME_output.txt"
    output_file = txt.split('.')[0]  #.../text/B01_1_Matthew_EN1WEBN2DA
    print(f'Absolute path of the .txt file without extention: "{output_file}"') 
    output_file_name = f'{output_file}_OUTPUT.txt' #.../text/B01_1_Matthew_EN1WEBN2DA_OUTPUT.txt
    print(f'Absolute path of the file name modified to "{output_file_name}"') 

    with open(output_file_name, "w") as f: #.../text/B01_1_Matthew_EN1WEBN2DA_OUTPUT.txt
        f.write(process.stdout)

    output = read_output(output_file_name)
    #print(f'G2P response: {output}') #debug
    url = get_url(output)
    print(f'Downloading: "{url[-1]}"')
    download_par_file(url[0], output_file)

    # Print output
    #print("STDOUT:", process.stdout)
    #print("STDERR:", process.stderr)


def extract_all_pars(text_files_path):  #/Users/matteo/Desktop/ENGWWH/text
    """Loop over the list of .txt filenames. Call the G2P service for each
     text. Write the .par files to ROOT. """
    file_names = list_files(text_files_path) #B01___08_Matthew_____EN1WEBN2DA.txt
    for txt in file_names:
        path = text_files_path+'/'+txt #/Users/matteo/Desktop/ENGWWH/text/B01___08_Matthew_____EN1WEBN2DA.txt
        #path = par_folder+'/'+txt
        print('-'*70)
        print(f'--Extracting: "{txt}"')
        get_par_files(path)


def list_files(text_files_path):
    """Take the ROOT path of the stored .txt files. Returns 
    a list of .txt filenames."""
    files = [txt for txt in os.listdir(text_files_path) if txt.endswith('.txt')]
    #assert len(files) == 28 # Debug
    return files


if __name__ == "__main__":
    LANG_ID = sys.argv[1] # LANG_ID (e.g. ENGWWH)
    text_files_path = f"/Volumes/One Touch/MacAir-2025/5LN709/audio/{LANG_ID}/text" #ROOT
    extract_all_pars(text_files_path)
   












