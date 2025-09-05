import subprocess
import os
import sys
import sys
from read_maus_output import get_url, download_textgrid, read_output


def align_files(audio_lst, text_lst): 
    """
    Aligns .wav files to their respective .par outputs. 
    Return: [(file1.wav, file1.par), (file2.par, file2.par), ...]
    """  
    couples = []
    for audio, txt in zip(sorted(audio_lst), sorted(text_lst)):
        couples.append((audio, txt))
    return couples


def maus_wrapper(audio_par):
    command = ['curl', '-v', 
            '-X', 'POST',
            '-H', 'content-type: multipart/form-data',
            '-F', f'SIGNAL=@{audio_par[0]}',
            '-F', 'LANGUAGE=eng-US', #rus-RU, spa-ES...
            '-F', 'MODUS=align',
            '-F', 'INSKANTEXTGRID=true', 
            '-F', 'RELAXMINDUR=false', 
            '-F', 'OUTFORMAT=TextGrid',
            '-F', 'TARGETRATE=100000',
            '-F', 'ENDWORD=999999',
            '-F', 'RELAXMINDURTHREE=true', # option: default "false"
            '-F', 'STARTWORD=0',
            '-F', 'INSYMBOL=sampa',
            '-F', 'PRESEG=false',
            '-F', 'USETRN=false', 
            '-F', f'BPF=@{audio_par[1]}',
            '-F', 'MAUSSHIFT=default',
            '-F', 'INSPROB=0.0',
            '-F', 'INSORTTEXTGRID=true',
            '-F', 'OUTSYMBOL=sampa',
            #'-F RULESET=@<filename>',
            '-F', 'MINPAUSLEN=5',
            '-F', 'WEIGHT=default',
            '-F', 'NOINITIALFINALSILENCE=false',
            '-F', 'ADDSEGPROB=false',
            "https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUS"
    ]

    process = subprocess.run(command, capture_output=True, text=True, encoding='latin-1')

    # Save the MAUS response to a .txt file
    output_file = audio_par[0].split('.')[0]
    print('Output File:')
    print(output_file)
    print('_'*40)
    output_file_name = f'{output_file}_MAUS_OUT.txt'

    with open(output_file_name, "w") as f:
        f.write(process.stdout)

    #print("STDOUT:", process.stdout)
    #print("STDERR:", process.stderr)

    # Save TextGrid to a file
    output = read_output(output_file_name)
    url = get_url(output)
    download_textgrid(url[0], output_file)


if __name__ == '__main__':
    bible_id = sys.argv[1]
    res = "/Volumes/One Touch/MacAir-2025/5LN709/audio" #ROOT
    audio = f'/{bible_id}/audio' #.wav FOLDER with trimmed audio
    par = f'/{bible_id}/par' # .par FOLDER
    audio_path = res+audio
    par_path = res+par
    print('--Audio path:')
    print(audio_path)
    print('-'*40)
    print('--Par path:')
    print(par_path)
    print('-'*40)

    # Loop over .wav files
    audio_lst = [os.path.join(audio_path, fil) for fil in os.listdir(audio_path) if fil.endswith(".wav")]
    # Loop over .par files
    par_lst = [os.path.join(par_path, fil) for fil in os.listdir(par_path) if fil.endswith(".par")]
    print('--Audio files:')
    print(sorted(audio_lst))
    print('-'*40)
    print('--Par files:')
    print(sorted(par_lst))
    print('-'*40)

    aligned = align_files(audio_lst, par_lst)
    print('--Aligned Files:')
    print(aligned)
    print('-'*40)

    for file_tpl in aligned:
        maus_wrapper(file_tpl)
