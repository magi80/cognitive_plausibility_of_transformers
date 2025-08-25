import os
import subprocess

def list_files(directory):
    """
    List all MP3 files in a given directory.
    """
    file_names = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith('B') and f.endswith("trimmed.mp3")] #f.startswith if you're using external hard drive
    return file_names

def convert_to_wav(input_file):
    """
    Convert an MP3 file to WAV using sox.
    """
    input_splt1 = input_file.split('-') 
    input_splt2 = input_splt1[0]+'-'+input_splt1[1]
    #print(input_splt2)
    #output_file = os.path.splitext(input_file)[0] + ".wav"
    output_file = input_splt2 + ".wav"
    #output_file = input_file + ".wav"
    print(f'Output file_wav: {output_file}')
    
    try:
        #command = ['sox', input_file, '-c', '1', '-b', '16', output_file, 'gain', '-n']
        res = subprocess.run(['sox', '-V3', input_file, '-c', '1', output_file], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, check=True)
        #subprocess.run(["sox", input_file, output_file], check=True)
        #subprocess.run(command, check=True)
        #print(f"Converted: {output_file}")
        #return output_file
        print(res.stdout)
        print(res.stderr)
    except subprocess.CalledProcessError:
        print(f"Error converting {input_file} to WAV!")


def main(absolute_path):
    """
    Convert all MP3 files in a folder to WAV.
    """
    #folder_name = 'wav'
    #if not os.path.exists(folder_name):
    #    os.mkdir(folder_name)
        
    file_names = list_files(absolute_path)
    for file in file_names:
        convert_to_wav(file)

if __name__ == "__main__": 
    # Path to FOLDER with trimmed audio files
    absolute_path = "/Volumes/One Touch/MacAir-2025/5LN709/audio/ENGKJV/audio_trim_last" #ENG
    #absolute_path = "/Volumes/One Touch/audio/audio/Russian_rus_DPI_NT_Non-Drama" #RUS
    #absolute_path = "/Volumes/One Touch/audio/audio/Spanish_spa_WTC_NT_Non-Drama" #SPA
    main(absolute_path)