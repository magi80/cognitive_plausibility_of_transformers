import subprocess
import sys
from convert_to_wav_pipe import list_files, convert_to_wav


class AudioSegment:

    def __init__(self):
        self.audio_folder = sys.argv[1] # Takes a folder of raw .mp3 files
        #self.audio_folder = "/Volumes/One Touch/audio/audio/Russian_rus_DPI_NT_Non-Drama" #rus
        #self.audio_folder = "/Volumes/One Touch/audio/audio/Spanish_spa_WTC_NT_Non-Drama" #spa


    def audio_segment(self, audio_file, trim_last=False):
        #audio_file = self.audio_file
        result = subprocess.run(['ffmpeg', '-i', f'{audio_file}',
                         '-af', 'silencedetect=n=-16dB:d=0.5', # best -16dB
                         '-f', 'null', '-'], stderr=subprocess.PIPE, text=True)
        silence = [line for line in result.stderr.split('\n') if 'silence_start' in line or 'silence_end' in line]
        print('--Intervals:')
        for line in result.stderr.split('\n'):
            print(line)
        
        output = self.process_output(silence)
        intervals = self.get_silence(output)
        print('--Intervals:')
        print(intervals)
        cut_point = self.get_cut_point(intervals)

        trimmed_file = audio_file.split('.')[0]
        trimmed_file = trimmed_file + '-trimmed.mp3'
        #print(trimmed_file)

        if trim_last:
            last_two_pauses = intervals[-2]
            paus_start, paus_end = last_two_pauses.get('silence').get('start'), last_two_pauses.get('silence').get('end')
            print((paus_start, paus_end))
            paus_cut = (float(paus_start) + float(paus_end)) / 2
            paus_cut_point = round(paus_cut, 6)
            print('--Last CutPoint (sec):')
            print(paus_cut_point)
            print('-'*20)
            segment_file = subprocess.run(['ffmpeg', '-i', 
                                           f'{audio_file}',
                                           '-ss', f'{cut_point}',
                                           '-to', f'{paus_cut_point}',
                                           '-c', 'copy', f'{trimmed_file}'])

        else:
            segment_file = subprocess.run(['ffmpeg', '-i', 
                                           f'{audio_file}',
                                           '-ss', f'{cut_point}',
                                           '-c', 'copy', f'{trimmed_file}'])
        return segment_file


    @staticmethod
    def process_output(silence):  
        time_slices = []
        for line in silence: # No [:6]
            lin = line.split()
            #print(lin)
            if 'silence_start:' in lin:     
                time_slices.append(lin[-2:])
            elif 'silence_end:' in lin:
                time_slices.append(lin[3:5])
                time_slices.append(lin[-2:])
        return time_slices


    @staticmethod
    def get_silence(time_slices):  
        stamp = []
        for i in range(0, len(time_slices), 3):
            info = {'pause': int, 'silence': {'start': float, 'end': float, 'dur': float}}
            info['pause'] = i
            info['silence']['start'] = float(time_slices[i][1])
            info['silence']['end'] = float(time_slices[i+1][1])
            info['silence']['dur'] = float(time_slices[i+2][1])
            stamp.append(info)
        return stamp


    @staticmethod
    def get_cut_point(stamp):
        """Detects pauses at the beginning of a file. RUS=4 pauses for each .wav, no trim_last
        SPA=3 pauses begin of chapter, otherwise 2. No trim_last"""
        dur = [dct.get('silence').get('dur') for dct in stamp[:3]] # For RUS
        #dur = dur[1:] # for RUS
        max_silence = (max(dur))
        #print('--Cut point:', max_silence)
        #print('-'*30)        cut_point = 0
        for dct in stamp:
            if max_silence in dct['silence'].values():
                start, end = dct.get('silence').get('start'), dct.get('silence').get('end')
                cut = (float(end) + float(start)) / 2
                #print(round(cut, 6))
                cut_point = round(cut, 6)
        print('--Cutpoint (sec):')
        print(cut_point)
        return cut_point


    def run(self):
        audio_list = list_files(self.audio_folder)
        print(len(audio_list))
        #assert len(audio_list) == 260, 'Nope.'
        #print(audio_list)
        #code = ('B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23',)
        for mp3 in audio_list:
            #for cod in code:
            #if 'B01' in mp3:
            self.audio_segment(mp3, trim_last=False)


if __name__ == "__main__":
    model = AudioSegment()
    model.run()