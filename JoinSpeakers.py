import scipy.io.wavfile as wav
import os
import numpy as np


audio_filename = "..\\Democratic Debate\\Democratic_Debate_part1.wav"
filename = "..\\Democratic Debate\\Part1Debate.txt"
speaker_list = []
speech_list = []
with open(filename) as file:
    temp_speech = []
    for temp_line in file:
        temp_line = temp_line.replace('\n','')
        if (temp_line in ('','\n','\t')): continue        
        if (temp_line.startswith('\t')):
            temp_line = temp_line.replace('\t', '')
            split = temp_line.split(' - ')
            start = split[0]
            end = split[1]
            if (len(start.split(':')) != 3):
                start = "0:" + start
            if (len(end.split(':')) != 3):
                end = "0:" + end
            temp_speech.append((start,end))       
        else:
            speaker_list.append(temp_line)
            if (temp_speech.__len__() > 0):
                speech_list.append(temp_speech)
                temp_speech = []
    if (temp_speech.__len__() > 0):
        speech_list.append(temp_speech)
        temp_speech = []

rate,sig = wav.read(audio_filename)
#one_second_blank = list(np.zeros(rate, dtype=int))
for x, i in enumerate(range(len(speaker_list))):
    output_audio_filename = "..\\Democratic Debate\\speaker_" + speaker_list[i] + "_part1.wav"
    speaker_sig = []
    for times in speech_list[i]:
        split_start = times[0].split(":")
        split_end = times[1].split(":")
        frame_start = (rate * (int(split_start[0]) * 3600 + int(split_start[1]) * 60 + int(split_start[2]))) - rate
        frame_end = (rate * (int(split_end[0]) * 3600 + int(split_end[1]) * 60 + int(split_end[2]))) + rate
        speaker_sig.extend(sig[frame_start:frame_end])
        #speaker_sig.extend(one_second_blank)
    speaker_sig = np.array(speaker_sig)
    wav.write(output_audio_filename, rate, speaker_sig)