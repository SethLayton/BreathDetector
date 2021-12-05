import glob, os
import scipy.io.wavfile as wav

def get_msec(time_str):
    time_str = time_str.replace('.', ":")
    m, s, ms = time_str.split(':')
    return int(m) * 60000 + int(s) * 1000 + int(ms)


breath_loc_files = "E:\\Documents\\School\\Grad Research\\DeepFake\\Breath Detection\\Switchboard_Manual_Annotations"
audio_files = "E:\\Documents\\School\\Grad Research\\DeepFake\\DataSets\\LDC97S62\\data"
breath_segment_files_no_noise = "E:\\Documents\\School\\Grad Research\\DeepFake\\Breath Detection\\Segments\\WithoutNoise"
breath_segment_files = "E:\\Documents\\School\\Grad Research\\DeepFake\\Breath Detection\\Segments\\WithNoise"
total = 0
numsegments = 0
for filename in glob.glob(os.path.join(breath_loc_files, '*.txt')):
    wav_file = filename.split('\\')[len(filename.split('\\'))-1].split('_')[0] + "_" + filename.split('\\')[len(filename.split('\\'))-1].split('_')[1] + ".wav"
    segment_counter = 0
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        lines = [i for i in lines if i]
    for location in lines[2:]:
        numsegments = numsegments + 1        
        start = get_msec(location.split(' ')[0])
        end = get_msec(location.split(' ')[1])
        # if ((end-start) > 1000 or (end-start) < 0):
        #     print ("Duration: ", (end-start), " File: ", wav_file, " Line:", segment_counter)
        wav_output_file = wav_file.split(".")[0] + "_" + str(segment_counter) + ".wav"
        segment_counter = segment_counter + 1
        rate,sig = wav.read(os.path.join(audio_files, wav_file))
        msrate = rate / 1000
        sclip = int(round(float(start) * msrate))
        eclip = int(round(float(end) * msrate))
        clip_sig = sig[sclip:eclip]    
        total = total + (end-start)
        if (len(location.split(' ')) < 3): #no noise in this segment
            wav.write(os.path.join(breath_segment_files_no_noise,wav_output_file), rate, clip_sig)  
        
        wav.write(os.path.join(breath_segment_files, wav_output_file), rate, clip_sig)  
print ("Total breath segments: " , numsegments)
print ("Total breath duration: " , total)
print ("Average breath duration: " , total/numsegments)