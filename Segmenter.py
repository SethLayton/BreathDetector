import glob, os
import scipy.io.wavfile as wav

##converts manually annotated timestamps to ms
def get_msec(time_str):
    time_str = time_str.replace('.', ":")
    m, s, ms = time_str.split(':')
    return int(m) * 60000 + int(s) * 1000 + int(ms)

##location of file that contains breath segment annotation locations
breath_loc_files = "E:\\Documents\\School\\Grad Research\\DeepFake\\Breath Detection\\Switchboard_Manual_Annotations"
##location of the actual switchboard audio files
audio_files = "E:\\Documents\\School\\Grad Research\\DeepFake\\DataSets\\LDC97S62\\data"
##output folder location for non-noise breath segments
breath_segment_files_no_noise = "E:\\Documents\\School\\Grad Research\\DeepFake\\Breath Detection\\Segments\\WithoutNoise"
##output folder location for all breath segments (includes ones marked noisy)
breath_segment_files = "E:\\Documents\\School\\Grad Research\\DeepFake\\Breath Detection\\Segments\\WithNoise"

total = 0
numsegments = 0

##loop through all the breath segment annotation files
for filename in glob.glob(os.path.join(breath_loc_files, '*.txt')):
    ##parse out the file to get the corresponding switchboard audio file
    wav_file = filename.split('\\')[len(filename.split('\\'))-1].split('_')[0] + "_" + filename.split('\\')[len(filename.split('\\'))-1].split('_')[1] + ".wav"
    segment_counter = 0

    ##open the annotated file and get a list of all the breath locations
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        lines = [i for i in lines if i]

    ##read in the actual switchboard audio file
    rate,sig = wav.read(os.path.join(audio_files, wav_file))

    ##loop through each breath location   
    for location in lines[2:]:
        numsegments = numsegments + 1
        ##get the start and end point of the breath        
        start = get_msec(location.split(' ')[0])
        end = get_msec(location.split(' ')[1])
        msrate = rate / 1000
        sclip = int(round(float(start) * msrate))
        eclip = int(round(float(end) * msrate))

        ##slice the audio file to only grab the breath segment
        clip_sig = sig[sclip:eclip]

        ##parse for the output file name
        wav_output_file = wav_file.split(".")[0] + "_" + str(segment_counter) + ".wav"
        segment_counter = segment_counter + 1

        ##sum up the total breath duration
        total = total + (end-start)

        ##decide if the segment has noise or not.
        ##ALL segments are written to the noise inclusive output, and only non-noise written to non-noise output
        if (len(location.split(' ')) < 3): #no noise in this segment
            wav.write(os.path.join(breath_segment_files_no_noise,wav_output_file), rate, clip_sig)         
        wav.write(os.path.join(breath_segment_files, wav_output_file), rate, clip_sig)  

##statistics        
print ("Total breath segments: " , numsegments)
print ("Total breath duration: " , total)
print ("Average breath duration: " , total/numsegments)