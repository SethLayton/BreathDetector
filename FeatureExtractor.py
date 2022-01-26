from asyncore import write
from calendar import c
import glob, os
import math
import numpy
import scipy.io.wavfile as wav
import numpy as np
from pydub import AudioSegment, effects
from python_speech_features import mfcc, delta
import pickle

##converts manually annotated timestamps to ms
def get_msec(time_str):
    time_str = time_str.replace('.', ":")
    m, s, ms = time_str.split(':')
    return int(m) * 60000 + int(s) * 1000 + int(ms)

##isConcat determines if we concatenate all features together
##win_length and win_hop default to 25ms and 10ms respectively
def extract_features(win_len = 0.025, win_hop = 0.01):
    ##define locations of the annotations and the actual audio files
    breath_loc_files = "E:\\Documents\\School\\Grad Research\\DeepFake\\Breath Detection\\Switchboard_Manual_Annotations"
    audio_files = "E:\\Documents\\School\\Grad Research\\DeepFake\\DataSets\\LDC97S62\\data"
    all_feats = []

    ##do this for all the mannually annotated audio files
    for filename in glob.glob(os.path.join(breath_loc_files, '*.txt')):
        locations = []
        ##pull the name of the audio wave file from the manually annotated file
        wav_file = filename.split('\\')[len(filename.split('\\'))-1].split('_')[0] + "_" + filename.split('\\')[len(filename.split('\\'))-1].split('_')[1] + ".wav"

        ##if the Feature set file exists, load it and skip reprocessing
        features_file = ".\\saved_objects\\" + wav_file + ".features"
        if os.path.exists(features_file):            
            ##Load the feature set from file
            DCRlistmatrix_normalized = pickle.load(open(features_file, 'rb'))
        else:
            ##if the file hasn't already been normalized do this
            output_file = wav_file.split(".wav")[0] + "_Normalized.wav"
            if not os.path.exists(output_file):
                rawsound = AudioSegment.from_file(audio_files + "\\" + wav_file, "wav")  
                normalizedsound = effects.normalize(rawsound)  
                normalizedsound.export(audio_files + "\\" + output_file, format="wav")

            ##Read in the wave file to get its signal and rate
            rate,sig = wav.read(os.path.join(audio_files, output_file))

            ##Process the MFCCs on this audio signal
            temp_mfcc_feat = mfcc(signal=sig,samplerate=rate,winlen=win_len,winstep=win_hop,preemph=.95, numcep=13, nfft=1103, winfunc=np.hanning, appendEnergy=False) #calculate the MFCC on this signal
            
            ##Calculate the Derivatives and Derivatives Derivatives
            delta_feat = delta(temp_mfcc_feat, 2)
            delta_delta_feats = delta(delta_feat, 2)
            
            ##Perform DC removal on the MFCC features
            temp_mfcc_feat = np.hstack((temp_mfcc_feat,delta_feat,delta_delta_feats))
            DCRmfcc_feat = np.empty((0,temp_mfcc_feat.shape[1]), float)
            for tempmfcc in temp_mfcc_feat:
                mean = np.mean(tempmfcc)
                DCRmfcc = np.subtract(tempmfcc,mean)
                DCRmfcc_feat = np.vstack((DCRmfcc_feat, DCRmfcc))
            DCRmfcc_matrix = DCRmfcc_feat
            DCRmfcc_matrix = np.array(DCRmfcc_matrix)


            ##Read in the locations of all the breath events and save it for later retrieval
            with open(filename) as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]
                lines = [i for i in lines if i]
            for location in lines[2:]:   
                start = get_msec(location.split(' ')[0]) * rate * 0.001
                end = get_msec(location.split(' ')[1]) * rate * 0.001
                locations.append((start,end))

            DCRlistmatrix_normalized = []
            step_count = 0
            curr_loc = 0

            ##Define the window length and hop size. (This corresponds with the MFCC window and hop)
            winlength = math.ceil(win_len * rate)
            step = math.ceil(win_hop * rate)

            ##Process the audio file to calculate the STE and ZCR for each window and append these values to the feature set
            ##also builds the label for the window
            while curr_loc < len(sig):                
                if curr_loc + winlength > len(sig):
                    tempsig = sig[curr_loc:]
                    padlen = int(winlength - len(tempsig))
                    zeros = numpy.zeros((padlen,))
                    tempsig = numpy.concatenate((tempsig,zeros))
                else:
                    tempsig = sig[curr_loc: curr_loc + winlength]  
                zero_crossing = np.nonzero(np.diff(tempsig > 0))[0].size / len(tempsig)
                np_sig = np.array(tempsig, dtype='float')
                hamming = np.hamming(len(tempsig))
                tsq = (np_sig * hamming) ** 2 #apply a hamming window to the signal
                tsum = np.sum(tsq)
                short_time_energy = math.log(tsum)
                temp_mf = DCRmfcc_matrix[step_count]      
                
                seg_start = curr_loc
                seg_end = curr_loc + winlength

                ##Derive the label for this window based on point intersections
                if (seg_end > len(sig)):
                    seg_end = len(sig)
                match = False
                for l in locations:
                    if (l[1] > seg_start and seg_start > l[0]) or (seg_end > l[0] and l[0] > seg_start) or (l[0] < seg_start and seg_end < l[1]):                    
                        match = True
                        break
                if match: ##if breath event mark 1
                    x = np.append(temp_mf, (zero_crossing, short_time_energy, 1))
                else: ##if non breath event mark 0
                    x = np.append(temp_mf, (zero_crossing, short_time_energy, 0))   

                curr_loc = curr_loc +  step
                step_count = step_count + 1
                DCRlistmatrix_normalized.append(x)
                if curr_loc + winlength > len(sig):
                    break

            ##Dump the feature set out to a file to save time on this step in the future          
            pickle.dump(DCRlistmatrix_normalized, open(features_file, 'wb'))

        ##Create the overall feature set list
        all_feats.append(DCRlistmatrix_normalized)

    np_all_feats = np.vstack(all_feats)
    ##Return the X_features and the Y_labels
    return np_all_feats[:, :-1], np_all_feats[:, -1]


def audio_concat(breath_loc_files, audio_files, final_breath_audio_file, final_breath_loc_file):

    final_signal = []
    final_locations = []
    current_offset = 0.0

    ##do this for all the mannually annotated audio files
    for filename in glob.glob(os.path.join(breath_loc_files, '*.txt')):
        
        ##pull the name of the audio wave file from the manually annotated file
        wav_file = filename.split('\\')[len(filename.split('\\'))-1].split('_breaths')[0] + ".wav"
    
        ##if the file hasn't already been normalized do this
        output_file = wav_file.split(".wav")[0] + "_Normalized.wav"
        if not os.path.exists(os.path.join(audio_files, output_file)):
            rawsound = AudioSegment.from_file(audio_files + "\\" + wav_file, "wav")  
            normalizedsound = effects.normalize(rawsound)  
            normalizedsound.export(audio_files + "\\" + output_file, format="wav")

        ##Read in the wave file to get its signal and rate
        rate,sig = wav.read(os.path.join(audio_files, output_file))

        with open(filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            lines = [i for i in lines if i]
        for location in lines[2:]:
            start_ms = get_msec(location.split(' ')[0])
            end_ms = get_msec(location.split(' ')[1])
            final_locations.append(((start_ms+current_offset),(end_ms+current_offset)))
        current_offset = current_offset + (len(sig)/rate * 1000)
        final_signal.extend(sig)
    
    wav.write(final_breath_audio_file,rate,np.array(final_signal))
    with open(final_breath_loc_file, "w") as output_file:
        for loc in final_locations:
            output_file.write(str(loc[0]) + " " + str(loc[1]) + "\n")

def train_test_split_RNN(sig_x, sig_y, test_size = 0.25):

    test_length = math.floor((len(sig_x) * test_size) / 2) 
    lsig = math.floor(len(sig_x) / 2)
    train_sig_b = sig_x[0:(lsig - test_length)]
    train_sig_e = sig_x[(lsig + test_length): len(sig_x)]
    train_sig_x =  np.append(train_sig_b, train_sig_e)
    test_sig_x = sig_x[(lsig - test_length):(lsig + test_length)]

    train_sig_b_y = sig_y[0:(lsig - test_length)]
    train_sig_e_y = sig_y[(lsig + test_length): len(sig_y)]
    train_sig_y =  np.append(train_sig_b_y, train_sig_e_y)
    test_sig_y = sig_y[(lsig - test_length):(lsig + test_length)]
    
    return train_sig_x, test_sig_x, train_sig_y, test_sig_y

def extract_features_RNN(input_audio, input_breaths, win_len = 0.025, win_hop = 0.01, test_size = 0.25):
    ##Read in the wave file to get its signal and rate
    rate,sig = wav.read(input_audio)    
    all_feats = []
    locations = []

    features_file = ".\\saved_objects\\" + input_audio + ".features"
    if os.path.exists(features_file):            
        ##Load the feature set from file
        DCRlistmatrix_normalized = pickle.load(open(features_file, 'rb'))
    else:
        ##Process the MFCCs on this audio signal
        temp_mfcc_feat = mfcc(signal=sig,samplerate=rate,winlen=win_len,winstep=win_hop,preemph=.95, numcep=13, nfft=1103, winfunc=np.hanning, appendEnergy=False) #calculate the MFCC on this signal
        
        ##Calculate the Derivatives and Derivatives Derivatives
        delta_feat = delta(temp_mfcc_feat, 2)
        delta_delta_feats = delta(delta_feat, 2)
        
        ##Perform DC removal on the MFCC features
        temp_mfcc_feat = np.hstack((temp_mfcc_feat,delta_feat,delta_delta_feats))
        DCRmfcc_feat = np.empty((0,temp_mfcc_feat.shape[1]), float)
        for tempmfcc in temp_mfcc_feat:
            mean = np.mean(tempmfcc)
            DCRmfcc = np.subtract(tempmfcc,mean)
            DCRmfcc_feat = np.vstack((DCRmfcc_feat, DCRmfcc))
        DCRmfcc_matrix = DCRmfcc_feat
        DCRmfcc_matrix = np.array(DCRmfcc_matrix)


        ##Read in the locations of all the breath events and save it for later retrieval
        with open(input_breaths) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            lines = [i for i in lines if i]
        for location in lines[2:]:   
            start = get_msec(location.split(' ')[0]) * rate * 0.001
            end = get_msec(location.split(' ')[1]) * rate * 0.001
            locations.append((start,end))

        DCRlistmatrix_normalized = []
        step_count = 0
        curr_loc = 0

        ##Define the window length and hop size. (This corresponds with the MFCC window and hop)
        winlength = math.ceil(win_len * rate)
        step = math.ceil(win_hop * rate)

        ##Process the audio file to calculate the STE and ZCR for each window and append these values to the feature set
        ##also builds the label for the window
        while curr_loc < len(sig):                
            if curr_loc + winlength > len(sig):
                tempsig = sig[curr_loc:]
                padlen = int(winlength - len(tempsig))
                zeros = numpy.zeros((padlen,))
                tempsig = numpy.concatenate((tempsig,zeros))
            else:
                tempsig = sig[curr_loc: curr_loc + winlength]  
            zero_crossing = np.nonzero(np.diff(tempsig > 0))[0].size / len(tempsig)
            np_sig = np.array(tempsig, dtype='float')
            hamming = np.hamming(len(tempsig))
            tsq = (np_sig * hamming) ** 2 #apply a hamming window to the signal
            tsum = np.sum(tsq)
            short_time_energy = math.log(tsum)
            temp_mf = DCRmfcc_matrix[step_count]      
            
            seg_start = curr_loc
            seg_end = curr_loc + winlength

            ##Derive the label for this window based on point intersections
            if (seg_end > len(sig)):
                seg_end = len(sig)
            match = False
            for l in locations:
                if (l[1] > seg_start and seg_start > l[0]) or (seg_end > l[0] and l[0] > seg_start) or (l[0] < seg_start and seg_end < l[1]):                    
                    match = True
                    break
            if match: ##if breath event mark 1
                x = np.append(temp_mf, (zero_crossing, short_time_energy, 1))
            else: ##if non breath event mark 0
                x = np.append(temp_mf, (zero_crossing, short_time_energy, 0))   

            curr_loc = curr_loc +  step
            step_count = step_count + 1
            DCRlistmatrix_normalized.append(x)
            if curr_loc + winlength > len(sig):
                break

        ##Dump the feature set out to a file to save time on this step in the future          
        pickle.dump(DCRlistmatrix_normalized, open(features_file, 'wb'))

    ##Create the overall feature set list
    all_feats.append(DCRlistmatrix_normalized)

    np_all_feats = np.vstack(all_feats)
    ##Return the X_features and the Y_labels
    return np_all_feats[:, :-1], np_all_feats[:, -1]