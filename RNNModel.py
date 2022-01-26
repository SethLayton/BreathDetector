import os
from matplotlib.pyplot import cla
from python_speech_features import mfcc
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier as knn
import sklearn
from sklearn.metrics import f1_score
import pickle
from FeatureExtractor import extract_features_RNN, audio_concat, train_test_split_RNN
from GridSearch import grid_search
from Plot import pca
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

audio_files = os.path.abspath(os.path.join(__file__ ,"../../..")) + "\\DataSets\\LDC97S62\\data"
breath_loc_files = "..\\Switchboard_Manual_Annotations"
final_breath_audio_file = "..\\RNNBreath\\audio_concatenated.wav"
final_breath_loc_file = "..\\RNNBreath\\locations_concatenated.txt"

if not os.path.exists(final_breath_audio_file):
    audio_concat(breath_loc_files, audio_files, final_breath_audio_file, final_breath_loc_file)

window_length = 0.020
window_hop = 0.0025
##Extract features from the audio files based on the manual annotations
all_feats_x, all_feats_y = extract_features_RNN(final_breath_audio_file, final_breath_loc_file, window_length, window_hop)

##split into training and test
##Test is 25% of the data taken from the middle of the audio stream
train_sig_x, test_sig_x, train_sig_y, test_sig_y = train_test_split_RNN (all_feats_x, all_feats_y, test_size=0.25)

##Do something with the data


