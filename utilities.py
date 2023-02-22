import numpy as np 
import pyaudio
import wave
import matplotlib.pyplot as plt
import math
from scipy.fftpack import dct,idct
import matplotlib.pyplot as plt 
import librosa
import librosa.display
import copy
import os
import re
import random
from num2words import num2words
'''
compute mfcc feature
'''
## sth for spectrogram:https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.specgram.html

rate = 16000
num_points_seg = 512

## load wav
def load_wav(filename):
    wf = wave.open(filename, 'rb') 
    params = wf.getparams()
    channels, sampwidth,rate, nframes = params[:4]
    str_data = wf.readframes(nframes)
    wf.close()
    wave_data = np.frombuffer(str_data,dtype=np.int16) 
    return wave_data

## preemphasize
def preemphasize(filename):
    wave_data = load_wav(filename)
    alpha = 0.95
    empha_wave = np.append(wave_data[0],wave_data[1:]-alpha * wave_data[:-1])    
    return empha_wave

## windowing & zeropadding
def windowing(filename):
    '''
    Parameters: empha_wave
    Return: n * 512 matrix (n is number of segments)
    to obtain segments of 20 ms with 10 ms shift, 
    and we have 1024 samples for 64 ms (16000Hz),
    therefore for 10 ms there should be 160 samples
    '''
    empha_wave = preemphasize(filename)
    # segmentation & windowing & zero-padding
    rate = 16000
    segment_len = 10
    segment_points = 16000 / 1000 * 10
    num_segments = int(len(empha_wave)/segment_points)
    segments_chopped =  np.array_split(empha_wave,num_segments)
    segments = np.ones((1,512))
    for i in range(len(segments_chopped)-1):
        segment = np.concatenate((segments_chopped[i],segments_chopped[i+1]))
        window = np.hanning(len(segment))
        windowed_seg = segment * window
        padded_seg = np.pad(windowed_seg,(0,512-len(windowed_seg)),'constant')
        segments = np.vstack((segments,padded_seg))
    ret = segments[1:,:]
    return ret


## FFT to get power spectrum
## reference: https://fairyonice.github.io/implement-the-spectrogram-from-scratch-in-python.html
def get_power_spectrum(filename):
    window_padded_segs = windowing(filename)
    fft_segs = np.fft.rfft(window_padded_segs)
    # fft_segs = np.fft.rfft(window_padded_segs)
    magnitude_spect = np.absolute(fft_segs)
    power_spect = np.power(magnitude_spect,2)
    #print(power_spect.shape)
    return fft_segs,power_spect ,magnitude_spect 
     
def hertz2mel(freq):
    return 2595 * np.log10(1+freq/700) 
def mel2hertz(mel):
    return 700 * (10**(mel/2595)) - 700

def get_filterbank(num_filter):
    '''
    return filter bank in 2D array 
    with one dimension matching the number of samples in one segment, 
    and another dimension being number of futers (40)
    formula of filterbank: HAN P.315
    '''
    rate = 16000
    freq_min = 133.33
    freq_max = 6855.4976
    mel_min = hertz2mel(freq_min)
    mel_max = hertz2mel(freq_max)
    mel_points = np.linspace(mel_min,mel_max,num_filter+2)
    freqs = mel2hertz(mel_points)
    freqs = np.floor(freqs / rate * 512)
    filterbank = np.zeros((num_filter,257))
    # apply formula
    for m in range(1,num_filter+1):
        f_m_left = int(freqs[m-1])
        f_m_center = int(freqs[m])
        f_m_right = int(freqs[m+1])
        for k in range(f_m_left,f_m_center):
            filterbank[m-1,k] = (k-freqs[m-1])/(freqs[m]-freqs[m-1])  
        for k in range(f_m_center,f_m_right):
            filterbank[m-1,k] = (freqs[m+1]-k)/(freqs[m+1]-freqs[m])
    return filterbank      

def get_spe_cep_trum(filename,num_filter):
    fft_segments, power_spectrum, magnitude_spectrum = get_power_spectrum(filename)
    filters = get_filterbank(num_filter)
    mel_spectrum = np.dot(power_spectrum,filters.T)
    log_mel_spectrum = np.log10(mel_spectrum)  
    
    ## take DCT, take first 13 dimensions for Mel cepstrum
    mel_ceptrum = dct(log_mel_spectrum,n=13)
    
    ## take inverse DCT for validation
    back_ceptrum = idct(mel_ceptrum,n=num_filter)

    return mel_spectrum,log_mel_spectrum,mel_ceptrum,back_ceptrum

def cepstra_normalization(mel_ceptrum):
    #mel_spectrum,log_mel_spectrum,mel_ceptrum,back_ceptrum = get_spe_cep_trum(filename, num_filter)
    ## mean normalization
    (nframes, ncoeff) = mel_ceptrum.shape
    total_cep = np.sum(mel_ceptrum,axis = 0)
    mean_cep = total_cep / nframes
    cepstra_mean_normal = mel_ceptrum - mean_cep
    
    ## variance normalization
    std = np.sqrt(np.sum(np.square(cepstra_mean_normal),axis=0)/nframes)
    cepstra_normal = 1/std*cepstra_mean_normal
    return cepstra_normal

def velocity(cepstra_normal):
    n_frames = len(cepstra_normal)
    denominator = 2 * sum([i**2 for i in range(1, 2)])
    vel_feat = np.empty_like(cepstra_normal)
    padded = np.pad(cepstra_normal, ((1, 1), (0, 0)), mode='edge')   
    for t in range(n_frames):
        vel_feat[t] = np.dot(np.arange(-1, 2), padded[t : t+3]) / denominator
    return vel_feat

def acceleration(cepstra_normal):
    n_frames = len(cepstra_normal)
    denominator = 2 * sum([i**2 for i in range(1, 3)])
    acce_feat = np.empty_like(cepstra_normal)
    padded = np.pad(cepstra_normal, ((2, 2), (0, 0)), mode='edge')   
    for t in range(n_frames):
        acce_feat[t] = np.dot(np.arange(-2, 3), padded[t : t+5]) / denominator  
    return acce_feat

def mfcc_features(filename,num_filter):
    y, sr = librosa.load(filename, sr=16000)
    melcep = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13,n_fft=512,n_mels=num_filter).T
    cepstra_normal = cepstra_normalization(melcep)
    vel_feat = velocity(cepstra_normal)
    acce_feat = acceleration(cepstra_normal)
    mfcc = np.hstack((cepstra_normal,vel_feat,acce_feat))
    return mfcc

 
#features = mfcc_features("zero3.wav",40)
#print(features.shape)

'''
train single gaussian
'''

#from mfcc import *

num_filter = 40
num_states=5
num_templ = 5
digits = ['zero','one','two','three','four','five','six','seven','eight','nine']

### define state
class State():
    def __init__(self,matrix,parent=None,next=None,name=None,id=None):
        '''
        Parameters
        ----------
        mean and cov: mean and covariance of a state as single gaussian
        parent: parent state (previous + self-transition)
        next: next state (next + self-transition)
        name: which hmm it belongs to (use negative to distinguish)
        id: index in hmm.states
        cost: cost of current state
        '''  
        self.matrix = matrix
        self.parent = parent
        self.next = next
        self.name = name
        self.id = id
         
    
    ### calculate mean and covariance of a state_matrix
    @property
    def mean_cov(self):
        '''
        Return: mean and cov of given matrix
        '''
        mean = np.mean(self.matrix,axis=0)
        cov = np.cov(self.matrix.T)
        return mean,cov

## define HMM model as a combination of states
class HMM():
    def __init__(self,states,start_state=0):
        '''
        Parameters
        ----------
        states: a list of state objects 
        transition_matrix: transition matrix calculated by states
        start_state: default is 0
        '''
        #super().__init__()
        self.states = states
        self.start_state = start_state
        self.transition_matrix = self.calculation_transition_matrix(states)
     
    def calculation_transition_matrix(self,states):
        '''
        Return transition matrix from states
        '''
        transition_matrix = np.zeros((num_states,num_states))
        for i in range(num_states-1):
            nframes = self.states[i].matrix.shape[0]
            transition_matrix[i][i+1] = num_templ / nframes
            transition_matrix[i][i] = (nframes - num_templ)/nframes
        transition_matrix[num_states-1][num_states-1] = 1 
        return transition_matrix
    
    #def update_transition_matrix(self):
        

## calculate node cost as negative log likelihood
def get_node_cost(x,state):
    '''
    Return node cose
    
    Parameters
    ----------
    x: frame vector from mfcc features
    '''
    mean,cov = state.mean_cov
    cov_diag = np.diagonal(cov)
     
    node_cost = 0.5 * np.sum(np.log(2*np.pi*cov_diag)) + 0.5 * np.sum(np.square(x-mean)/cov_diag)
    return node_cost

## calculate edge cost as negative log transition prob
def get_edge_cost(transition_matrix):
    zero_index = np.where(transition_matrix == 0)
    transition_matrix[zero_index] = 1/(np.iinfo(np.int32).max)
    edge_cost = -np.log(transition_matrix)
    return edge_cost


## initialization 
def initialization(templ_files):
    '''
    Return a hmm_initial class
    
    Parameters
    ----------
    templ_files: list of template wave file names
    '''
        
    segs = []
    for file in templ_files:
        feature = mfcc_features(file,num_filter)
        uniform_seg = np.array_split(feature,num_states)
        segs.append(uniform_seg)
    
    states = []
    for i in range(num_states):
        state_matrix = segs[0][i]
        for j in range(1,len(segs)):
            sta = segs[j][i]
            state_matrix = np.vstack((state_matrix,sta))
        state = State(state_matrix)
        states.append(state)
    hmm_initial = HMM(states)
    return hmm_initial

## dtw alignment
def dtw_alignment(test,hmm):
    '''
    Return state_ids (list-type) of new alignment
    
    Parameters
    ----------
    test: mfcc feature of test file
    hmm: template states as hmm
    '''
    ## dtw
    test_nframes = test.shape[0]
    back_pointer_table = np.zeros((num_states,test_nframes))
    
    for j in range(test_nframes):
        if j == 0:
            cost = np.zeros(num_states) + np.inf
            cost[0] = get_node_cost(test[0],hmm.states[0])
            back_pointer_table[0][0] = 0
            continue
        
        prev_cost = copy.deepcopy(cost)
        
        for i in range(num_states):
             
            edge_cost = get_edge_cost(hmm.transition_matrix)[:,i]
 
            parent_state = np.argmin(edge_cost+prev_cost)
            
            back_pointer_table[i][j] = parent_state   
                 
            min_cost = (edge_cost+prev_cost)[parent_state]
            local_node_cost = get_node_cost(test[j],hmm.states[i])
           
            cost[i] = min_cost + local_node_cost

    ## back-pointer
    
    state_id = num_states-1 #np.argmin(cost) 
    state_ids = [state_id]
     
    for j in list(range(test_nframes-1,0,-1)):
        state_id = int(back_pointer_table[state_id][j])
        state_ids.append(state_id)
    state_ids = state_ids[::-1]
     
    return state_ids,cost

## update hmm from states_ids of new alignment
def update_hmm(templ_files,states_ids):
    '''
    Return a updated hmm object
    
    Parameters
    ----------
    templ_files: list of template file names
    states_ids: list of state_ids
    '''
    states = []
    for i in range(num_states):
        state_matrix = np.zeros((1,39))
        for k in range(len(templ_files)):
            templ = mfcc_features(templ_files[k],num_filter)
            state_ids = np.array(states_ids[k]) #list->array
            state_matrix = np.vstack((state_matrix,templ[np.where(state_ids==i)]))
        state = State(state_matrix[1:])
        states.append(state)
    hmm = HMM(states)
    
    return hmm 


def get_hmm_mean(hmm):
    means = 0
    for i in range(num_states):
        mean,cov = hmm.states[i].mean_cov
        means += mean
    return np.mean(mean)
 
## training
def training_single_gaussian(templ_files):
    
    ## initialization
    hmm = initialization(templ_files)
    mean = get_hmm_mean(hmm)

    converge = True
    means = [mean]
    
    ## converge
    cnt = 0
    while converge:
  
        states_ids = []
        for k in range(len(templ_files)):
            test = mfcc_features(templ_files[k],num_filter)
            
            ## alignment
            state_ids,_ = dtw_alignment(test,hmm)
            states_ids.append(state_ids)
            
            
        ## update 5 templates together!!!
        hmm = update_hmm(templ_files,states_ids)

        ## check for converge
        mean = get_hmm_mean(hmm)
        means.append(mean)
        
        cnt += 1
        if np.abs((means[cnt] - means[cnt-1])/means[cnt-1]) < 0.01:
            converge = False
        
    return hmm

## get hmm models of 0-9
def get_hmm_digits(filenames):
    '''
    Return a list of 10 hmm models associated with 0-9 digits
    
    Parameters
    ----------
    digits: list of 'one' to 'nine'
    filenames: all filenames (10 files for 0-9 digits each)
    '''
    ## training hmm model for each digit
    hmm_digits = [] # list to store 10 hmm models associated with 10 digits
    
    # order: 0 -> 9
    for digit in digits:
        digit_files = []
        for filename in filenames:
            if digit in filename:
                digit_files.append(filename)
        
        random.shuffle(digit_files)
        
        ## choose 5 files of each digit as templates
        templ_files = digit_files[:5]
        hmm = training_single_gaussian(templ_files)
        hmm_digits.append(hmm)
    return hmm_digits

## test
def testing(test_files,hmm_digits):
    '''
    Return a list of digit that test_files are recognized by hmm
    
    Parameters
    ----------
    test_files: randomly chosen
    hmm_digis: list of hmm models of 0-9
    '''
    
    recognized_digits = []
    for k in range(len(test_files)):
        test = mfcc_features(test_files[k],num_filter)
        costs = [] # costs associated with hmm alignment, size is 10
        for hmm in hmm_digits:
            _,cost = dtw_alignment(test,hmm)
            costs.append(cost[-1])
        recognized_digit = num2words(np.argmin(costs))
        recognized_digits.append(recognized_digit)
    return recognized_digits

'''
4 gaussians
'''

