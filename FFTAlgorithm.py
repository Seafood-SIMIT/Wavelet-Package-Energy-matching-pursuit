#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:46:09 2020

@author: seafood
"""

import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import ReadData
import math
import os
import PyEMD
from scipy import signal
import pandas as pd
import pywt

#set the sample point number
sample_N=200;
sample_point_number = np.linspace(1,1,sample_N)
frame_length = 1024
inc = frame_length
wavelet_depth=6
wavename = 'bior3.5'
#read data
path=r'/media/seafood/3CE4B50EE4B4CC00/Database/Acoustic-and-seismic-synchronous-signal/20200805ASSSFromLYan/sequenceDataSet2/target'
savedir='../output/'
files=ReadData.list_all_files(rootdir=path)

def averageOfFrameEMD(frame_semi):
    IMF_frame_semi=PyEMD.EMD().emd(frame_semi,t)
    N_frame_semi=IMF_frame_semi.shape[0]
    return IMF_frame_semi


def signalFftTransform(signal):
    atom_fft=fft(signal)
    atom_fft_amplitude = (abs(atom_fft)/sample_N)[range(int(sample_N/2))]
    atom_fft_phase = np.angle(atom_fft)
    return atom_fft_amplitude

def wpd_plt(signal):
    #wpd分解
    wp = pywt.WaveletPacket(data=signal, wavelet=wavename,mode='symmetric',maxlevel=wavelet_depth)
    #energy stract
    re = []
    for i in [node.path for node in wp.get_level(wavelet_depth, 'freq')]:
        re.append(wp[i].data)
    #能量特征
    energy=[]
    for i in re:
        energy.append(pow(np.linalg.norm(i,ord=None), 2))
    return energy

def preprocessFromFile(item_file):
    atom_origin = np.loadtxt(item_file)
    #atom_origin = atom_origin[::8]
    atom_origin = atom_origin[:,0]
    atom_origin = atom_origin - np.mean(atom_origin)
    
    #fft transform
    #atom_fft_amplitude_aver = np.zeros(int(sample_N/2))
    #for i in range(math.floor((len(atom_origin)-frame_length+inc)/inc)):
    #for i in range(20):
     #   frame_semi_atom = atom_origin[i*inc:i*inc+frame_length]
     #   atom_fft_amplitude_aver=atom_fft_amplitude_aver+signalFftTransform(frame_semi_atom)
    #print(np.min(atom_fft_amplitude_aver),np.max(atom_fft_amplitude_aver))
    #atom_fft_amplitude_normorlize=(atom_fft_amplitude_aver-np.min(atom_fft_amplitude_aver))/(np.max(atom_fft_amplitude_aver)-np.min(atom_fft_amplitude_aver))
    #return atom_fft_amplitude_normorlize
    #此处使用小波包能量系数代替fft
    atom_wavelet_energy = np.zeros(2**wavelet_depth)
    for i in range(20):
        frame_aco_atom = atom_origin[i*inc:i*inc+frame_length]
        atom_wavelet_energy=atom_wavelet_energy+wpd_plt(frame_aco_atom)
    #归一化
    atom_wavelet_energy_normorlize=(atom_wavelet_energy-np.min(atom_wavelet_energy))/(np.max(atom_wavelet_energy)-np.min(atom_wavelet_energy))
    return atom_wavelet_energy
def dictionaryGenerate():
    dictionary_path = '/media/seafood/3CE4B50EE4B4CC00/Database/Acoustic-and-seismic-synchronous-signal/20200805ASSSFromLYan/sequenceDataSet2/dictionary'
    files_dict=ReadData.list_all_files(rootdir=dictionary_path)
    atom=np.zeros((2**wavelet_depth,3))
    for item_file in files_dict:
        if item_file.startswith(dictionary_path+'/[S]01smallwheel'):
            atom[:,0] = preprocessFromFile(item_file)
        elif item_file.startswith(dictionary_path+'/[S]01largewheel'):
            atom[:,1] = preprocessFromFile(item_file)
        elif item_file.startswith(dictionary_path+'/[S]01track'):
            atom[:,2] = preprocessFromFile(item_file)

    return atom

def csOmp(target, atom,K):
    residual=target
    (M,N)=atom.shape
    index=np.zeros(N,dtype=int)-1
    result = np.zeros(N)
    
    for j in range(K):
        product = np.fabs(np.dot(atom.T, residual))
        pos=np.argmax(product)
        index[pos]=1
        my = np.linalg.pinv(atom[:,index>=0])
        a=np.dot(my,target) #最小二乘,看参考文献1     
        residual=target-np.dot(atom[:,index>=0],a)
    result[index>=0]=a
    Candidate = np.where(index>=0) #返回所有选中的列
    return  result, Candidate

if 'smallwheel' in files[0]:
    label=0
elif 'largewheel' in files[0]:
    label=1
elif 'track' in files[0]:
    label=2

target_origin = np.loadtxt(files[0])
target_origin = target_origin[:,0]
#target_origin = target_origin[::8]
target_origin = target_origin - np.mean(target_origin)
num_length_target = math.floor((len(target_origin)-frame_length+inc)/inc)
t=np.linspace(0,1,inc)


atom=dictionaryGenerate()
sparse_matrix_target=np.zeros((num_length_target,3))
pinlvrange = range(int(sample_N/2))
count = 0
for i in range(num_length_target):
    frame_semi_target = target_origin[i*inc:i*inc+frame_length]
    
    target_fft_amplitude = signalFftTransform(frame_semi_target)
    #print(np.min(target_fft_amplitude),np.max(target_fft_amplitude))
    target_fft_amplitude=(target_fft_amplitude-np.min(target_fft_amplitude))/(np.max(target_fft_amplitude)-np.min(target_fft_amplitude))
    x_pre_target, candidate = csOmp(target_fft_amplitude, atom, 3)  
    sparse_matrix_target[i,0:len(x_pre_target)]=x_pre_target[0:len(x_pre_target)]
    pred_label = np.argmax(x_pre_target)
    if label == pred_label:
        count+=1
    plt.figure()
    plt.plot(pinlvrange,atom[:,0],'r','-.')
    plt.plot(pinlvrange,atom[:,1],'g','-.')
    plt.plot(pinlvrange,atom[:,2],'b','-.')
    plt.plot(pinlvrange,target_fft_amplitude,'k','-')
    plt.show()
    foldername='/home/seafood/workdir/Mynetwork/thirdMultitarget/output/FFTOMP'
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    picturename=foldername+'/fftfigure'+str(i)+'.png'
    plt.savefig(picturename,dpi=720)
    plt.close()
    print(x_pre_target)
    
    print('num. ',i,'frame, recognizition:',pred_label)
    
print('total',num_length_target,'frame, accurate ',count/num_length_target)
    
    