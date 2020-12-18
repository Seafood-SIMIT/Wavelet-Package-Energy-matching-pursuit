#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:46:09 2020

@author: seafood
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import ReadData
import math
import os
#import PyEMD
from scipy import signal
import pywt

#set the sample point number
sample_N=200;
sample_point_number = np.linspace(1,1,sample_N)
frame_length = 10240
inc = frame_length
wavelet_depth=6
wavename = 'db1'
#read data
path=r'/Users/sunlin/workdir/Mynetwork/forthvoicefilter/dataset/testset'
savedir='../output/'
files=ReadData.list_all_files(rootdir=path)


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
    #atom_origin = atom_origin[:,0]
    #atom_origin = atom_origin - np.mean(atom_origin)
    
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
    #for i in range(20):
    for i in range(math.floor((len(atom_origin)-frame_length+inc)/inc)):
        frame_aco_atom = atom_origin[i*inc:i*inc+frame_length]
        atom_wavelet_energy=atom_wavelet_energy+wpd_plt(frame_aco_atom)
    #归一化
    atom_wavelet_energy = atom_wavelet_energy/math.floor((len(atom_origin)-frame_length+inc)/inc)
    atom_wavelet_energy = atom_wavelet_energy/np.sqrt(np.dot(atom_wavelet_energy,atom_wavelet_energy.T))
    #atom_wavelet_energy_regularization=(atom_wavelet_energy-np.min(atom_wavelet_energy))/(np.max(atom_wavelet_energy)-np.min(atom_wavelet_energy))
    #print(np.sqrt(np.dot(atom_wavelet_energy,atom_wavelet_energy.T)))
    #正则化
    
    return atom_wavelet_energy
def dictionaryGenerate():
    dictionary_path = '/Users/sunlin/workdir/Mynetwork/forthvoicefilter/dataset/dictionary'
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

def csOmp1(target, atom,K):
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

def csOmp2(target, atom,K):
    residual=target
    (M,N)=atom.shape
    index=np.zeros(N,dtype=int)
    result = np.zeros(N)
    
    for j in range(K):
        product = np.fabs(np.dot(atom.T, residual))
        pos=np.argmax(product)
        index[pos]+=1
        my = np.linalg.pinv(atom[:,index>0])
        a=np.dot(my,target) #最小二乘,看参考文献1     
        residual=target-np.dot(atom[:,index>0],a)
    result[index>0]=a
    #Candidate = np.where(index>0) #返回所有选中的列
    return index.reshape(1,3)

def csOmp(target, atom,K):
    residual=target
    (M,N)=atom.shape
    index=np.zeros(N,dtype=int)
    result = np.zeros(N)
    
    for j in range(K):
        product = np.fabs(np.dot(atom.T, residual))
        pos=np.argmax(product)
        index[pos]+=1
        my = np.linalg.pinv(atom[:,index>0])
        a=np.dot(my,target) #最小二乘,看参考文献1     
        residual=target-np.dot(atom[:,index>0],a)
    result[index>0]=a
    #Candidate = np.where(index>0) #返回所有选中的列
    return index.reshape(1,3)

if 'smallwheel' in files[1]:
    label=0
elif 'largewheel' in files[1]:
    label=1
elif 'track' in files[1]:
    label=2

target_origin = np.loadtxt(files[1])
#target_origin = target_origin[:,0]
#target_origin = target_origin[::8]
#target_origin = target_origin - np.mean(target_origin)
num_length_target = math.floor((len(target_origin)-frame_length+inc)/inc)
t=np.linspace(0,1,inc)


atom=dictionaryGenerate()
sparse_matrix_target=np.zeros((num_length_target,3))
pinlvrange = range(int(sample_N/2))
count = 0
for i in range(num_length_target):
    frame_semi_target = target_origin[i*inc:i*inc+frame_length]
    
    target_fft_amplitude = np.array(wpd_plt(frame_semi_target))
    #print(np.min(target_fft_amplitude),np.max(target_fft_amplitude))
    #target_fft_amplitude=(target_fft_amplitude-np.min(target_fft_amplitude))/(np.max(target_fft_amplitude)-np.min(target_fft_amplitude))
    #target_fft_amplitude = target_fft_amplitude/np.sqrt(np.dot(target_fft_amplitude,target_fft_amplitude.T))
    #x_pre_target, candidate = csOmp1(target_fft_amplitude, atom, 3)  
    x_pre_target = csOmp2(target_fft_amplitude, atom, 1)
    #sparse_matrix_target[i,0:len(x_pre_target)]=x_pre_target[0:len(x_pre_target)]
    pred_label = np.argmax(x_pre_target)
    #print("x_pre_target:",x_pre_target,"    candidate:",candidate)
    print("x_pre_target:",x_pre_target)
    if label == pred_label:
        count+=1
    #plt.figure()
    #plt.plot(pinlvrange,atom[:,0],'r','-.')
    #plt.plot(pinlvrange,atom[:,1],'g','-.')
    #plt.plot(pinlvrange,atom[:,2],'b','-.')
    #plt.plot(pinlvrange,target_fft_amplitude,'k','-')
    #plt.show()
    #foldername='/home/seafood/workdir/Mynetwork/thirdMultitarget/output/waveletOMP'
    #if not os.path.exists(foldername):
        #os.makedirs(foldername)
    #picturename=foldername+'/fftfigure'+str(i)+'.png'
    #plt.savefig(picturename,dpi=720)
    #plt.close()
    #print(x_pre_target)
    
    print('num. ',i,'frame, recognizition:',pred_label)
    
print('total',num_length_target,'frame, accurate ',count/num_length_target)
    
    