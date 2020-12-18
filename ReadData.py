#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:24:27 2020

@author: seafood
"""
import os
import numpy as np
import math 

def list_all_files(rootdir):
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
               _files.append(path)
#               if path.find('Data_')!=-1:
#                   _files.append(path)
#           if os.path.isfile(path):
#              if path.find('[S]')!=-1:
#                  _files.append(path)
    return _files

def get_data_infile(path,framelength,inc):
    files=list_all_files(path)
    datas_aco=[]
    datas_semi=[]
    num=np.zeros(len(files))
    labels=np.zeros(len(files))
    #print(files)
    for i in range(len(files)):
        if files[i].endswith('.TXT') or files[i].endswith('.txt'):
            correct=0
        
            if files[i].lower().find('largewheel')!=-1:
                label=0
            else:
                if files[i].lower().find('smallwheel')!=-1:
                    label=1
                if files[i].lower().find('track')!=-1:
                    label=2
            #if files[i].lower().find('track')!=-1:
                #label=3
            #if files[i].lower().find('helicopter')!=-1:
                #label=4
            #if files[i].lower().find('noise')!=-1:
                #label=5
            data=np.loadtxt(files[i])
            #datas=datas[:,0]
            if files[i].startswith('/media/seafood/3CE4B50EE4B4CC00/Database/Acoustic-and-seismic-synchronous-signal/20200805ASSSFromLYan/sequenceDataSet/[A]'):
                data=data[:,0]#信号多通道时取其中一个通道
                data=data[::8]
                datas_aco=data
                
            elif files[i].startswith('/media/seafood/3CE4B50EE4B4CC00/Database/Acoustic-and-seismic-synchronous-signal/20200805ASSSFromLYan/sequenceDataSet/[S]'):
                data=data[::8]
                datas_semi=data
            num=math.floor((len(datas_aco)-framelength+inc)/inc)
    return label, datas_aco, datas_semi, num
