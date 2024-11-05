

import sys
waffles_dir = '/afs/cern.ch/user/a/anbalbon/waffles'
sys.path.append(waffles_dir+'/src') 


import waffles
import pickle
import numpy as np

import waffles.plotting.drawing_tools as draw

import os


directory_path = "/eos/home-f/fegalizz/public/to_Anna"
if os.path.exists(directory_path):
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
else:
    file_paths = "Directory non trovata."

Energy = '+1'
Endpoint = 109
Channels = [37]


print('inizio')

for file in file_paths : 
    with open(file, 'rb') as f:
        wfset = pickle.load(f)
    
    print(f"File: {file.split('/')[-1]} -> N waveforms: {len(wfset.Waveforms)}")

    '''
    min_list = []
    max_list = []
    for wf in st_wfset.Waveforms:
        min_list.append(min(wf.Adcs[50:120]))
        max_list.append(max(wf.Adcs[50:120]))

    min_list = [int(i) for i in min_list]

    i=0
    for ampl in min_list:
        if ampl < 500:
            i+=1
    if i > 0:
        print(f'{i} saturated waveforms')
    '''

    
print('fine')
