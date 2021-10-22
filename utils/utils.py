import sys
import h5py
import os
import shutil
import cv2


def mkdirs(path, remove=False):
	if os.path.isdir(path):
		if remove:
			shutil.rmtree(path)
		else:
			return
	os.makedirs(path)

def get_image_list(data_root, split):
    filelist = []

    with open('filelists/{}.txt'.format(split)) as f:
        for line in f:
            line = line.strip()
            if ' ' in line: line = line.split()[0]
            filelist.append(os.path.join(data_root, line))

    return filelist


def create_optical_flow(frames):
    flows = [] 
    n_frames, _, _ = frames.shape
    i = 0
    while(i + 1 < n_frames):
        cur_frame = frames[i]
        next_frame = frames[i+1]

        flow = cv2.calcOpticalFlowFarneback(cur_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        flows.append(flow)
        i = i+1
    
    if len(flows) == n_frames-1:
        flows.append(cv2.calcOpticalFlowFarneback(next_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0))

    return flows