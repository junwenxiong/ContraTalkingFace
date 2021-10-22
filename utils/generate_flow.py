import cv2

def create_optical_flow(frames):
    flows = [] 
    n_frames = frames.shape[0]
    i = 0
    while(i + 1 < n_frames):
        cur_frame = frames[i]
        next_frame = frames[i+1]

        flow = cv2.calcOpticalFlowFarneback(cur_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        flows.append(flow)
        i = i+1

    return flows