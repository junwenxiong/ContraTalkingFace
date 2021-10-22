import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

from os import listdir, path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=2, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=96, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", default='/data_8T/xjw/DeepFake/generated_result/10_19_vgg_with_weight_0.1_26W_random_audio/')
args = parser.parse_args()


def process_video_file(vfile):
	video_stream = cv2.VideoCapture(vfile)
	
	frames = []
	while 1:
		still_reading, frame = video_stream.read()
		if not still_reading:
			video_stream.release()
			break
		frames.append(frame)
	
	if not len(frames):
		print(vfile)


def mp_handler(job):
	vfile, args, gpu_id = job
	try:
		process_video_file(vfile)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()
		
def main(args):
	print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))

	# filelist = glob(path.join(args.data_root, '*/*.mp4'))
	filelist = glob(path.join(args.data_root, '*.mp4'))

	jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(filelist)]
	p = ThreadPoolExecutor(args.ngpu)
	futures = [p.submit(mp_handler, j) for j in jobs]
	_ = [r.result() for r in as_completed(futures)]
	print(len(filelist))


if __name__ == '__main__':
	main(args)