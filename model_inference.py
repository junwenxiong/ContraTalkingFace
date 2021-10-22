from os import listdir, makedirs, path
from audioread import audio_open
import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
from dataset import audio
import platform
from models.wav2lip import Wav2Lip_ori
from utils.checkpoint import load_test_model
from options.test_options import TestOptions
from utils.detectFaces import datagen
from utils.utils import mkdirs
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# TODO 继续改进读取文件的代码，lrs3和lrw数据集的读取不能使用
def select_txt_file(data_type):
    if data_type.lower() == 'lrs2':
        txt_path = 'evaluation/test_filelists/lrs2.txt'
        video_path = '/home/wgl/data/main'
        audio_path = '/data/home/xjw/codingFiles/Python/audio-visual_distribution_similarity/Wav2Lip-original/lrs2_preprocessed'
    if data_type.lower() == 'lrs3':
        txt_path = 'evaluation/test_filelists/lrs3.txt'
        video_path = '/home/wgl/data/lrs3_test'
        audio_path = '/home/wgl/data/lrs3_test'
    if data_type.lower() == 'lrw':
        txt_path = 'evaluation/test_filelists/lrw.txt'
        video_path = '/data/home/zy/LRW/LRW/lipread_mp4'
        audio_path = '/data/home/zy/LRW/LRW/lipread_mp4'

    return txt_path, video_path, audio_path

def get_image_list(txtPath):
    audio_files = []
    video_files = []

    with open(txtPath) as f:
        for line in f:
            line = line.strip()
            line0, line1 = line.split()[0], line.split()[1]
            audio_files.append(line0)
            video_files.append(line1)

    return audio_files, video_files

def extract_mels(audio_path, fps):
    if not audio_path.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        audio_path = 'temp/temp.wav'

    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    return mel_chunks

def extract_frames(args, video_path):

    if not os.path.isfile(video_path):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif video_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(video_path)]
        fps = args.fps

    else:
        video_stream = cv2.VideoCapture(video_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    print("Number of frames available for inference: " + str(len(full_frames)))
    return full_frames, fps

def main():
    out_root = args.outpath
    checkpoint_path = args.checkpoint_path
    txtPath, face_root, audio_root = select_txt_file(args.data_type)

    model = Wav2Lip_ori()
    model = load_test_model(device, checkpoint_path, model)
    print("Model loaded")

    audio_files, video_files = get_image_list(txtPath)

    if not os.path.exists(out_root):
        mkdirs(out_root)

    batch_size = args.wav2lip_batch_size
    count = 0
    for i, (audio_path, video_path) in enumerate(zip(audio_files, video_files)):

        face = os.path.join(face_root, "{}.mp4".format(video_path))
        out_face = str(video_path) + ".mp4"
        audio = os.path.join(audio_root, audio_path, 'audio.wav')
        outfile = os.path.join(out_root, out_face)
        filename = os.path.join(out_root, video_path.split('/')[0])

        if not os.path.exists(filename):
            mkdirs(filename)

        full_frames, fps = extract_frames(args, face)
        mel_chunks = extract_mels(audio, fps)
        full_frames = full_frames[:len(mel_chunks)]

        gen = datagen(args, full_frames.copy(), mel_chunks, device)

        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):

            if i == 0:
                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
                # out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)

        out.release()

        # command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio, 'temp/result.avi', outfile)
        command = 'ffmpeg -analyzeduration 5M -ignore_unknown -probesize 5M -y -i {} -i {} -strict -2 -q:v 1 -crf 22 {}'.format(audio, 'temp/result.avi', outfile)
        subprocess.call(command, shell=platform.system() != 'Windows')

        count +=1 

    print('test file length: {}, the number of completed: {}'.format(len(audio_files), count))

if __name__ == '__main__':
    args, _ = TestOptions().parse()

    mel_step_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} for inference.'.format(device))

    main()
