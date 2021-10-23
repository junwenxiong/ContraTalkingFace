from dataset import audio
import numpy as np
import torch
import os, random, cv2, argparse
from glob import glob
from os.path import dirname, join, basename, isfile
from .lipreading_preprocess import *
from utils.utils import get_image_list


class LRS2Dataset(object):
    def __init__(self, args, split):
        self.args = args
        self.all_videos = get_image_list(self.args.data_root, split)
    
    def name(self):
        return "lrs2_dataset"

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        """
        获取图片名字

        self.args:
            start_frame ([type]): [description]

        Returns:
            [type]: [description]
        """
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + self.args.syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        """
        加载图片

        self.args:
            window_fnames ([type]): [description]

        Returns:
            [type]: [description]
        """
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (self.args.img_size, self.args.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        """
        截取一段mel spetrogram

        self.args:
            spec ([type]): [description]
            start_frame ([type]): [description]

        Returns:
            [type]: [description]
        """
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(self.args.fps)))

        end_idx = start_idx + self.args.syncnet_mel_step_size

        return spec[start_idx:end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        """
        截取多段mels

        Args:
            spec ([type]): [description]
            start_frame ([type]): [description]

        Returns:
            [type]: [description]
        """
        mels = []
        assert self.args.syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1  # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + self.args.syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != self.args.syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * self.args.syncnet_T:
                continue

            # 随机选择一帧
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            # 读取输入图片名称
            window_fnames = self.get_window(img_name)
            # 读取预测图片名称
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, self.args.sample_rate)

                orig_mel = audio.melspectrogram(self.args, wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            # 反向mel
            inverse_mel = self.crop_audio_window(orig_mel.copy(), wrong_img_name)

            if (mel.shape[0] != self.args.syncnet_mel_step_size):
                continue

            if (inverse_mel.shape[0] != self.args.syncnet_mel_step_size):
                continue

            # 对应input frames的音频片段
            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            # 分割反向mel spectrogram
            inverse_indiv_mels = self.get_segmented_mels(orig_mel.copy(), wrong_img_name)
            if inverse_indiv_mels is None: continue

            # 处理视频帧
            window = self.prepare_window(window)
            # reference frames
            wrong_window = self.prepare_window(wrong_window)

            y = window.copy()
            window[:, :, window.shape[2] // 2:] = 0.
            x = np.concatenate([window, wrong_window], axis=0)

            inverse_y = wrong_window.copy()
            wrong_window[:, :, wrong_window.shape[2] // 2:] = 0.
            inverse_x = np.concatenate([wrong_window, window], axis=0)

            # 正向数据
            x = torch.FloatTensor(x)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            y = torch.FloatTensor(y)

            # 反向数据
            inverse_x = torch.FloatTensor(inverse_x)
            inverse_indiv_mels = torch.FloatTensor(inverse_indiv_mels).unsqueeze(1)
            inverse_mel = torch.FloatTensor(inverse_mel.T).unsqueeze(0)
            inverse_y = torch.FloatTensor(inverse_y)

            return x, indiv_mels, mel, y, inverse_x, inverse_indiv_mels, inverse_mel, inverse_y
