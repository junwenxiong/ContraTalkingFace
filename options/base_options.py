import argparse
import os
import torch
import datetime
from utils import utils
from utils.logger import Logger, print_dict
from hparams import hparams


class BaseOption():
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description=
            'Code to train the Wav2Lip model WITH the visual quality discriminator'
        )
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            "--data_root",
            default='lrs2_preprocessed/',
            help="Root folder of the preprocessed LRS2 dataset",
            type=str)
        self.parser.add_argument(
            '--name',
            type=str,
            default='Wav2Lip_original',
            help='name of the experiment. It decides where to store models')
        self.parser.add_argument('--checkpoint_root',
                                 default='/data_8T/xjw/DeepFake/checkpoints',
                                 help='Save checkpoints to this directory',
                                 type=str)

        self.parser.add_argument(
            '--syncnet_checkpoint_path',
            default='checkpoints/lipsync_expert.pth',
            help='Load the pre-trained Expert discriminator',
            type=str)

        self.parser.add_argument('--checkpoint_path',
                                 help='Resume generator from this checkpoint',
                                 default=None,
                                 type=str)
        self.parser.add_argument(
            '--disc_checkpoint_path',
            help='Resume quality disc from this checkpoint',
            default=None,
            type=str)
        self.parser.add_argument(
            '--num_mels',
            default=80,
            help=
            'Number of mel-spectrogram channels and local conditioning dimensionality'
        )
        self.parser.add_argument(
            '--rescale',
            default=True,
            help='Whether to rescale audio prior to preprocessing')
        self.parser.add_argument('--rescaling_max',
                                 default=0.9,
                                 help='Rescaling value')

        self.parser.add_argument('--use_lws', default=False)
        self.parser.add_argument(
            '--n_fft',
            default=800,
            help=
            'Extra window size is filled with 0 paddings to match this paramete'
        )
        self.parser.add_argument(
            '--hop_size',
            default=200,
            help='For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate')
        self.parser.add_argument(
            '--win_size',
            default=800,
            help=
            'For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate'
        )
        self.parser.add_argument(
            '--sample_rate',
            default=16000,
            help='16000Hz (corresponding to librispeech) (sox --i <filename>')
        self.parser.add_argument(
            '--frame_shift_ms',
            default=None,
            help='Can replace hop_size parameter. (Recommended: 12.5')

        # Mel and Linear spectrograms normalization/scaling and clipping
        self.parser.add_argument('--signal_normalization', default=True)
        # Whether to normalize mel spectrograms to some predefined range (following below parameters)
        self.parser.add_argument(
            '--allow_clipping_in_normalization',
            default=True,
            help='Only relevant if mel_normalization = True')
        self.parser.add_argument('--symmetric_mels', default=True)
        # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2,
        # faster and cleaner convergence)
        self.parser.add_argument('--max_abs_value', default=4.)
        # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not
        # be too big to avoid gradient explosion,
        # not too small for fast convergence)
        # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude
        # levels. Also allows for better G&L phase reconstruction)
        self.parser.add_argument('--preemphasize',
                                 default=True,
                                 help='whether to apply filte')
        self.parser.add_argument('--preemphasis',
                                 default=0.97,
                                 help='filter coefficient')

        # Limits
        self.parser.add_argument('--min_level_db', default=-100)
        self.parser.add_argument('--ref_level_db', default=20)
        self.parser.add_argument('--fmin', default=55)
        # Set this to 55 if your speaker is male! if female, 95 should help 'taking off noise. (To
        # test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
        self.parser.add_argument(
            '--fmax',
            default=7600,
            help='To be increased/reduced depending on data')

        self.parser.add_argument('--img_size', default=96)
        self.parser.add_argument('--fps', default=25)
        self.parser.add_argument('--batch_size', default=16)
        self.parser.add_argument('--num_workers', default=8, help='')
        self.initialized = True

    def parse(self):
        if not self.initialized: 
            self.initialize()

        self.opt = self.parser.parse_args()

        self.opt.checkpoint_dir = os.path.join(self.opt.checkpoint_root,
                                               self.opt.name)
        utils.mkdirs(self.opt.checkpoint_dir)

        today = datetime.date.today()
        self.opt.configPath = os.path.join(
            self.opt.checkpoint_dir,
            'result_{}_{}.log'.format(str(today.month), str(today.day)))
        configLogger = Logger(log_fn=self.opt.configPath)
        configLogger.add_line("-" * 20 + "Model Config" + "-" * 20)
        print_dict(configLogger, vars(self.opt))

        return self.opt, configLogger
