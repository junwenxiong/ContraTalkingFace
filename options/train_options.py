from options.base_options import BaseOption


class TrainOptions(BaseOption):
    def initialize(self):
        BaseOption.initialize(self)
        self.parser.add_argument('--initial_learning_rate', default=1e-4, type=float)
        self.parser.add_argument('--nepochs', default=1000000, type=int, help='ctrl + c, stop whenever eval loss is consistently greater than train loss for ~10 epoch')
        self.parser.add_argument('--checkpoint_interval', default=10000, type=int, help='time interval for saving model weights')
        self.parser.add_argument('--save_image_interval', default=5000, type=int, help='time interval for saving model weights')
        self.parser.add_argument('--eval_interval', default=5000, type=int, help='evaluation interval')
        self.parser.add_argument('--save_optimizer_state', default=True,type=bool)
        self.parser.add_argument( '--syncnet_T', default=5, type=int, help='length of visual input frames')
        self.parser.add_argument( '--syncnet_mel_step_size', default=16, type=int, help='length of audio input frames')
        self.parser.add_argument( '--syncnet_wt', default=0.03, type=float, help='initially zero, will be set automatically to 0.03 later. Leads to faster convergence.')
        self.parser.add_argument('--syncnet_batch_size', default=64, type=int)
        self.parser.add_argument('--syncnet_lr', default=1e-4, type=float)
        self.parser.add_argument('--syncnet_eval_interval', default=10000, type=int)
        self.parser.add_argument('--syncnet_checkpoint_interval', default=10000, type=int)
        # self.parser.add_argument('--disc_wt', defalut=0.07, type=float, help='discriminatior loss weight')
        self.parser.add_argument('--disc_wt', type=float, default=0.07 )
        self.parser.add_argument('--disc_initial_learning_rate', default=1e-4, type=float)
        self.parser.add_argument('--vgg_wt', type=float, default=0.10 )
        self.mode = 'train'
