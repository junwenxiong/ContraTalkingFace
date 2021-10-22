from options.base_options import BaseOption


class TestOptions(BaseOption):
    def initialize(self):
        BaseOption.initialize(self)
        self.parser.add_argument(
            '--data_type',
            type=str,
            default='LRS2',
            help='dataset type'
        )
        self.parser.add_argument(
            '--outpath',
            type=str,
            help='Video path to save result. See default for an e.g.',
            required=True)

        self.parser.add_argument(
            '--static',
            type=bool,
            help='If True, then use only first video frame for inference',
            default=False)

        self.parser.add_argument(
            '--pads',
            nargs='+',
            type=int,
            default=[0, 10, 0, 0],
            help=
            'Padding (top, bottom, left, right). Please adjust to include chin at least'
        )

        self.parser.add_argument('--face_det_batch_size',
                                 type=int,
                                 help='Batch size for face detection',
                                 default=64)
        self.parser.add_argument('--wav2lip_batch_size',
                                 type=int,
                                 help='Batch size for Wav2Lip model(s)',
                                 default=128)

        self.parser.add_argument(
            '--resize_factor',
            default=1,
            type=int,
            help=
            'Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p'
        )

        self.parser.add_argument(
            '--crop',
            nargs='+',
            type=int,
            default=[0, -1, 0, -1],
            help=
            'Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
            'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width'
        )

        self.parser.add_argument(
            '--box',
            nargs='+',
            type=int,
            default=[-1, -1, -1, -1],
            help=
            'Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
            'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).'
        )

        self.parser.add_argument(
            '--rotate',
            default=False,
            action='store_true',
            help=
            'Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
            'Use if you get a flipped result, despite feeding a normal looking video'
        )

        self.parser.add_argument(
            '--nosmooth',
            default=False,
            action='store_true',
            help=
            'Prevent smoothing face detections over a short temporal window')
        self.mode = 'test'
