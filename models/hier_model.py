from numpy import add
import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d
from .lipreading import lipreading_model
from .attention import ChannelAttention, SpatialAttention

# for testing
# from conv import Conv2dTranspose, Conv2d, nonorm_Conv2d
# from lipreading import lipreading_model
# from attention import ChannelAttention, SpatialAttention

class HModel(nn.Module):
    def __init__(self, add_mouths=False):
        super(HModel, self).__init__()

        self.add_mouths = add_mouths

        # VGG model
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)), # 96,96

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 48,48
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # 24,24
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 12,12
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # 6,6
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),     # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.face_emb = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )

        self.audio_encoder_list = nn.ModuleList([
            nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), # (80, 16)
            ),
            nn.Sequential(
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), # (27, 16)
            ),
            nn.Sequential(
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), # (9, 6)
            ),
            nn.Sequential(
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # (3, 3)
            ),
            nn.Sequential(
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),), # (1, 1)

            # 多加了一层卷积层
            nn.Sequential(
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),), # (1, 1)
            ]
        )

        self.audio_emb = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.corr_blocks = nn.ModuleList([
         nn.Sequential(Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, residual=True)),
         nn.Sequential(Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, residual=True)),
         nn.Sequential(Conv2d(512, 256, kernel_size=3, stride=1, padding=1, residual=True)),
         nn.Sequential(Conv2d(256, 128, kernel_size=3, stride=1, padding=1, residual=True)),
         nn.Sequential(Conv2d(128, 64, kernel_size=3, stride=1, padding=1, residual=True))
        ])

        self.face_decoder_blocks = nn.ModuleList([

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # 12, 12

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),),]) # 96,96

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())
        
        self.audio_attens = nn.ModuleList([
            ChannelAttention(in_planes=512),
            ChannelAttention(in_planes=512),
            ChannelAttention(in_planes=256)
        ])

    def forward_audio_embed(self, audio_sequences):
        """

        Args:
            audio_sequences ([type]): (bs*T, 1, 80, 16)

        Returns:
            audio_feature: (bs*T, 512, 1, 1)
            audio_embedding: (bs*T, 512, 1, 1)
        """
        feats = []
        # print('audio input shape: {}'.format(audio_sequences.shape))
        x = audio_sequences
        for f in self.audio_encoder_list:
            x= f(x)
            feats.append(x)
            # print("audio feature {}".format(x.shape))

        return feats

    def forward_face_embed(self, face_sequences):
        """

        Args:
            face_sequences ([type]): (bs*T, 6, 96, 96)

        Returns:
            feats: list of face features
            face_embedding: (bs*T, 512)
        """
        feats = []
        up_list = []
        # print('visual input shape: {}'.format(face_sequences.shape))
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)
            # print("visual.shape: {}".format(x.shape))
            up_list.append(torch.nn.Upsample(x.shape[2:], mode='bilinear'))

        return feats, up_list

    def forward(self, audio_sequences, face_sequences, mouths=None):
        """
        Args:
            audio_sequences ([type]): shape (bs, T, 1, 80, 16)
            face_sequences ([type]): shape (bs, 6, T, 96, 96)

        Returns:
            [type]: shape (bs, 6, T, 96, 96)
        """
        B, T = audio_sequences.size(0), audio_sequences.size(1)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            # audio seq reshapes to (bs*T, ...)
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            # face seq reshapes to (bs*T, ...)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_feats = self.forward_audio_embed(audio_sequences)


        # TODO 尝试将mouth_embedding加入到face embedding中
        face_feats, up_list = self.forward_face_embed(face_sequences)

        assert len(up_list) != len(self.corr_blocks), "dimension mismatch"

        up_length = len(up_list)
        
        step = 3
        up_conv = zip(up_list[::-1], self.audio_attens, self.face_decoder_blocks[:up_length])
        for i, (up, atten, dec_conv) in enumerate(up_conv):

            audio_feature = audio_feats[-1]
            visual_feature = face_feats[-1]
            if not audio_feature.size() == visual_feature.size():
                audio_feature = up(audio_feature)  # 可以先对audio feature融入spatial attention
            
            audio_feature = audio_feature * atten(audio_feature) # channel attention

            # corr_feature = torch.matmul(audio_feature, visual_feature.transpose(-1, -2))

            if i == 0:
                dec_feature = dec_conv(torch.cat([audio_feature, visual_feature], dim=1))
            else:
                dec_feature = dec_conv(torch.cat([audio_feature, dec_feature], dim=1))

            # print("{} dec_feautre shape: {}".format(i + 1, dec_feature.shape))
            audio_feats.pop()
            face_feats.pop()
            if i+1 >= step: break

        
        dec_feature = torch.cat((dec_feature, face_feats[-1]), dim=1)
        face_feats.pop()
        for f in self.face_decoder_blocks[step:]:
            dec_feature = f(dec_feature)
            try:
                dec_feature = torch.cat((dec_feature, face_feats[-1]), dim=1)
            except Exception as e:
                print(dec_feature.size())
                print(face_feats[-1].size())
                raise e
            face_feats.pop()

        dec_feature = self.output_block(dec_feature)  # (bs*T, 80, 96, 96) --> (bs*T, 3, 96, 96)

        if input_dim_size > 4:
            dec_feature = torch.split(dec_feature, B, dim=0)  # [(bs, C, H, W) * T]
            outputs = torch.stack(dec_feature, dim=2)  # (bs, C, T, H, W)

        else:
            outputs = dec_feature

        return outputs

class Wav2Lip_disc_qual(nn.Module):
    def __init__(self):
        super(Wav2Lip_disc_qual, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)), # 48,96

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2), # 48,48
            nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),    # 24,24
            nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),   # 12,12
            nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),       # 6,6
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),     # 3,3
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.last_conv = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0),)
        self.sigmoid = nn.Sigmoid()

        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2) // 2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat(
            [face_sequences[:, :, i] for i in range(face_sequences.size(2))],
            dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        """

        Args:
            false_face_sequences ([type]): shape (bs, 3, T, 96, 96)

        Returns:
            [type]: shape ()
        """
        false_face_sequences = self.to_2d(
            false_face_sequences)  # shape (bs*T, 3, 96, 96)
        false_face_sequences = self.get_lower_half(
            false_face_sequences)  # shape (bs*T, 3, 96, 96)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        false_feats = self.last_conv(false_feats)
        false_feats = self.sigmoid(false_feats)

        false_pred_loss = F.binary_cross_entropy(false_feats.view(len(false_feats), -1), torch.ones((len(false_feats), 1)).cuda())

        return false_pred_loss

    def forward(self, face_sequences, use_sigmoid=True):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        x = self.last_conv(x)

        if use_sigmoid:
            x = self.sigmoid(x)

        return x.view(len(x), -1)


if __name__ == '__main__':
    model = Wav2Lip()

    visual = torch.randn(4, 6, 5, 96, 96)
    audio = torch.randn(4, 5, 1, 80, 16)

    output = model(audio, visual)