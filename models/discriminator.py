from numpy import add
import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

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

