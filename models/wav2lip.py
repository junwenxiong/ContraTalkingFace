from numpy import add
import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d
from .lipreading import lipreading_model

# for testing
# from conv import Conv2dTranspose, Conv2d, nonorm_Conv2d
# from lipreading import lipreading_model

class Wav2Lip_ori(nn.Module):
    def __init__(self, add_mouths=False):
        super(Wav2Lip_ori, self).__init__()

        self.add_mouths = add_mouths

        if add_mouths:
            config_path = 'models/lipreading/lrw_snv1x_tcn2x.json'
            weight_path = 'models/lipreading/lipreading_best.pth'
            self.lipreading = lipreading_model.build_lipreadingnet(config_path, weight_path, extract_feats=True)
            in_c = 1024
        else:
            in_c = 512

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


        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)


        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),

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

    def forward(self, audio_sequences, face_sequences, mouths=None):
        # audio_sequences = (B, T, 1, 80, 16)
        B, T = audio_sequences.size(0), audio_sequences.size(1)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        if self.add_mouths and mouths is not None:
            mouth_embedding, _ = self.lipreading(mouths, T)
            mouth_embedding = mouth_embedding.reshape(B*T, mouth_embedding.size(1), 1, 1)
        

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        if self.add_mouths and mouths is not None:
            x = torch.cat([audio_embedding, mouth_embedding], dim=1)
        else:
            x = audio_embedding

        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            
            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x
            
        return outputs

class Wav2Lip(nn.Module):
    def __init__(self, add_mouths=False):
        super(Wav2Lip, self).__init__()

        self.add_mouths = add_mouths

        if add_mouths:
            config_path = 'models/lipreading/lrw_snv1x_tcn2x.json'
            weight_path = 'models/lipreading/lipreading_best.pth'
            self.lipreading = lipreading_model.build_lipreadingnet(config_path, weight_path, extract_feats=True)
            in_c = 1024
        else:
            in_c = 512

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

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)
        
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

        self.face_decoder_blocks = nn.ModuleList([
            # nn.Sequential(Conv2d(in_c, 512, kernel_size=1, stride=1, padding=0),),

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

    def forward_audio_embed(self, audio_sequences):
        """

        Args:
            audio_sequences ([type]): (bs*T, 1, 80, 16)

        Returns:
            audio_feature: (bs*T, 512, 1, 1)
            audio_embedding: (bs*T, 512, 1, 1)
        """
        feats = []
        print('audio input shape: {}'.format(audio_sequences.shape))
        x = audio_sequences
        for f in self.audio_encoder_list:
            x= f(x)
            feats.append(x)
            print("audio feature {}".format(x.shape))

        audio_feature = self.audio_encoder(audio_sequences)  # (bs*T, 1, 80, 16) --> (bs*T, 512, 1, 1)
        audio_embedding = audio_feature.view(-1, audio_feature.size(1))
        audio_embedding = self.audio_emb(audio_embedding) # (bs*T, 512)
        return audio_feature, audio_embedding
    
    def forward_face_embed(self, face_sequences):
        """

        Args:
            face_sequences ([type]): (bs*T, 6, 96, 96)

        Returns:
            feats: list of face features
            face_embedding: (bs*T, 512)
        """
        feats = []
        print('visual input shape: {}'.format(face_sequences.shape))
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)
            print("visual.shape: {}".format(x.shape))
        x = x.view(-1, feats[-1].size(1))
        face_embedding = self.face_emb(x) # (bs*T, 512)
        return feats[:-1], face_embedding  # 去除最后一层卷积特征，使用face_embedding

    # TODO 保留原始的模型文件
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

        audio_feature, audio_embedding = self.forward_audio_embed(audio_sequences) 

        # 添加mouths embedding
        if self.add_mouths and mouths is not None:
            mouth_embedding, _ = self.lipreading(mouths, T)
            mouth_embedding = mouth_embedding.reshape(B*T, mouth_embedding.size(1), 1, 1)

        # TODO 尝试将mouth_embedding加入到face embedding中
        feats, face_embedding = self.forward_face_embed(face_sequences)

        if self.add_mouths and mouths is not None:
            x = torch.cat([audio_feature, mouth_embedding], dim=1)
        else:
            x = audio_feature
        
        # TODO 对齐audio_embedding和face_embedding
        x = torch.cat([audio_embedding, face_embedding], dim=1).view(-1, 2*audio_embedding.size(1), 1, 1).contiguous()

        for f in self.face_decoder_blocks:
            print('generated shape: {}'.format(x.shape))
            x = f(x)

            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        x = self.output_block(x)  # (bs*T, 80, 96, 96) --> (bs*T, 3, 96, 96)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)  # [(bs, C, H, W) * T]
            outputs = torch.stack(x, dim=2)  # (bs, C, T, H, W)

        else:
            outputs = x

        return outputs, audio_embedding, face_embedding


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