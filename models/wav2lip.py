from numpy import add
import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d
from .lipreading import lipreading_model
from .loss import InfoNCE, C2Loss

# for testing
# from conv import Conv2dTranspose, Conv2d, nonorm_Conv2d
# from lipreading import lipreading_model
# from loss import InfoNCE, C2Loss

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
        
        self.audio_emb = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

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

    def forward_audio_embed(self, audio_sequences):
        """

        Args:
            audio_sequences ([type]): (bs*T, 1, 80, 16)

        Returns:
            audio_feature: (bs*T, 512, 1, 1)
            audio_embedding: (bs*T, 512, 1, 1)
        """

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
        # print('visual input shape: {}'.format(face_sequences.shape))
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)
            # print("visual.shape: {}".format(x.shape))

        x = x.view(-1, feats[-1].size(1))
        face_embedding = self.face_emb(x) # (bs*T, 512)
        return feats[:-1], face_embedding  # 去除最后一层卷积特征，使用face_embedding

    def forward(self, audio_sequences, face_sequences, mouths=None):
        """
        Args:
            audio_sequences ([type]): shape (bs, T, 1, 80, 16)
            face_sequences ([type]): shape (bs, 6, T, 96, 96)

        Returns:
            [type]: shape (bs, 6, T, 96, 96)
        """
        self.B, self.T = audio_sequences.size(0), audio_sequences.size(1)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            # audio seq reshapes to (bs*T, ...)
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            # face seq reshapes to (bs*T, ...)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_feature, audio_embedding = self.forward_audio_embed(audio_sequences)  # (bs*T, 512)

        # 添加mouths embedding
        if self.add_mouths and mouths is not None:
            mouth_embedding, _ = self.lipreading(mouths, self.T)
            mouth_embedding = mouth_embedding.reshape(self.B*self.T, mouth_embedding.size(1), 1, 1)

        # TODO 尝试将mouth_embedding加入到face embedding中
        feats, face_embedding = self.forward_face_embed(face_sequences)

        if self.add_mouths and mouths is not None:
            x = torch.cat([audio_feature, mouth_embedding], dim=1)
        else:
            x = audio_feature
        
        # TODO 对齐audio_embedding和face_embedding
        x = torch.cat([audio_embedding, face_embedding], dim=1).view(-1, 2*audio_embedding.size(1), 1, 1).contiguous()

        for f in self.face_decoder_blocks:
            # print('generated shape: {}'.format(x.shape))
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
            x = torch.split(x, self.B, dim=0)  # [(bs, C, H, W) * T]
            outputs = torch.stack(x, dim=2)  # (bs, C, T, H, W)

        else:
            outputs = x
        
        # loss = self.compute_sync_loss(audio_embedding, face_embedding)

        return outputs, audio_embedding, face_embedding
    
    def compute_sync_loss(self, x, y):
        x = x.reshape(self.B, -1)
        y = y.reshape(self.B, -1)

        sync_loss = InfoNCE(temperature=1)
        loss = sync_loss(x, y.detach())

        # c2loss = C2Loss()
        # loss2 = c2loss(x, y)

        return loss

if __name__ == '__main__':
    model = Wav2Lip()

    visual = torch.randn(4, 6, 5, 96, 96)
    audio = torch.randn(4, 5, 1, 80, 16)

    output, loss = model(audio, visual)
    print(output[0].shape, loss[0], loss[1])