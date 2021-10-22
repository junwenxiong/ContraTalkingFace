import torch
import torch.nn as nn
import torchvision


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(
            pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat(
            [face_sequences[:, :, i] for i in range(face_sequences.size(2))],
            dim=0)
        return face_sequences

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2) // 2:]

    def forward(self, x):
        x = self.to_2d(x)
        x = self.get_lower_half(x)

        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
