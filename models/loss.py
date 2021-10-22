import torch
import torch.nn as nn
import torch.nn.functional as F
from models import VGG19
import itertools

# class ArcFaceLoss(nn.Module):
#     def __init__(self, arcface=iresnet18()):
#         super().__init__()
#         self.arcface = arcface.cuda()
#         weight = torch.load('checkpoints/arcface/arcface_r18.pth')
#         self.arcface.load_state_dict(weight)

#         self.criterion = nn.L1Loss()
    
#     def forward(self, x, y):
#         x_arc, y_arc = self.arcface(x), self.arcface(y)
#         loss = self.criterion(x_arc, y_arc.detach())
#         return loss


class VGGLoss(nn.Module):
    def __init__(self, vgg=VGG19()):
        super(VGGLoss, self).__init__()
        self.vgg = vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y, layer=0):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            if i >= layer:
                loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class Sync_loss(nn.Module):
    def __init__(self, syncnet, syncnet_T): 
        super().__init__()
        self.syncnet = syncnet
        self.syncnet_T = syncnet_T
        self.criterion = nn.BCELoss()

    def cosine_loss(self, a, v, y):
        d = nn.functional.cosine_similarity(a, v)
        loss = self.criterion(d.unsqueeze(1), y)
        return loss

    def forward(self, mel, g):
        g = g[:, :, :, g.size(3) // 2:]
        g = torch.cat([g[:, :, i] for i in range(self.syncnet_T)], dim=1)
        # B, 3 * T, H//2, W
        a, v = self.syncnet(mel, g)
        y = torch.ones(g.size(0), 1, device=a.device).float()
        return self.cosine_loss(a, v, y)


class Contrastive_loss(nn.Module):
    def __init__(self, syncnet, syncnet_T):
        super().__init__()
        self.syncnet = syncnet
        self.syncnet_T = syncnet_T

    def forward(self, a_1, v_1, a_2, v_2):   
        """

        Args:
            a_1 ([type]): shape (bs, 1, 80, 16)
            v_2 ([type]): shape (bs, 3, T, 96, 96)
        """
        
        b = a_1.size(0)
        audio = torch.cat([a_1, a_2], dim=0) # (2*bs, 1, 80, 16)
        visual = torch.cat([v_1, v_2], dim=0)
        visual = visual[:,:,:,visual.size(3)//2:]
        visual = torch.cat([visual[:,:,i] for i in range(self.syncnet_T)], dim=1)

        a, v = self.syncnet(audio, visual)

        audio_list = []
        visual_list = []
        for i in range(b):
            a_1 = a[i].unsqueeze(0)
            a_2 = a[i+b].unsqueeze(0)
            v_1 = v[i].unsqueeze(0)
            v_2 = v[i+b].unsqueeze(0)

            audio_list.append(torch.cat([a_1, a_2], dim=0))
            visual_list.append(torch.cat([v_1, v_2], dim=0))

        similarity_list = []
        audio_visual = zip(audio_list, visual_list)
        
        all_sim_list = []
        for audio, visual in audio_visual:
            sim = 0
            # 笛卡尔积
            for i in itertools.product(audio, visual):
                sim += torch.exp(torch.dot(i[0], i[1]))

            # 按顺序的排列组合
            for i in itertools.combinations(audio, 2):
                sim += torch.exp(torch.dot(i[0], i[1]))
            
            for i in itertools.combinations(visual, 2):
                sim += torch.exp(torch.dot(i[0], i[1]))

            all_sim_list.append(sim)

        # 求分母
        all_sim_list_2 = []
        for audio in audio_list:
            all_sim = 0
            for i in itertools.product(audio, v):
                all_sim += torch.exp(torch.dot(i[0], i[1]))
            all_sim_list_2.append(all_sim)

        numerator = torch.stack(all_sim_list)
        denomerator = torch.stack(all_sim_list_2)

        loss = -torch.log(numerator/denomerator).mean()

        return loss