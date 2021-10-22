import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import grad
from tqdm import tqdm
from models import SyncNet_color as SyncNet
from models import Wav2Lip, Wav2Lip_disc_qual, VGGLoss, Wav2Lip_ori, Sync_loss
from utils.checkpoint import load_checkpoint, save_checkpoint, save_sample_images


class W2LNet(nn.Module):
    def __init__(self, args, configLogger, global_step=0, global_epoch=0):
        super().__init__()

        self.use_cuda = torch.cuda.is_available()
        self.args = args
        self.model = Wav2Lip_ori(add_mouths=False).cuda()
        self.disc = Wav2Lip_disc_qual().cuda()
        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.args.initial_learning_rate,
            betas=(0.5, 0.999))
        self.disc_optimizer = optim.Adam(
            [p for p in self.disc.parameters() if p.requires_grad],
            lr=self.args.disc_initial_learning_rate,
            betas=(0.5, 0.999))

        self.syncnet = SyncNet().cuda()
        for p in self.syncnet.parameters():
            p.requires_grad = False

        self.criterionVGG = VGGLoss()
        self.criterionSync = Sync_loss(self.syncnet, self.args.syncnet_T)
        self.logloss = nn.BCELoss()
        self.recon_loss = nn.L1Loss()

        self.configLogger = configLogger

        self.configLogger.add_line("-" * 20 + "Training" + "-" * 20)
        self.configLogger.add_line('total trainable params {}'.format(
            sum(p.numel() for p in self.model.parameters()
                if p.requires_grad)))
        self.configLogger.add_line('total DISC trainable params {}'.format(
            sum(p.numel() for p in self.disc.parameters() if p.requires_grad)))

        self.global_step = global_step
        self.global_epoch = global_epoch

        self.load_model()

    def load_model(self):
        self.syncnet, _, _ = load_checkpoint(self.use_cuda ,
                                        self.args.syncnet_checkpoint_path,
                                        self.syncnet,
                                        None,
                                        reset_optimizer=True,
                                        overwrite_global_states=False)

        if self.args.checkpoint_path is not None:
            self.model, step, epoch = load_checkpoint(self.use_cuda , self.args.checkpoint_path,
                                                 self.model, self.optimizer,
                                                 reset_optimizer=False)
            self.global_step = step
            self.global_epoch = epoch

        if self.args.disc_checkpoint_path is not None:
            self.disc, _, _ = load_checkpoint(self.use_cuda ,
                                         self.args.disc_checkpoint_path,
                                         self.disc,
                                         self.disc_optimizer,
                                         reset_optimizer=False,
                                         overwrite_global_states=False)

    def train_network(self, loader, test_data_loader):
        self.model.train()
        self.disc.train()

        print('Starting Epoch: {}'.format(self.global_epoch))
        running_sync_loss, running_l1_loss, running_perceptual_loss, running_vgg_loss = 0., 0., 0., 0.
        running_disc_real_loss, running_disc_fake_loss = 0., 0.
        running_total_loss = 0.
        prog_bar = tqdm(enumerate(loader))
        for step, (x, indiv_mels, mel, mouths, gt) in prog_bar:
            x = x.cuda()
            mel = mel.cuda()  # 视觉输入所对应的完整的mel spectrogram
            indiv_mels = indiv_mels.cuda()  # mel分段后的mel spectrogram
            mouths = mouths.cuda()
            gt = gt.cuda()

            ### Train generator now. Remove ALL grads.
            self.optimizer.zero_grad()
            self.disc_optimizer.zero_grad()

            g = self.model.forward(indiv_mels, x, mouths)

            if self.args.syncnet_wt > 0.:
                sync_loss = self.criterionSync(mel, g)
            else:
                sync_loss = 0.

            if self.args.disc_wt > 0.:
                perceptual_loss = self.disc.perceptual_forward(g)
            else:
                perceptual_loss = 0.

            vggloss = self.criterionVGG(g, gt)
            l1loss = self.recon_loss(g, gt)

            loss = self.args.syncnet_wt * sync_loss + self.args.disc_wt * perceptual_loss + \
               (1. - self.args.syncnet_wt - self.args.disc_wt - self.args.vgg_wt) * l1loss + self.args.vgg_wt * vggloss

            loss.backward()
            self.optimizer.step()

            # 训练discriminator
            ### Remove all gradients before Training disc
            self.disc_optimizer.zero_grad()

            pred = self.disc(gt)
            disc_real_loss = F.binary_cross_entropy(
                pred,
                torch.ones((len(pred), 1)).cuda())
            disc_real_loss.backward()

            pred = self.disc(g.detach())
            disc_fake_loss = F.binary_cross_entropy(
                pred,
                torch.zeros((len(pred), 1)).cuda())
            disc_fake_loss.backward()

            self.disc_optimizer.step()

            running_disc_real_loss += disc_real_loss.item()
            running_disc_fake_loss += disc_fake_loss.item()

            # Logs
            self.global_step += 1

            running_total_loss += loss.item()
            running_l1_loss += l1loss.item()
            running_vgg_loss += vggloss.item()
            if self.args.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            if self.args.disc_wt > 0.:
                running_perceptual_loss += perceptual_loss.item()
            else:
                running_perceptual_loss += 0.

            if self.global_step == 1 or self.global_step % self.args.save_image_interval == 0:
                save_sample_images(x, g, gt, self.global_step, self.args.checkpoint_dir)  # 展示训练效果图

            if self.global_step % self.args.checkpoint_interval == 0:
                save_checkpoint(self.args, self.model, self.optimizer, self.global_step, self.args.checkpoint_dir, self.global_epoch)
                save_checkpoint(self.args, self.disc, self.disc_optimizer, self.global_step, self.args.checkpoint_dir, self.global_epoch, prefix='disc_')

            if self.global_step % self.args.eval_interval == 0:
                with torch.no_grad():
                    average_sync_loss = self.eval_model(test_data_loader)

                if average_sync_loss < .75:
                    print("change syncnet_wt to 0.03")
                    self.args.syncnet_wt = 0.03

            prog_bar.set_description(
                'Epoch: {}, Total Loss: {:.5f}, L1: {:.5f}, Sync: {:.5f}, Percep: {:.5f}, VGG: {:.5f} | Fake: {:.5f}, Real: {:.5f}'
                .format(self.global_epoch, 
                        running_total_loss / (step + 1),
                        running_l1_loss / (step + 1),
                        running_sync_loss / (step + 1),
                        running_perceptual_loss / (step + 1),
                        running_vgg_loss / (step + 1),
                        running_disc_fake_loss / (step + 1),
                        running_disc_real_loss / (step + 1)))

        self.global_epoch += 1
        self.configLogger.add_line(
            'Training. Epoch: {}, Total Loss: {:.5f}, L1: {:.5f}, Sync: {:.5f}, Percep: {:.5f}, VGG: {:.5f} | Fake: {:.5f}, Real: {:.5f}'
            .format(self.global_epoch, 
                    running_total_loss / (step + 1),
                    running_l1_loss / (step + 1),
                    running_sync_loss / (step + 1),
                    running_perceptual_loss / (step + 1),
                    running_vgg_loss / (step + 1),
                    running_disc_fake_loss / (step + 1),
                    running_disc_real_loss / (step + 1)))

    def eval_model(self, test_data_loader):
        eval_steps = 300
        print('Evaluating for {} steps'.format(eval_steps))
        running_sync_loss, running_l1_loss, running_disc_real_loss, running_disc_fake_loss, running_perceptual_loss = [], [], [], [], []
        running_vgg_loss, running_total_loss = [], []
        for step, (x, indiv_mels, mel, mouths, gt) in enumerate((test_data_loader)):
            self.model.eval()
            self.disc.eval()

            x = x.cuda()
            mel = mel.cuda()
            indiv_mels = indiv_mels.cuda()
            mouths = mouths.cuda()
            gt = gt.cuda()

            pred = self.disc(gt)
            disc_real_loss = F.binary_cross_entropy(
                pred,
                torch.ones((len(pred), 1)).cuda())
            g = self.model(indiv_mels, x, mouths)
            pred = self.disc(g)
            disc_fake_loss = F.binary_cross_entropy(
                pred,
                torch.zeros((len(pred), 1)).cuda())

            running_disc_real_loss.append(disc_real_loss.item())
            running_disc_fake_loss.append(disc_fake_loss.item())

            sync_loss = self.criterionSync(mel, g)
            l1loss = self.recon_loss(g, gt)
            running_l1_loss.append(l1loss.item())
            running_sync_loss.append(sync_loss.item())

            vggloss = self.criterionVGG(g, gt)
            running_vgg_loss.append(vggloss.item())

            if self.args.disc_wt > 0.:
                perceptual_loss = self.disc.perceptual_forward(g)
                running_perceptual_loss.append(perceptual_loss.item())
            else:
                running_perceptual_loss.append(0.)

            loss = self.args.syncnet_wt * sync_loss + self.args.disc_wt * perceptual_loss + \
               (1. - self.args.syncnet_wt - self.args.disc_wt - self.args.vgg_wt) * l1loss + self.args.vgg_wt * vggloss
            running_total_loss.append(loss.item())

            if self.global_step % self.args.save_image_interval == 0:
                save_sample_images(x, g, gt, self.global_step, self.args.checkpoint_dir, stage='val')

            if step > eval_steps: break

        self.configLogger.add_line( '-' * 3 + 'Evaluation' + '-' * 3 + 'Epoch: {}, Total Loss: {:.5f}, L1: {:.5f}, Sync: {:.5f}, Percep: {:.5f}, VGG: {:.5f} | Fake: {:.5f}, Real: {:.5f}' .format(
                self.global_epoch,
                sum(running_total_loss) / len(running_total_loss),
                sum(running_l1_loss) / len(running_l1_loss),
                sum(running_sync_loss) / len(running_sync_loss),
                sum(running_perceptual_loss) /
                len(running_perceptual_loss),
                sum(running_vgg_loss) / len(running_vgg_loss),
                sum(running_disc_fake_loss) / len(running_disc_fake_loss),
                sum(running_disc_real_loss) / len(running_disc_real_loss)))

        return sum(running_sync_loss) / len(running_sync_loss)

    def train_network_lrs2(self, loader, test_data_loader):
        self.model.train()
        self.disc.train()

        print('Starting Epoch: {}'.format(self.global_epoch))
        running_sync_loss, running_l1_loss, running_perceptual_loss, running_vgg_loss = 0., 0., 0., 0.
        running_disc_real_loss, running_disc_fake_loss = 0., 0.
        prog_bar = tqdm(enumerate(loader))
        for step, (x, indiv_mels, mel, gt, inver_x, inver_indiv_mels, inver_mels, inver_y) in prog_bar:
            x = x.cuda()
            mel = mel.cuda()  # 视觉输入所对应的完整的mel spectrogram shape: (bs, 1, 80, 16)
            indiv_mels = indiv_mels.cuda()  # mel分段后的mel spectrogram
            gt = gt.cuda()

            inver_x = inver_x.cuda()
            inver_indiv_mels = inver_indiv_mels.cuda()
            inver_mels = inver_mels.cuda()
            inver_y = inver_y.cuda()

            ### Train generator now. Remove ALL grads.
            self.optimizer.zero_grad()
            self.disc_optimizer.zero_grad()

            g = self.model.forward(indiv_mels, x)
            inver_g = self.model.forward(inver_indiv_mels, inver_x)

            # import pdb; pdb.set_trace()
            if self.args.syncnet_wt > 0.:
                sync_loss = self.criterionContra(mel, g, inver_mels, inver_g)
                # sync_loss = self.criterionSync(mel, g) + self.criterionSync(inver_mels, inver_g)
            else:
                sync_loss = 0.

            if self.args.disc_wt > 0.:
                perceptual_loss = self.disc.perceptual_forward(g) + self.disc.perceptual_forward(inver_g)
            else:
                perceptual_loss = 0.

            vggloss = self.criterionVGG(g, gt) + self.criterionVGG(inver_g, inver_y)
            l1loss = self.recon_loss(g, gt) + self.recon_loss(inver_g, inver_y)

            loss = self.args.syncnet_wt * sync_loss + self.args.disc_wt * perceptual_loss + \
               (1. - self.args.syncnet_wt - self.args.disc_wt - self.args.vgg_wt) * l1loss + self.args.vgg_wt * vggloss

            loss.backward()
            self.optimizer.step()

            # 训练discriminator
            ### Remove all gradients before Training disc
            self.disc_optimizer.zero_grad()

            pred = self.disc(gt)
            inver_pred = self.disc(inver_y)
            disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).cuda()) \
                + F.binary_cross_entropy(inver_pred, torch.ones((len(inver_pred), 1)).cuda())
            disc_real_loss.backward()

            pred = self.disc(g.detach())
            inver_pred = self.disc(inver_g.detach())
            disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).cuda()) \
                + F.binary_cross_entropy(inver_pred, torch.zeros((len(inver_pred), 1)).cuda())
            disc_fake_loss.backward()

            self.disc_optimizer.step()

            running_disc_real_loss += disc_real_loss.item()
            running_disc_fake_loss += disc_fake_loss.item()

            # Logs
            self.global_step += 1

            running_l1_loss += l1loss.item()
            running_vgg_loss += vggloss.item()
            if self.args.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            if self.args.disc_wt > 0.:
                running_perceptual_loss += perceptual_loss.item()
            else:
                running_perceptual_loss += 0.

            if self.global_step == 1 or self.global_step % self.args.save_image_interval == 0:
                save_sample_images(x, g, gt, self.global_step, self.args.checkpoint_dir)  # 展示训练效果图

            if self.global_step % self.args.checkpoint_interval == 0:
                save_checkpoint(self.args, self.model, self.optimizer, self.global_step, self.args.checkpoint_dir, self.global_epoch)
                save_checkpoint(self.args, self.disc, self.disc_optimizer, self.global_step, self.args.checkpoint_dir, self.global_epoch, prefix='disc_')

            if self.global_step % self.args.eval_interval == 0:
                with torch.no_grad():
                    average_sync_loss = self.eval_model_lrs2(test_data_loader)

                if average_sync_loss < .75:
                    print("change syncnet_wt to 0.03")
                    self.args.syncnet_wt = 0.03

            prog_bar.set_description(
                'Epoch: {}, L1: {}, Sync: {}, Percep: {}, VGG: {} | Fake: {}, Real: {}'
                .format(self.global_epoch, running_l1_loss / (step + 1),
                        running_sync_loss / (step + 1),
                        running_perceptual_loss / (step + 1),
                        running_vgg_loss / (step + 1),
                        running_disc_fake_loss / (step + 1),
                        running_disc_real_loss / (step + 1)))

        self.global_epoch += 1
        self.configLogger.add_line(
            'Training. Epoch: {}, L1: {}, Sync: {}, Percep: {}, VGG: {} | Fake: {}, Real: {}'
            .format(self.global_epoch, running_l1_loss / (step + 1),
                    running_sync_loss / (step + 1),
                    running_perceptual_loss / (step + 1),
                    running_vgg_loss / (step + 1),
                    running_disc_fake_loss / (step + 1),
                    running_disc_real_loss / (step + 1)))

    def eval_model_lrs2(self, test_data_loader):
        eval_steps = 300
        print('Evaluating for {} steps'.format(eval_steps))
        running_sync_loss, running_l1_loss, running_disc_real_loss, running_disc_fake_loss, running_perceptual_loss = [], [], [], [], []
        running_vgg_loss = []
        for step, (x, indiv_mels, mel, gt, inver_x, inver_indiv_mels, inver_mels, inver_y) in enumerate((test_data_loader)):
            self.model.eval()
            self.disc.eval()
            x = x.cuda()
            mel = mel.cuda()  # 视觉输入所对应的完整的mel spectrogram
            indiv_mels = indiv_mels.cuda()  # mel分段后的mel spectrogram
            gt = gt.cuda()

            inver_x = inver_x.cuda()
            inver_indiv_mels = inver_indiv_mels.cuda()
            inver_mels = inver_mels.cuda()
            inver_y = inver_y.cuda()

            pred = self.disc(gt)
            inver_pred = self.disc(inver_y)
            disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).cuda()) \
                + F.binary_cross_entropy(inver_pred, torch.ones((len(inver_pred), 1)).cuda())

            g = self.model(indiv_mels, x)
            pred = self.disc(g)
            inver_g = self.model(inver_indiv_mels, inver_x)
            inver_pred = self.disc(inver_g)
            disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).cuda()) \
                + F.binary_cross_entropy(inver_pred, torch.zeros((len(inver_pred), 1)).cuda())

            running_disc_real_loss.append(disc_real_loss.item())
            running_disc_fake_loss.append(disc_fake_loss.item())

            sync_loss = self.criterionContra(mel, g, inver_mels, inver_g)
            # sync_loss = self.criterionSync(mel, g) + self.criterionSync(inver_mels, inver_g)
            l1loss = self.recon_loss(g, gt) + self.recon_loss(inver_g, inver_y)
            running_l1_loss.append(l1loss.item())
            running_sync_loss.append(sync_loss.item())

            vggloss = self.criterionVGG(g, gt) + self.criterionVGG(inver_g, inver_y)
            running_vgg_loss.append(vggloss.item())

            if self.args.disc_wt > 0.:
                perceptual_loss = self.disc.perceptual_forward(g) + self.disc.perceptual_forward(inver_g)
                running_perceptual_loss.append(perceptual_loss.item())
            else:
                running_perceptual_loss.append(0.)

            if self.global_step % self.args.save_image_interval == 0:
                save_sample_images(x, g, gt, self.global_step, self.args.checkpoint_dir, stage='val')

            if step > eval_steps: break

        self.configLogger.add_line( '-' * 5 + 'Evaluation' + '-' * 5 + 'Epoch: {}, L1: {}, Sync: {}, Percep: {}, VGG: {} | Fake: {}, Real: {}' .format(
                self.global_epoch,
                sum(running_l1_loss) / len(running_l1_loss),
                sum(running_sync_loss) / len(running_sync_loss),
                sum(running_perceptual_loss) /
                len(running_perceptual_loss),
                sum(running_vgg_loss) / len(running_vgg_loss),
                sum(running_disc_fake_loss) / len(running_disc_fake_loss),
                sum(running_disc_real_loss) / len(running_disc_real_loss)))

        return sum(running_sync_loss) / len(running_sync_loss)
