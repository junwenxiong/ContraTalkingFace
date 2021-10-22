import torch
import os, cv2
import numpy as np
from os.path import join

def save_sample_images(x, g, gt, global_step, checkpoint_dir, stage="train"):
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)

    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(
        np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(
        np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(
        np.uint8)
    refs, inps = x[..., 3:], x[..., :3]
    collage = np.concatenate((refs, inps, g, gt), axis=-2)

    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}_{}.jpg'.format(folder, stage, batch_idx, t), c[t])

def save_checkpoint(args, model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, step))
    optimizer_state = optimizer.state_dict() if args.save_optimizer_state else None
    torch.save({
            "state_dict": model.state_dict(),
            "optimizer": optimizer_state,
            "global_step": step,
            "global_epoch": epoch,
        }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(use_cuda, checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(use_cuda,
                    path,
                    model,
                    optimizer,
                    step = 0, 
                    epoch = 0,
                    reset_optimizer=False,
                    overwrite_global_states=True):

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(use_cuda, path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        step = checkpoint["global_step"]
        epoch = checkpoint["global_epoch"]

    return model, step, epoch

def load_test_model(use_cuda, path, model):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(use_cuda, path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.cuda()
    return model.eval()