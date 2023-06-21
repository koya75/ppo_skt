import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

def save_checkpoint(checkpoint_path, policy_model, optimizer):
        torch.save({'optimizer1_state_dict': optimizer.state_dict(),
                    'policy_old_state_dict': policy_model.state_dict(),}, checkpoint_path)
        
def load_checkpoint(checkpoint_path, policy_model, old_policy_model, optimizer=None, device=None):
        ### load checkpoint
        if device is not None:
            _ckpt = torch.load(checkpoint_path, map_location=torch.device(device))
        else:
            _ckpt = torch.load(checkpoint_path)

        ### load state dicts
        policy_model.load_state_dict(_ckpt['policy_old_state_dict'])
        old_policy_model.load_state_dict(_ckpt['policy_old_state_dict'])
        if optimizer is not None :
            optimizer.load_state_dict(_ckpt['optimizer1_state_dict'])
        
        #self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        #self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

        return policy_model, optimizer, old_policy_model

def save_args(save_filename, args):
    with open(save_filename, 'w') as f:
        json.dump(args.__dict__, f, indent=4)


def load_args(load_filename):
    with open(load_filename, 'r') as f:
        args = json.load(f)
    return args

def log_plot(save_path, dir):
    reader = np.loadtxt(save_path, delimiter=',', skiprows=1, unpack=True)
    step = reader[1]#[row[0] for row in l[1:]]
    reward = reader[2]#[row[1] for row in l[1:]]
    episode = reader[0]#[row[2] for row in l[1:]]

    data01_axis1 = episode
    data01_rewards = reward
    fig_1 = plt.figure(figsize=(12, 6))
    ax_1 = fig_1.add_subplot(111)
    ax_1.plot(data01_axis1, data01_rewards,  color="k", label="score")
    ax_1.set_xlabel("episode")
    ax_1.set_ylabel("reward")
    ax_1.legend(loc="upper left")
    plt.savefig(dir+'reward_graph.png', dpi=300)

def min_max(x, mins, maxs, axis=None):
    """_summary_

    Args:
        x (_type_): _description_
        mins (_type_): _description_
        maxs (_type_): _description_
        axis (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    result = (x - mins)/(maxs - mins)
    return result

def make_en_attention(attns):
    reshaped_attns = attns[0].view((256, 16, 16)) # rainbow aqt
    reshaped_attns = torch.mean(reshaped_attns, axis=0)

    return reshaped_attns.cpu().detach().numpy()

def make_en_img(attns, raw_img, step, epi_dir, mode="normal"):
    if mode != "mean":
        raw_img = cv2.imread(epi_dir + "/raw_img/raw_{0:06d}.png".format(step))
        mask = cv2.resize(attns, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
        masked_img = cv2.addWeighted(raw_img, 0.4, mask, 0.6, 0)
        masked_img = cv2.resize(masked_img, dsize=(200, 200))
        cv2.imwrite(epi_dir + "/image_encoder/en_{0:06d}.png".format(step), masked_img)
    else:
        raw_img = cv2.imread(epi_dir + "/raw_img/raw_{0:06d}.png".format(step))
        mask = cv2.resize(attns, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
        masked_img = cv2.addWeighted(raw_img, 0.4, mask, 0.6, 0)
        masked_img = cv2.resize(masked_img, dsize=(200, 200))
        cv2.imwrite(epi_dir + "/en_mean.png", masked_img)
    
    return


def make_de_attention(attns):
    action_attns = []
    ac_attn = attns[0, 0].view(16, 16) # detr_model9
    
    #action_attns.append(ac_attn.cpu().detach().numpy())
    return ac_attn.cpu().detach().numpy() #action_attns

def make_de_img(attns, step, epi_dir):
    raw_img = cv2.imread(epi_dir + "/raw_img/raw_{0:06d}.png".format(step))

    mask = cv2.resize(attns, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)

    masked_img = cv2.addWeighted(raw_img, 0.4, mask, 0.6, 0)
    masked_img = cv2.resize(masked_img, dsize=(200, 200))
    cv2.imwrite(epi_dir + "/decoder_act/de_{:06d}.png".format(step), masked_img)


def sk_make_en_attention(attns, epi_dir):
    attns = attns[0].cpu().detach().numpy()
    fig, ax = plt.subplots(figsize=(16, 16))
    im = ax.imshow(attns)

    ax.set_xticks(range(len(attns)))
    ax.set_yticks(range(len(attns)))
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(attns)):
        for j in range(len(attns)):
            text = ax.text(j, i, "{:.2f}".format(attns[i, j]),
                            ha="center", va="center", color="w")
    """plt.plot(sk_att)
    plt.xticks(range(len(sk_att)))"""
    plt.savefig(epi_dir + '/sketch_encoder/sketch_attention.png', dpi=300)
    plt.close()