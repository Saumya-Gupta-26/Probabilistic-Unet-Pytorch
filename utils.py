import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np 
import pdb, os, glob, sys, shutil


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def save_mask_prediction_example(mask, pred, iter):
	plt.imshow(pred[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_prediction.png")
	plt.imshow(mask[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_mask.png")

# saumya
def compute_ece_plot(pred_lm, target, plotfile=None, num_bins=100):

    #pred_lm = pred_lm[:,:565] # for DRIVE
    #pdb.set_trace()

    # flatten to 1D
    pred_binary = np.where(pred_lm >= 0.5, 1., 0.).flatten()
    pred_lm = pred_lm.flatten()
    target = target.flatten()

    binpoints = np.linspace(start=0., stop=1., num=num_bins+1)

    # sort based on probability values ; create the bins
    cat = np.rec.fromarrays([pred_lm, pred_binary, target]) # f0, f1, f2
    cat.sort()

    stats = {'counts':np.zeros(len(binpoints)),'acc':np.zeros(len(binpoints)),'conf':np.zeros(len(binpoints))}
    running = {'counts':0., 'acc':0., 'conf':0.}
    idx = 0 # iterates through samples in cat ; jidx iterates through bins
    
    for jdx, val in enumerate(binpoints):
        if jdx == 0:
            continue
        while idx < len(cat.f0) and cat.f0[idx] <= val:
            running['counts'] += 1
            running['conf'] += cat.f0[idx]
            if cat.f1[idx] == cat.f2[idx]:
                running['acc'] += 1
            idx += 1
        
        stats['counts'][jdx] = running['counts']
        if running['counts'] != 0:
            stats['acc'][jdx] = running['acc']/running['counts']
            stats['conf'][jdx] = running['conf']/running['counts']

        if idx >= len(cat.f0):
            break
        running = {'counts':0., 'acc':0., 'conf':0.}
            
    # ece computation and plot dump
    total_samples = len(target)
    ece_val = 0.

    if plotfile:
        writefile = open(plotfile, 'w')
        writefile.write("counts,accuracy,confidence; total count={}\n".format(total_samples))

    for idx in range(stats['counts'].shape[0]):
        ece_val += (stats['counts'][idx] * np.abs(stats['acc'][idx] - stats['conf'][idx]))
        
        if plotfile:
            writefile.write("{},{},{}\n".format(stats['counts'][idx], stats['acc'][idx],stats['conf'][idx]))

    ece_val = ece_val / total_samples
    
    if plotfile:
        writefile.close()

    return ece_val