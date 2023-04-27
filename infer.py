'''
does folder inference alongwith heatmap
'''

import torch
import numpy as np

import argparse, json
import os, glob, sys, shutil
from time import time
import pdb
from dataloader import DRIVE_folder, ROSE_folder
import torch
torch.cuda.empty_cache()
from torchvision.utils import save_image
from probabilistic_unet import ProbabilisticUnet
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
num_samples = 10

def parse_func(args):
    ### Reading the parameters json file
    print("Reading params file {}...".format(args.params))
    with open(args.params, 'r') as f:
        params = json.load(f)

    activity = params['common']['activity']
    mydict = {}
    mydict['folder'] = [params['common']['img_file'], params['common']['gt_file']]
    mydict['checkpoint_restore'] = params['common']['checkpoint_restore']

    mydict['test_datalist'] = params['validation']['validation_datalist']
    mydict['output_folder'] = params['validation']['output_folder']
    mydict['validation_batch_size'] = int(params['validation']['validation_batch_size'])
    
    return activity, mydict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help="Path to the parameters file")
    parser.add_argument('--dataset', type= str, default = "CREMI")
    parser.add_argument('--folder', type= str, default = "version1")   

    if len(sys.argv) == 1:
        print("Path to parameters file not provided. Exiting...")

    else:
        args = parser.parse_args()
        activity, mydict = parse_func(args)

    with open(args.params, 'r') as f:
        params = json.load(f)

    #mydict['output_folder'] = 'experiments/' + args.dataset + '/' + args.folder
    # import pdb; pdb.set_trace()
    
    # call train
    print("Inference!")
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    if not os.path.exists(mydict['output_folder']):
        os.makedirs(mydict['output_folder'])

    # Test Data
    if args.dataset == 'DRIVE':
        validation_set = DRIVE_folder(mydict['test_datalist'], mydict['folder'])
        validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['validation_batch_size'],shuffle=False,num_workers=1, drop_last=False)
        net = ProbabilisticUnet(input_channels=3, num_classes=1, num_filters=[32,64,128,192], latent_dim=1, no_convs_fcomb=4, beta=10.0)

    elif args.dataset == 'ROSE':
        validation_set = ROSE_folder(mydict['test_datalist'], mydict['folder'])
        validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['validation_batch_size'],shuffle=False,num_workers=1, drop_last=False)
        net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=1, no_convs_fcomb=4, beta=10.0)

    else:
        print ('Wrong dataloader!')

    if mydict['checkpoint_restore'] != "":
        checkpoint = torch.load(mydict['checkpoint_restore'] )
        net.load_state_dict(checkpoint['state_dict'])
    else:
        print("No model found!")
        sys.exit()

    validation_start_time = time()

    with torch.no_grad():
        net.eval()
        validation_iterator = iter(validation_generator)
        for i in range(len(validation_generator)):

            mask_pred_list = []
            x, y_gt, filename = next(validation_iterator)
            x = x.to(device, non_blocking=True)
            y_gt = y_gt.to(device, non_blocking=True)

            net.forward(x, segm=y_gt, training=False)

            # num_preds = 4
            # predictions = []
            # for i in range(num_preds):
            #     mask_pred = net.sample(testing=True, use_prior_mean = False)
            #     mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
            #     mask_pred = torch.squeeze(mask_pred, 0)
            #     predictions.append(mask_pred)
            # predictions = torch.cat(predictions, 0)

            for ns in range(num_samples):
                mask_pred = torch.sigmoid(net.sample(testing=True))
                mask_pred_list.append(torch.squeeze(mask_pred, 0))
                
            # mask_pred = torch.sigmoid(mask_pred) 
            # mask_pred_binary = (mask_pred > 0.5).float()
            # mask_pred_binary = torch.squeeze(mask_pred_binary, 0)
            # mask_pred_binary = np.squeeze(mask_pred, 0)
            # pdb.set_trace()
            filename = filename[0]
            tempdir = os.path.join(mydict['output_folder'], filename.split('/')[-2])
            if not os.path.exists(tempdir):
                os.makedirs(tempdir)
            #pdb.set_trace()
            for j in range(x.shape[0]):
                save_image(x[j,:,:,:], os.path.join(tempdir, 'img_' + filename.split('/')[-1]  + '.png'))
                save_image(y_gt[j,:,:], os.path.join(tempdir, 'gt_' + filename.split('/')[-1] + '.png'))
                # save_image(mask_pred[j,:,:].float(), os.path.join(mydict['output_folder'], 'img' + str(mydict['validation_batch_size'] * i + j) + '_pred.png'))
                
                var_map = np.var(torch.squeeze(torch.stack(mask_pred_list, 0)).cpu().detach().numpy(), axis=0)
                var_map = var_map/np.max(var_map)
                ax = sns.heatmap(var_map, cmap=plt.cm.coolwarm, vmin=0, vmax=1)
                ax.set_axis_off()
                plt.show()
                plt.savefig(os.path.join(tempdir, 'heatmap_' + filename.split('/')[-1] + '.png'), bbox_inches='tight', pad_inches=0)
                plt.clf()

                for ns in range(num_samples): # save binary mask
                    tempimg = mask_pred_list[ns][j,:,:].detach().cpu().numpy()
                    tempimg = np.where(tempimg >= 0.5, 1., 0.)
                    tempimg = (tempimg*255.).astype(np.uint8)
                    im_pred = Image.fromarray(tempimg)
                    im_pred.save(os.path.join(tempdir, 'pred_' + filename.split('/')[-1] + '_sample{}.png'.format(ns)))
                    
                    #save_image(mask_pred_list[ns][j,:,:], os.path.join(tempdir, 'pred_' + filename.split('/')[-1] + '_sample{}.png'.format(ns)))
            print("Done: {}".format(filename))