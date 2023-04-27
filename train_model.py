from builtins import print
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
from dataloader import CREMI, ISBI2013, DRIVE, ROSE
import argparse, sys, json
import os, shutil
from saver import Saver
import torch.nn as nn
import pdb
def parse_func(args):
    ### Reading the parameters json file
    print("Reading params file {}...".format(args.params))
    with open(args.params, 'r') as f:
        params = json.load(f)

    activity = params['common']['activity']
    mydict = {}
    mydict['files'] = [params['common']['img_file'], params['common']['gt_file']]

    mydict['output_folder'] = params['train']['output_folder']
    mydict['train_datalist'] = params['train']['train_datalist']
    mydict['validation_datalist'] = params['train']['validation_datalist']
    mydict['train_batch_size'] = int(params['train']['train_batch_size'])
    mydict['validation_batch_size'] = int(params['train']['validation_batch_size'])

    return activity, mydict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help="Path to the parameters file")
    parser.add_argument('--dataset', type= str, default = "CREMI")
    parser.add_argument('--pretrain', type= str, default = "True")
    parser.add_argument('--weight_reconstruction', type= float, default = 1)
    parser.add_argument('--weight_kl', type= float, default = 10)
    parser.add_argument('--weight_reg', type= float, default = 1e-5)
    parser.add_argument('--lr', type= float, default = 1e-4) # decrease it in case of nan issues
    parser.add_argument('--epochs', type= int, default = 10000)
    parser.add_argument('--train_batch', type=int, default = 16, help="batch size for training")
    parser.add_argument('--resume', type= str, default = "scratch")
    
    if len(sys.argv) == 1:
        print("Path to parameters file not provided. Exiting...")

    else:
        args = parser.parse_args()
        activity, mydict = parse_func(args)

    mydict['train_batch_size'] = args.train_batch

    with open(args.params, 'r') as f:
        params = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    saver = Saver(args)

    # Train Data
    if args.dataset == 'CREMI':
        training_set = CREMI(mydict['train_datalist'], mydict['files'], is_training= True)
        training_generator = torch.utils.data.DataLoader(training_set,batch_size=mydict['train_batch_size'],shuffle=True,num_workers=1, drop_last=True)

        # Validation Data
        validation_set = CREMI(mydict['validation_datalist'], mydict['files'])
        validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['validation_batch_size'],shuffle=False,num_workers=1, drop_last=False)
        net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=1, no_convs_fcomb=4, beta=10.0)

    elif args.dataset == 'ISBI2013':         
        training_set = ISBI2013(mydict['train_datalist'], mydict['files'], is_training= True)
        training_generator = torch.utils.data.DataLoader(training_set,batch_size=mydict['train_batch_size'],shuffle=True,num_workers=1, drop_last=True)

        # Validation Data
        validation_set = ISBI2013(mydict['validation_datalist'], mydict['files'])
        validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['validation_batch_size'],shuffle=False,num_workers=1, drop_last=False)
        net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=1, no_convs_fcomb=4, beta=10.0)
    elif args.dataset == 'DRIVE':         
        training_set = DRIVE(mydict['train_datalist'], mydict['files'], is_training= True)
        training_generator = torch.utils.data.DataLoader(training_set,batch_size=mydict['train_batch_size'],shuffle=True,num_workers=1, drop_last=True)

        # Validation Data
        validation_set = DRIVE(mydict['validation_datalist'], mydict['files'], is_training= True, constcorner=True)
        validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['validation_batch_size'],shuffle=False,num_workers=1, drop_last=False)
        net = ProbabilisticUnet(input_channels=3, num_classes=1, num_filters=[32,64,128,192], latent_dim=1, no_convs_fcomb=4, beta=10.0)

    elif args.dataset == 'ROSE':         
        training_set = ROSE(mydict['train_datalist'], mydict['files'], is_training= True)
        training_generator = torch.utils.data.DataLoader(training_set,batch_size=mydict['train_batch_size'],shuffle=True,num_workers=1, drop_last=True)

        # Validation Data
        validation_set = ROSE(mydict['validation_datalist'], mydict['files'], is_training= True, constcorner=True)
        validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['validation_batch_size'],shuffle=False,num_workers=1, drop_last=False)
        net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=1, no_convs_fcomb=4, beta=10.0)

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0)

    best_pred = 10000
    if args.resume == "best":
        print('Finetune the best model!')
        path = os.path.join(mydict['output_folder'], os.path.join('experiments', args.dataset)) + '/model_best.pth.tar'
        checkpoint = torch.load(path)
        args.start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler']),
        best_pred = checkpoint['best_pred']
    elif args.resume == "last":
        print('Finetune the last model!')
        path = os.path.join(mydict['output_folder'], os.path.join('experiments', args.dataset)) + '/model_last.pth.tar'
        checkpoint = torch.load(path)
        args.start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler']),
        best_pred = checkpoint['best_pred']
    elif args.resume == "baseline":        
        print('Finetune the baseline model!')
        path = os.path.join(mydict['output_folder'], os.path.join('experiments', args.dataset)) + '/model_baseline.pth.tar'
        checkpoint = torch.load(path)
        args.start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # import pdb; pdb.set_trace()
        # scheduler.load_state_dict(checkpoint['scheduler']),
        best_pred = 10000
    else:
        print('Train from scratch!')

    saver.save_experiment_config()

    for epoch in range(args.epochs):
        for step, (patch, mask) in enumerate(training_generator): 
            print("Training at step {} of epoch {}".format(step, epoch))
            patch = patch.to(device, dtype=torch.float)
            mask = mask.to(device, dtype=torch.float)
            net.forward(patch, mask, training=True)
            elbo = net.elbo(mask)
            reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        is_best = False

        if epoch % 10 == 0:
            with torch.no_grad():
                net.eval()
                validation_iterator = iter(validation_generator)
                avg_val_loss = 0.0
                for _ in range(len(validation_generator)):
                    x, y_gt = next(validation_iterator)
                    x = x.to(device, non_blocking=True)
                    y_gt = y_gt.to(device, non_blocking=True)
                    criterion = nn.BCELoss(size_average = False, reduce=False, reduction=None)
                    net.forward(x, y_gt, training = False)
                    y_pred = net.sample(testing=True)
                    
                    avg_val_loss += torch.mean(criterion(torch.sigmoid(y_pred).type(torch.DoubleTensor), y_gt.type(torch.DoubleTensor)))
                avg_val_loss /= len(validation_generator)

            print("The loss at epoch {} is: {}\n".format(epoch, avg_val_loss))
            if avg_val_loss < best_pred:
                best_pred = avg_val_loss
                is_best = True
                print("Update best loss: {}\n".format(best_pred))
            
            model_state_dict = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': avg_val_loss,
            }, is_best)
        