#compute metrics for 2D BINARY images
from skimage.morphology import skeletonize, skeletonize_3d
from PIL import Image
import numpy as np
import os, glob, sys
from skimage.metrics import hausdorff_distance, adapted_rand_error, variation_of_information
import pdb
#predir = "/data/saumgupta/crf-dmt/DRIVE/unet/test-outputs/trial20-ce-dmt-pystruct-Th01"
#gtdir = "/data/saumgupta/crf-dmt/DRIVE/test/1st_manual-skeleton"

#predir = "/data/saumgupta/DRIVE/csnet/test-outputs/bce/0-1000/epoch905"
#gtdir = "/data/saumgupta/DRIVE/test/1st_manual-png"

predir = "/data/saumgupta/prob_unet/experiments/ROSE/test-outputs/model_best/img"
gtdir = "/data/saumgupta/prob_unet/experiments/ROSE/test-outputs/model_best/img"

#predir = "/scr/saumgupta/crf-dmt/testing-temp/data/DRIVE/unet/test-outputs/mcsamples/trial28-ce-dmt-mlp-uncertainty-Th002/epoch628-gt03-Th002/unionCNN"
#gtdir = "/scr/saumgupta/crf-dmt/testing-temp/data/DRIVE/test/1st_manual-png"

def main():

    filelist = glob.glob(gtdir+'/gt*.png')
    filelist.sort()
    
    avg = {'dice':[], 'hd':[], '1-ari':[], 'voi':[], 'cldice':[]}
    cnt = 0
    with open(os.path.join(predir,'metric_values.txt'), 'w') as wfile:
        wfile.write("Folder: {}".format(predir))
        for i, gtpath in enumerate(filelist):
            predname = gtpath.split('/')[-1].split('_')[1].replace(".png", "")
            predpath = glob.glob(predir + "/pred_" + predname + "*.png")
            assert len(predpath) >= 1
            predpath = predpath[0] # taking any one sample

            if os.path.exists(predpath):
                cnt+=1
                
                pred = np.array(Image.open(predpath))/255.
                target = np.array(Image.open(gtpath))[:,:,0]/255. # if saved as an RGB image (png)
                #target = np.array(Image.open(gtpath))/255. # if saved as tif

                #print("Pred: ", pred.shape, np.min(pred), np.max(pred))
                #print("Target: ", target.shape, np.min(target), np.max(target))
                #pred = pred[:,:565] # if DRIVE

                diceCoeff = calculateDiceCoeff(pred, target)
                hdCoeff = calculateHDCoeff(pred,target)
                ariCoeff  = calculateARICoeff(pred,target)
                voiCoeff = calculateVOICoeff(pred,target)
                clCoeff = clDice(pred,target)
                avg['dice'].append(diceCoeff)
                avg['hd'].append(hdCoeff)
                avg['1-ari'].append(ariCoeff)
                avg['voi'].append(voiCoeff)
                avg['cldice'].append(clCoeff)

                wfile.write("\n\nFile: {}\nDice: {}\nHD: {}\n1-ARI: {}\nVOI: {}\nclDice: {}".format(gtpath, diceCoeff, hdCoeff, ariCoeff, voiCoeff, clCoeff))

        wfile.write("\n\nFolder: {}\nTotal: {}\n".format(predir,cnt))
        for key,item in avg.items():
            wfile.write("Avg {}: {}\n".format(key,np.mean(np.array(item))))
            wfile.write("Sttdev {}: {}\n".format(key,np.std(np.array(item))))


def calculateDiceCoeff(pred,target):

    m1 = pred.flatten().astype(np.float32)  # Flatten
    m2 = target.flatten().astype(np.float32)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection) / (m1.sum() + m2.sum())


def calculateHDCoeff(pred,target):
    return hausdorff_distance(pred, target)

def calculateARICoeff(pred,target):
    return 1.-adapted_rand_error(target.astype(np.int32),pred.astype(np.int32))[0]

def calculateVOICoeff(pred,target):
    return np.sum(variation_of_information(target.astype(np.int32),pred.astype(np.int32)))



def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)




if __name__ == "__main__":
    main()