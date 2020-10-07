from utils import Evaluator, generate_samples, run_evaluate
from networks.net_unetgenerator import UnetGenerator as G
from easydict import EasyDict as edict
from tqdm import tqdm
from data.STB_dataset import STBdataset
from data.RHD_dataset import RHDdataset
import json
import os
import shutil
import cv2
from multiprocessing import Pool
TEMP_PATH = "P2P_TEMP"

def run_js(*args):
    os.system(f"node test.js {args[0]}")
if __name__ == "__main__":
    #generate temp file
    if os.path.exists(TEMP_PATH): shutil.rmtree(TEMP_PATH)
    os.mkdir(TEMP_PATH)
    # init generator model
    model = G(3, 3, 8)
    device = 0
##################################################################
    # define model's specific weights
    cpm2d = './weights/cpm/stb_hpm2d.pth'
    cpm3d = './weights/cpm/stb_hpm3d.pth'
    weights = "./weights/pix2pix/stb_net_G.pth"
    # init paramters
    evaluate = Evaluator(model, weights,cpm2d, cpm3d, device)
    opt = edict()
    opt.dataroot = "./datasets/stb-dataset/test"
    opt.isTrain = False
    dataset = STBdataset(opt)
    # generate samples
    output_path = os.path.join(TEMP_PATH, "STB")
    generate_samples(model , evaluate, dataset, output_path)
    # run evaluation
    samples_dir = [os.path.join(output_path, i) for i in os.listdir(output_path)]
    p = Pool()
    r = list(tqdm(p.imap(run_js, samples_dir),
                  total=len(samples_dir)))
    p.close()
    p.join()

    #os.system(f"node test.js {output_path}")
    run_evaluate(output_path)
##################################################################3
    # opt.dataroot = "./datasets/rhd-dataset/test"
    # cpm2d = "./weights/cpm/rhd_hpm2d.pth"
    # weights = "./weights/pix2pix/rhd_net_G.pth"

    # dataset = RHDdataset(opt)
    # evaluate = Evaluator(model, weights, cpm2d, cpm3d, device)
    # output_path = os.path.join(TEMP_PATH, "RHD")
    # generate_samples(model, evaluate, output_path)
    # # run evaluation
    # os.system(f"node test.js {output_path}")
    # run_evaluate(output_path)
####################################################################
    shutil.rmtree(TEMP_PATH)
