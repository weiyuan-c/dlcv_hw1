import os
import random
import argparse
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import skimage
import numpy as np
from PIL import Image
from tqdm import tqdm

from mean_iou_evaluate import mean_iou_score, read_masks
from model import vgg_fcn8, vgg_fcn8_2


class Inf_SegDataset(Dataset):
    def __init__(self, path, sat=None):
        super(Inf_SegDataset).__init__()
        self.path = path
        self.sat = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith('.jpg')])
        if sat != None:
            self.sat = sat

    def __len__(self):
        return len(self.sat)

    def __getitem__(self, idx):
        sat_name = self.sat[idx]
        img = Image.open(sat_name)
        img = T.ToTensor()(img)
        label = str(sat_name.split('/')[-1].split('_')[0])

        return img, label


def recon_seg(mask):
    mapping = [[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]
    recon = np.zeros((512, 512, 3), dtype=np.int32)
    for i in range(512):
        for j in range(512):
            idx = mask[i, j]
            recon[i, j, :] = mapping[idx]
    return recon * 255


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument('-t', '--test_dir', type=str, default='hw1_data/p2_data/validation')
    parser.add_argument('--output_dir', type=str, default='test')
    parser.add_argument('-w1', '--weight1', type=str, default='p2_model1.ckpt')
    parser.add_argument('-w2', '--weight2', type=str, default='p2_model2.ckpt')
    parser.add_argument('-w3', '--weight3', type=str, default='p2_model3.ckpt')

    # hyper-parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=512)
    cfg = parser.parse_args()
    

    # setting seed
    seed = 4956238
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    val_set = Inf_SegDataset(cfg.test_dir)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # simple model
    model1 = vgg_fcn8_2(weights=None).to(device)
    model2 = vgg_fcn8_2(weights=None).to(device)
    # 2
    model3 = vgg_fcn8(weights=None).to(device)
    

    if device == 'cpu':
        model1.load_state_dict(torch.load(cfg.weight1, map_location='cpu'))
        model2.load_state_dict(torch.load(cfg.weight2, map_location='cpu'))
        model3.load_state_dict(torch.load(cfg.weight3, map_location='cpu'))
    else:
        model1.load_state_dict(torch.load(cfg.weight1))
        model2.load_state_dict(torch.load(cfg.weight2))
        model3.load_state_dict(torch.load(cfg.weight3))

    model1.eval()
    model2.eval()
    model3.eval()


    for batch in tqdm(val_loader):
        img, name = batch
        with torch.no_grad():
            output1 = model1(img)
            output2 = model2(img)
            output3 = model3(img)
        output = (output1 + output2 + output3) / 3
        output = output.argmax(1)

        for i in range(output.shape[0]):
            im = output[i].detach().cpu().numpy()
            fname = name[i]
            recon = recon_seg(im)
            recon = np.uint8(recon)
            save_dir = f'{cfg.output_dir}/{fname}.png'
            skimage.io.imsave(save_dir, recon, check_contrast=False)