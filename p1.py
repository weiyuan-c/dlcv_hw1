import argparse
import os
import glob
import random
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from model import Classifier

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

class ImgDataset(Dataset):
    def __init__(self, path, tfm=None, files=None):
        super(ImgDataset).__init__()
        self.path = path
        self.files = sorted(glob.glob(os.path.join(path, '*')))
        if files != None:
            self.files = files
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        img = Image.open(file_name)
        img = self.transform(img)
        try:
            label = int(file_name.split('/')[-1].split('_')[0])
        except:
            label = -1
        return img, label


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument('-t', '--test_dir', type=str, default='./hw1_data/p1_data/val_50')
    parser.add_argument('-c', '--csv_dir', type=str, default='./prediction.csv')
    parser.add_argument('-w1', '--weight1', type=str, default='./model1.ckpt')
    parser.add_argument('-w2', '--weight2', type=str, default='./model2.ckpt')

    # hyper-parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=224)
    cfg = parser.parse_args()

    # data transfomation
    norm = T.Normalize(mean=[0.485,0.456,0.406], std = [0.229, 0.224, 0.225])
    test_tfm = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(),
        norm
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    # load model
    model1 = Classifier(weights=None).to(device)
    model2 = Classifier(weights=None).to(device)
    # dataset
    val_set = ImgDataset(cfg.test_dir, tfm = test_tfm)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

    # load weight
    model1.load_state_dict(torch.load(cfg.weight1, map_location=device))
    model2.load_state_dict(torch.load(cfg.weight2, map_location=device))

    # validation 
    model1.eval()
    model2.eval()
    val_pred = []
    val_acc = []
    total_len = 0
    for i, batch in enumerate(tqdm(val_loader)):
        img, label = batch
        img = img.to(device)
        label = label.to(device)

        with torch.no_grad():
            output = model1(img)
            output2 = model2(img)
            output = (output + output2) / 2
            pred = list(output.argmax(dim=1).squeeze().detach().cpu().numpy())
        val_pred += pred
        acc = (output.argmax(dim=-1) == label).float().sum()
        val_acc.append(acc)
        total_len += int(img.shape[0])

    validation_acc = sum(val_acc) / total_len 
    print(f'[ Validation ] acc = {validation_acc:.5f}')


    df = pd.DataFrame()
    ids = sorted([x.split('/')[-1] for x in glob.glob(os.path.join(cfg.test_dir, '*.png'))])
    df['filename'] = np.array(ids)
    df['label'] = val_pred
    df.to_csv(cfg.csv_dir, index=False)
    
