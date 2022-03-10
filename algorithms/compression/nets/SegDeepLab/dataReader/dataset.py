import random
import numpy as np
import torch
import torch.utils.data as udata
import glob
import os
import cv2
import sys
sys.path.append('')


class Reader(udata.Dataset):
    def __init__(self, args, mode='train'):
        if (mode != 'train') & (mode != 'eval') & (mode != 'test'):
            raise Exception("Invalid mode!", mode)
        self.mode = mode
        items = []
        if self.mode == 'train':
            img_path = os.path.join(args.data_root, 'Train/image')
            mask_path = os.path.join(args.data_root, 'Train/label')

            # # data_v0
            # img_name = glob.glob(os.path.join(img_path, '*.tif'))
            # mask_name = glob.glob(os.path.join(mask_path, '*.tif'))
            # data_1119
            img_name = glob.glob(os.path.join(img_path, '*.jpg'))
            mask_name = glob.glob(os.path.join(mask_path, '*.png'))
            img_name.sort()
            mask_name.sort()

            for i in range(len(img_name)):  # //1000
                item = (img_name[i], mask_name[i])
                items.append(item)
        elif self.mode == 'eval':
            img_path = os.path.join(args.data_root, 'Test/image')
            mask_path = os.path.join(args.data_root, 'Test/label')

            img_name = glob.glob(os.path.join(img_path, '*.jpg'))
            mask_name = glob.glob(os.path.join(mask_path, '*.png'))
            img_name.sort()
            mask_name.sort()

            for i in range(len(img_name)):
                item = (img_name[i], mask_name[i])
                items.append(item)
        else:
            img_path = os.path.join(args.data_root, 'Valid/image')
            mask_path = os.path.join(args.data_root, 'Valid/label')
            # # data_v0
            # img_name = glob.glob(os.path.join(img_path, '*.tif'))
            # mask_name = glob.glob(os.path.join(mask_path, '*.tif'))
            # data_1119
            img_name = glob.glob(os.path.join(img_path, '*.jpg'))
            mask_name = glob.glob(os.path.join(mask_path, '*.png'))

            img_name.sort()
            mask_name.sort()
            for i in range(len(img_name)):
                item = (img_name[i], mask_name[i])
                items.append(item)
        self.keys = items
        if self.mode == 'train':
            random.shuffle(self.keys)
        else:
            self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        img = cv2.imread(self.keys[index][0])
        label = cv2.imread(self.keys[index][1], cv2.IMREAD_GRAYSCALE)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.float32)
        img = np.transpose(img, [2, 0, 1]) / 255.0
        img = torch.Tensor(img)

        label = torch.Tensor(label).long()

        return img, label


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
    parser.add_argument('--data_root', default='./dataset/Mars_Seg_1119/Data', type=str, help='dataset path')
    args = parser.parse_args()

    from torch.utils.data import DataLoader

    train_data = Reader(args, mode='train')
    print("Train set samples: ", len(train_data))
    val_data = Reader(args, mode='test')
    print("Validation set samples: ", len(val_data))

    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True, num_workers=1,
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=2, shuffle=False, num_workers=1,
                            pin_memory=False, drop_last=True)
    for i, (images, labels) in enumerate(val_loader):
        labels = labels
        images = images
        print(np.unique(labels))


