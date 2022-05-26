from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
'''
tips: img dataLoader for actual dataset.
'''


class MarsImgDataset(Dataset):
    # root_dir:the url of the dataset
    def __init__(self, root_dir, scale=1, mask_suffix=''):
        self.root_dir = root_dir
        img_dir_list = [
            root_dir + 'ortho_slope/',
            # root_dir + 'ortho_mosaic/',
            root_dir + 'ortho_depth/',
            root_dir + 'ortho_rough/'
        ]

        self.channel_dirs = img_dir_list

        self.label_dir = root_dir + 'ortho_traver/'

        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.ids = [splitext(file)[0] for file in listdir(self.channel_dirs[0]) if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def channel_loader(cls, file):
        with open(file) as f:
            file_lines = f.readlines()
        data_list = []
        for line in file_lines:
            line.strip()
            line = (line.replace("\n", " ")).split(" ")[0:-1]
            line = np.array(line).astype(np.float)
            read_line = line.tolist()
            data_list.append(read_line)

        channel = torch.from_numpy(np.array(data_list)).type(torch.FloatTensor).unsqueeze(0)
        return channel

    @classmethod
    def preprocess(cls, pil_img, scale, istraver=0):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if istraver == 1:
            img_nd = img_nd > 128 + 0
            img_nd = img_nd * 255

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans[0]

    def __getitem__(self, i):
        idx = self.ids[i]
        label_file = glob(self.label_dir + idx + self.mask_suffix + '.*')
        input_txt_files = []
        for channel_dir in self.channel_dirs:
            txt_file = glob(channel_dir + idx + '.*')
            input_txt_files.append(txt_file)

        assert len(label_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {label_file}'
        for input_txt_file in input_txt_files:
            assert len(input_txt_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {input_txt_file}'
        # assert len(slope_file) == 1, \
        #     f'Either no image or multiple images found for the ID {idx}: {slope_file}'
        label = Image.open(label_file[0])
        label_process = self.preprocess(label, self.scale)
        label_tensor = torch.from_numpy(label_process).type(torch.FloatTensor).unsqueeze(0)
        input_tensor_list = []
        for input_txt_file in input_txt_files:
            # input_channel = self.channel_loader(input_txt_file[0])
            input_channel = Image.open(input_txt_file[0])
            # assert input_channel.size == label.size, \
            #     f'Image and mask {idx} should be the same size, but are {input_channel.size} and {label.size}'
            input_channel = self.preprocess(input_channel, self.scale)
            input_channel_tensor = torch.from_numpy(input_channel).type(torch.FloatTensor).unsqueeze(0)
            input_tensor_list.append(input_channel_tensor)
        input_tensor = torch.cat(input_tensor_list, dim=0)
        return {'image': input_tensor, 'label': label_tensor}
