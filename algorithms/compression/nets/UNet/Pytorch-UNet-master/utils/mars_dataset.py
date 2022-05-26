from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging

'''
tips: txt dataLoader for virtual dataset.
'''

class MarsDataset(Dataset):
    def __init__(self, root_dir, scale=1, mask_suffix=''):
        self.root_dir = root_dir
        '''tips: here change input channel, geometry use only elevation and slope'''
        txt_dir_list = [root_dir + 'elevation/',
                            root_dir + 'roughness/',
                            root_dir + 'gap/',
                            root_dir + 'granularity/',
                            root_dir + 'slope/']
        self.txt_dirs = txt_dir_list
        self.label_dir = root_dir + 'traversability/'

        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.ids = [splitext(file)[0] for file in listdir(self.txt_dirs[0])
                    if not file.startswith('.')]
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
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)


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
        for txt_dir in self.txt_dirs:
            txt_file = glob(txt_dir + idx + '.*')
            input_txt_files.append(txt_file)
        assert len(label_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {label_file}'
        for input_txt_file in input_txt_files:
            assert len(input_txt_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {input_txt_file}'
        label = self.channel_loader(label_file[0])
        input_tensor_list = []
        for input_txt_file in input_txt_files:
            input_channel = self.channel_loader(input_txt_file[0])
            input_tensor_list.append(input_channel)
        input_tensor = torch.cat(input_tensor_list, dim=0)
        return {
            'image': input_tensor,
            'label': label
        }
