import numpy as np
import  cv2, torch
from os.path import abspath, dirname, join
from engine.file_utils import save_layer_img

# from engine.layer_result_generators import get_outputs_generator

def save_layer_outputs(model, hooks, graph, layer_name, input_folder, input_name, out_folder, use_gpu, image_size):

    img_cv = cv2.imread(join(abspath(input_folder), input_name))
    img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    height, width = img.shape[:2]

    if image_size is not None:
        width = image_size[-1]
        height = image_size[-2]
    
    if 'KeTi2Tracking' in input_folder: # special codes for KeTi2 Tracking
        img = cv2.resize(img, (width, height))

        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        img_tensor = torch.randn(2, 1, 3, width, height)

        img_tensor = img_tensor.cuda()
        outputs = model(img_tensor)
    elif 'KeTi1DepthEst' in input_folder:
        from algorithms.compression.nets.CFNet.datasets.zjlab_dataset_quiver import ZjlabDataset
        data_name = 'second_round_front/navi_cam/left_origin/1517157368.373382.png second_round_front/navi_cam/right_origin/1517157368.373382.png second_round_front/tof/origin/1517157368.373382.tif second_round_front/tof/origin/1517157368.373382.tif'
        splits = data_name.split()
        prefix = '/demo_502_dataset'

        dataset = ZjlabDataset(
            datapath=input_folder+prefix,
            list_filename=None,
            training=False
        )
        sample = dataset.getitem_from_splits(splits)
        imgL, imgR, disp_sparse, sparse_mask = sample['left'].cuda(), sample['right'].cuda(), \
            sample['sparse'].cuda(), sample['sparse_mask'].cuda()
        for p in model.parameters(): # 防止节点显示不全
            p.requires_grad = False
        with torch.no_grad():
            outputs = model(imgL, imgR, disp_sparse, sparse_mask)[0][0]
    elif 'KeTi2Location' in input_folder:
        from algorithms.compression.nets.Hsmnet.dataloader.listfiles import dataloader
        from algorithms.compression.nets.Hsmnet.dataloader.preprocess import get_transform
        from _init_paths import C
        from skimage import io
        from torch.autograd import Variable

        test_data_path = os.path.join(C.cache_dir, 'dataset/KeTi2Location/test_data_part')
        test_left_img, test_right_img, left_gt, _ = dataloader(test_data_path)
        # 读左右rgb及gt
        inx = 0
        imgL_o = (io.imread(test_left_img[inx]).astype('float32'))[:, :, :3]
        imgR_o = (io.imread(test_right_img[inx]).astype('float32'))[:, :, :3]
        # 归一化及正则化
        processed = get_transform()
        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()
        # 重排列到[N,C,H,W]
        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])
        ##fast pad 补到64的倍数
        max_h = int(imgL.shape[2] // 64 * 64)
        max_w = int(imgL.shape[3] // 64 * 64)
        if max_h < imgL.shape[2]:
            max_h += 64
        if max_w < imgL.shape[3]:
            max_w += 64
        top_pad = max_h - imgL.shape[2]
        left_pad = max_w - imgL.shape[3]
        imgL = np.lib.pad(imgL, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
        imgR = np.lib.pad(imgR, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
        # val
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        imgL = Variable(torch.FloatTensor(imgL).to(DEVICE))
        imgR = Variable(torch.FloatTensor(imgR).to(DEVICE))
        for p in model.parameters():  # 防止节点显示不全
            p.requires_grad = False
        with torch.no_grad():
            outputs = model(imgL, imgR)[0]
    else:
        img = cv2.resize(img, (width, height))

        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        img_tensor = torch.tensor(img, dtype=torch.float32)

        if use_gpu and torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        else:
            img_tensor = img_tensor.cpu()

        outputs = model(img_tensor)

    layers = graph["config"]["layers"]
    layer_id = None
    for layer in layers:
        if layer["name"] == layer_name:
            config =  layer["config"]
            if config !="None" and "layer_id" in config:
                layer_id = config["layer_id"]
                break
    
    results = []
    if layer_id != None:
        for hook in hooks:
            if hook.layer_id == layer_id:
                channel = np.shape(hook.output)[1]
                max_channel = min([channel, channel])
                for channel in range(max_channel):
                    filename = save_layer_img(hook.output[0,channel,:,:], layer_name, channel, out_folder, input_name)
                    results.append(filename)
                break
    
    return results
