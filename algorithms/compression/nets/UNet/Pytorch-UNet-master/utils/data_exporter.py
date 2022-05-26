import os
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd

'''
tips: runtime data exporter for .event file
'''

def search_event(path):
    # 用于获取path目录下所有的event文件的数据,需要注意的有以下几点: (ps:通过pytorch_lightning调用的tensorboard)
    # 1. 没有通过'/'分层的全在path目录下的那个events文件中,通过'/'分层的数据,即使是通过tb.add_scalars()API添加的,也是一个scalar一个文件夹
    # 2. 通过'/'分层的数据,单个scalar独享一个文件夹,且只有该文件夹下最大的event文件中的数据是正常的；另外,该event文件中scalar的名字是(例:'树干/树支')干上的名字:
    # 例如:train_loss_dict = {"loss": loss, "ae_loss": loss_ae, "recon_loss": loss_recon, "liear_loss": loss_linear}
    #     tensorboard.add_scalars("train/loss", train_loss_dict, self.global_step)
    # 则 event 文件中的scalar的名字为:'train/loss',所有上面四个文件夹中的event文件里的scalar的名字都是'train/loss'
    dir_name = []
    file_name = []
    file_size = []
    event_num = 0       # 用于计数当前path目录下(不包括子目录)的event文件个数
    for filename in os.listdir(path):
        # print(filename)
        temp_path = os.path.join(path, filename)
        if os.path.isdir(temp_path):
            # 如果是文件夹的话，除了根目录，就是通过"/"add的scalar的文件夹了
            # 如果是文件夹，则递归地取出文件夹下有效的events文件
            temp_file_name = search_event(temp_path)
            # 将该文件夹下有效的events文件路径添加到list中
            file_name.extend(temp_file_name)
        elif os.path.isfile(temp_path):
            # 如果文件名中包含有'tfevents'字符串，则认为其是一个events文件
            if 'tfevents' in temp_path:
                event_num += 1
                # 记录该目录下的events文件数量、路径、文件尺寸
                file_name.append(temp_path)
                file_size.append(os.path.getsize(temp_path))
    if event_num > 1:
        # 如果当前目录下的event文件个数>1,则取size最大的那个
        index = file_size.index(max(file_size))
        temp_file_path = file_name[index]
        if isinstance(temp_file_path, str):
            temp_file_path = [temp_file_path]
        return temp_file_path
    return file_name

def readEvent(event_path):
    '''返回tensorboard生成的event文件中所有的scalar的值和名字
            event_path:event文件路径
    '''
    event = event_accumulator.EventAccumulator(event_path)
    event.Reload()
    print("\033[1;34m数据标签：\033[0m")
    print(event.Tags())
    print("\033[1;34m标量数据关键词：\033[0m")
    # print(event.scalars.Keys())
    scalar_name = []
    scalar_data = []
    for name in event.scalars.Keys():
        print(name)
        if 'hp_metric' not in name:
            scalar_name.append(name)
            # event.scalars.Items(name)返回的是list,每个元素为ScalarEvent,有wall_time,step(即我们add_scalar时的step)，value（该scalar在step时的值）
            scalar_data.append(event.scalars.Items(name))
    return scalar_name, scalar_data

def exportToexcel(file_name, excelName):
    '''
        将不同的标量数据导入到同一个excel中，放置在不同的sheet下
            注：excel中sheet名称的命名不能有：/\?*这些符号
    '''
    writer = pd.ExcelWriter(excelName)
    for i in range(len(file_name)):
        event_path = file_name[i]
        scalar_name, scalar_data = readEvent(event_path)
        for i in np.arange(len(scalar_name)):
            scalarValue = scalar_data[i]
            scalarName = scalar_name[i]
            if "/" in scalarName:
                temp_names = scalar_name[i].split("/")
                # temp_paths = os.path.split(event_path)
                scalarName = temp_names[0]
            data = pd.DataFrame(scalarValue)
            data.to_excel(writer, sheet_name=scalarName)
    writer.save()
    print("数据保存成功")

if __name__ == "__main__":
    # mode: 0--从events中提取数据到excel,1--从excel读取数据到array
    eve_path = '../runs/Apr10_23-43-41_DESKTOP-TI0TGUKLR_0.0001_BS_2_ep_100_cl_5_data_2340_test_world1_remap_NoNoise_naive'
    file_name = search_event(eve_path)
    excelName = eve_path +'/test.xlsx'
    # 保存到同一目录下
    exportToexcel(file_name, excelName)
