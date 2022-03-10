import network
import torch
import torch.nn as nn

if __name__ == '__main__':
    model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=19, output_stride=16)
    model.eval()

    prune_name_list = []
    last_name = None
    for name, module, in model.named_modules():
        # if ('conv' in name and 'static_padding' not in name) or ('_se_reduce' in name and 'static_padding' not in name):

        # print in_place dict
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
            # print(name, module)
            if last_name is None:
                print('\'%s\': (%s, ),' % (name, last_name))
            else:
                print('\'%s\': (\'%s\', ),' % (name, last_name))
            if isinstance(module, nn.Conv2d):
                last_name = name

        # # print op_names dict
        # if isinstance(module, nn.Conv2d):
        #     prune_name_list.append(name)

        # if 'conv' in name or 'bn' in name:
        # print(name, module)
        # if 'aux' not in name and name != '' and '.' in name:
        #     if 'conv' in name or 'Conv2d' in str(module):
        # prune_name_list.append(name)
    print(prune_name_list)
    print(len(prune_name_list))
    # print(model)

# 'op_names': ['backbone.conv1', 'backbone.layer1.0.conv1', 'backbone.layer1.0.conv2',
#                                          'backbone.layer1.0.conv3', 'backbone.layer1.0.downsample.0',
#                                          'backbone.layer1.1.conv1', 'backbone.layer1.1.conv2',
#                                          'backbone.layer1.1.conv3', 'backbone.layer1.2.conv1',
#                                          'backbone.layer1.2.conv2', 'backbone.layer1.2.conv3',
#                                          'backbone.layer2.0.conv1', 'backbone.layer2.0.conv2',
#                                          'backbone.layer2.0.conv3', 'backbone.layer2.0.downsample.0',
#                                          'backbone.layer2.1.conv1', 'backbone.layer2.1.conv2',
#                                          'backbone.layer2.1.conv3', 'backbone.layer2.2.conv1',
#                                          'backbone.layer2.2.conv2', 'backbone.layer2.2.conv3',
#                                          'backbone.layer2.3.conv1', 'backbone.layer2.3.conv2',
#                                          'backbone.layer2.3.conv3', 'backbone.layer3.0.conv1',
#                                          'backbone.layer3.0.conv2', 'backbone.layer3.0.conv3',
#                                          'backbone.layer3.0.downsample.0', 'backbone.layer3.1.conv1',
#                                          'backbone.layer3.1.conv2', 'backbone.layer3.1.conv3',
#                                          'backbone.layer3.2.conv1', 'backbone.layer3.2.conv2',
#                                          'backbone.layer3.2.conv3', 'backbone.layer3.3.conv1',
#                                          'backbone.layer3.3.conv2', 'backbone.layer3.3.conv3',
#                                          'backbone.layer3.4.conv1', 'backbone.layer3.4.conv2',
#                                          'backbone.layer3.4.conv3', 'backbone.layer3.5.conv1',
#                                          'backbone.layer3.5.conv2', 'backbone.layer3.5.conv3',
#                                          'backbone.layer3.6.conv1', 'backbone.layer3.6.conv2',
#                                          'backbone.layer3.6.conv3', 'backbone.layer3.7.conv1',
#                                          'backbone.layer3.7.conv2', 'backbone.layer3.7.conv3',
#                                          'backbone.layer3.8.conv1', 'backbone.layer3.8.conv2',
#                                          'backbone.layer3.8.conv3', 'backbone.layer3.9.conv1',
#                                          'backbone.layer3.9.conv2', 'backbone.layer3.9.conv3',
#                                          'backbone.layer3.10.conv1', 'backbone.layer3.10.conv2',
#                                          'backbone.layer3.10.conv3', 'backbone.layer3.11.conv1',
#                                          'backbone.layer3.11.conv2', 'backbone.layer3.11.conv3',
#                                          'backbone.layer3.12.conv1', 'backbone.layer3.12.conv2',
#                                          'backbone.layer3.12.conv3', 'backbone.layer3.13.conv1',
#                                          'backbone.layer3.13.conv2', 'backbone.layer3.13.conv3',
#                                          'backbone.layer3.14.conv1', 'backbone.layer3.14.conv2',
#                                          'backbone.layer3.14.conv3', 'backbone.layer3.15.conv1',
#                                          'backbone.layer3.15.conv2', 'backbone.layer3.15.conv3',
#                                          'backbone.layer3.16.conv1', 'backbone.layer3.16.conv2',
#                                          'backbone.layer3.16.conv3', 'backbone.layer3.17.conv1',
#                                          'backbone.layer3.17.conv2', 'backbone.layer3.17.conv3',
#                                          'backbone.layer3.18.conv1', 'backbone.layer3.18.conv2',
#                                          'backbone.layer3.18.conv3', 'backbone.layer3.19.conv1',
#                                          'backbone.layer3.19.conv2', 'backbone.layer3.19.conv3',
#                                          'backbone.layer3.20.conv1', 'backbone.layer3.20.conv2',
#                                          'backbone.layer3.20.conv3', 'backbone.layer3.21.conv1',
#                                          'backbone.layer3.21.conv2', 'backbone.layer3.21.conv3',
#                                          'backbone.layer3.22.conv1', 'backbone.layer3.22.conv2',
#                                          'backbone.layer3.22.conv3', 'backbone.layer4.0.conv1',
#                                          'backbone.layer4.0.conv2', 'backbone.layer4.0.conv3',
#                                          'backbone.layer4.0.downsample.0', 'backbone.layer4.1.conv1',
#                                          'backbone.layer4.1.conv2', 'backbone.layer4.1.conv3',
#                                          'backbone.layer4.2.conv1', 'backbone.layer4.2.conv2',
#                                          'backbone.layer4.2.conv3', 'classifier.project.0',
#                                          'classifier.aspp.convs.0.0', 'classifier.aspp.convs.1.0',
#                                          'classifier.aspp.convs.2.0', 'classifier.aspp.convs.3.0',
#                                          'classifier.aspp.convs.4.1', 'classifier.aspp.project.0',
#                                          'classifier.classifier.0']
