in_replace_dic = {
    'efficientnet_features._conv_stem.conv': (None,),
    'efficientnet_features._bn0': ('efficientnet_features._conv_stem.conv',),
    'efficientnet_features._blocks.0._depthwise_conv.conv': ('efficientnet_features._conv_stem.conv',),
    # layer  0
    'efficientnet_features._blocks.0._bn1': ('efficientnet_features._blocks.0._depthwise_conv.conv',),
    'efficientnet_features._blocks.0._se_reduce.conv': ('efficientnet_features._blocks.0._depthwise_conv.conv',),
    'efficientnet_features._blocks.0._se_expand.conv': ('efficientnet_features._blocks.0._se_reduce.conv',),
    'efficientnet_features._blocks.0._project_conv.conv': ('efficientnet_features._blocks.0._depthwise_conv.conv',),
    'efficientnet_features._blocks.0._bn2': ('efficientnet_features._blocks.0._project_conv.conv',),
    'efficientnet_features._blocks.1._depthwise_conv.conv': ('efficientnet_features._blocks.0._project_conv.conv',),
    # layer  1
    'efficientnet_features._blocks.1._bn1': ('efficientnet_features._blocks.1._depthwise_conv.conv',),
    'efficientnet_features._blocks.1._se_reduce.conv': ('efficientnet_features._blocks.1._depthwise_conv.conv',),
    'efficientnet_features._blocks.1._se_expand.conv': ('efficientnet_features._blocks.1._se_reduce.conv',),
    'efficientnet_features._blocks.1._project_conv.conv': ('efficientnet_features._blocks.1._depthwise_conv.conv',),
    'efficientnet_features._blocks.1._bn2': ('efficientnet_features._blocks.1._project_conv.conv',),
    'efficientnet_features._blocks.2._expand_conv.conv': ('efficientnet_features._blocks.1._project_conv.conv',),
    # layer  2
    'efficientnet_features._blocks.2._bn0': ('efficientnet_features._blocks.2._expand_conv.conv',),
    'efficientnet_features._blocks.2._depthwise_conv.conv': ('efficientnet_features._blocks.2._expand_conv.conv',),
    'efficientnet_features._blocks.2._bn1': ('efficientnet_features._blocks.2._depthwise_conv.conv',),
    'efficientnet_features._blocks.2._se_reduce.conv': ('efficientnet_features._blocks.2._depthwise_conv.conv',),
    'efficientnet_features._blocks.2._se_expand.conv': ('efficientnet_features._blocks.2._se_reduce.conv',),
    'efficientnet_features._blocks.2._project_conv.conv': ('efficientnet_features._blocks.2._depthwise_conv.conv',),
    'efficientnet_features._blocks.2._bn2': ('efficientnet_features._blocks.2._project_conv.conv',),
    'efficientnet_features._blocks.3._expand_conv.conv': ('efficientnet_features._blocks.2._project_conv.conv',),
    # layer  3
    'efficientnet_features._blocks.3._bn0': ('efficientnet_features._blocks.3._expand_conv.conv',),
    'efficientnet_features._blocks.3._depthwise_conv.conv': ('efficientnet_features._blocks.3._expand_conv.conv',),
    'efficientnet_features._blocks.3._bn1': ('efficientnet_features._blocks.3._depthwise_conv.conv',),
    'efficientnet_features._blocks.3._se_reduce.conv': ('efficientnet_features._blocks.3._depthwise_conv.conv',),
    'efficientnet_features._blocks.3._se_expand.conv': ('efficientnet_features._blocks.3._se_reduce.conv',),
    'efficientnet_features._blocks.3._project_conv.conv': ('efficientnet_features._blocks.3._depthwise_conv.conv',),
    'efficientnet_features._blocks.3._bn2': ('efficientnet_features._blocks.3._project_conv.conv',),
    'efficientnet_features._blocks.4._expand_conv.conv': ('efficientnet_features._blocks.3._project_conv.conv',),
    # layer  4
    'efficientnet_features._blocks.4._bn0': ('efficientnet_features._blocks.4._expand_conv.conv',),
    'efficientnet_features._blocks.4._depthwise_conv.conv': ('efficientnet_features._blocks.4._expand_conv.conv',),
    'efficientnet_features._blocks.4._bn1': ('efficientnet_features._blocks.4._depthwise_conv.conv',),
    'efficientnet_features._blocks.4._se_reduce.conv': ('efficientnet_features._blocks.4._depthwise_conv.conv',),
    'efficientnet_features._blocks.4._se_expand.conv': ('efficientnet_features._blocks.4._se_reduce.conv',),
    'efficientnet_features._blocks.4._project_conv.conv': ('efficientnet_features._blocks.4._depthwise_conv.conv',),
    'efficientnet_features._blocks.4._bn2': ('efficientnet_features._blocks.4._project_conv.conv',),
    'efficientnet_features._blocks.5._expand_conv.conv': ('efficientnet_features._blocks.4._project_conv.conv',),
    # layer  5
    'efficientnet_features._blocks.5._bn0': ('efficientnet_features._blocks.5._expand_conv.conv',),
    'efficientnet_features._blocks.5._depthwise_conv.conv': ('efficientnet_features._blocks.5._expand_conv.conv',),
    'efficientnet_features._blocks.5._bn1': ('efficientnet_features._blocks.5._depthwise_conv.conv',),
    'efficientnet_features._blocks.5._se_reduce.conv': ('efficientnet_features._blocks.5._depthwise_conv.conv',),
    'efficientnet_features._blocks.5._se_expand.conv': ('efficientnet_features._blocks.5._se_reduce.conv',),
    'efficientnet_features._blocks.5._project_conv.conv': ('efficientnet_features._blocks.5._depthwise_conv.conv',),
    'efficientnet_features._blocks.5._bn2': ('efficientnet_features._blocks.5._project_conv.conv',),
    'efficientnet_features._blocks.6._expand_conv.conv': ('efficientnet_features._blocks.5._project_conv.conv',),
    # layer  6
    'efficientnet_features._blocks.6._bn0': ('efficientnet_features._blocks.6._expand_conv.conv',),
    'efficientnet_features._blocks.6._depthwise_conv.conv': ('efficientnet_features._blocks.6._expand_conv.conv',),
    'efficientnet_features._blocks.6._bn1': ('efficientnet_features._blocks.6._depthwise_conv.conv',),
    'efficientnet_features._blocks.6._se_reduce.conv': ('efficientnet_features._blocks.6._depthwise_conv.conv',),
    'efficientnet_features._blocks.6._se_expand.conv': ('efficientnet_features._blocks.6._se_reduce.conv',),
    'efficientnet_features._blocks.6._project_conv.conv': ('efficientnet_features._blocks.6._depthwise_conv.conv',),
    'efficientnet_features._blocks.6._bn2': ('efficientnet_features._blocks.6._project_conv.conv',),
    'efficientnet_features._blocks.7._expand_conv.conv': ('efficientnet_features._blocks.6._project_conv.conv',),
    # layer  7
    'efficientnet_features._blocks.7._bn0': ('efficientnet_features._blocks.7._expand_conv.conv',),
    'efficientnet_features._blocks.7._depthwise_conv.conv': ('efficientnet_features._blocks.7._expand_conv.conv',),
    'efficientnet_features._blocks.7._bn1': ('efficientnet_features._blocks.7._depthwise_conv.conv',),
    'efficientnet_features._blocks.7._se_reduce.conv': ('efficientnet_features._blocks.7._depthwise_conv.conv',),
    'efficientnet_features._blocks.7._se_expand.conv': ('efficientnet_features._blocks.7._se_reduce.conv',),
    'efficientnet_features._blocks.7._project_conv.conv': ('efficientnet_features._blocks.7._depthwise_conv.conv',),
    'efficientnet_features._blocks.7._bn2': ('efficientnet_features._blocks.7._project_conv.conv',),
    'efficientnet_features._blocks.8._expand_conv.conv': ('efficientnet_features._blocks.7._project_conv.conv',),
    # layer  8
    'efficientnet_features._blocks.8._bn0': ('efficientnet_features._blocks.8._expand_conv.conv',),
    'efficientnet_features._blocks.8._depthwise_conv.conv': ('efficientnet_features._blocks.8._expand_conv.conv',),
    'efficientnet_features._blocks.8._bn1': ('efficientnet_features._blocks.8._depthwise_conv.conv',),
    'efficientnet_features._blocks.8._se_reduce.conv': ('efficientnet_features._blocks.8._depthwise_conv.conv',),
    'efficientnet_features._blocks.8._se_expand.conv': ('efficientnet_features._blocks.8._se_reduce.conv',),
    'efficientnet_features._blocks.8._project_conv.conv': ('efficientnet_features._blocks.8._depthwise_conv.conv',),
    'efficientnet_features._blocks.8._bn2': ('efficientnet_features._blocks.8._project_conv.conv',),
    'efficientnet_features._blocks.9._expand_conv.conv': ('efficientnet_features._blocks.8._project_conv.conv',),
    # layer  9
    'efficientnet_features._blocks.9._bn0': ('efficientnet_features._blocks.9._expand_conv.conv',),
    'efficientnet_features._blocks.9._depthwise_conv.conv': ('efficientnet_features._blocks.9._expand_conv.conv',),
    'efficientnet_features._blocks.9._bn1': ('efficientnet_features._blocks.9._depthwise_conv.conv',),
    'efficientnet_features._blocks.9._se_reduce.conv': ('efficientnet_features._blocks.9._depthwise_conv.conv',),
    'efficientnet_features._blocks.9._se_expand.conv': ('efficientnet_features._blocks.9._se_reduce.conv',),
    'efficientnet_features._blocks.9._project_conv.conv': ('efficientnet_features._blocks.9._depthwise_conv.conv',),
    'efficientnet_features._blocks.9._bn2': ('efficientnet_features._blocks.9._project_conv.conv',),
    'efficientnet_features._blocks.10._expand_conv.conv': ('efficientnet_features._blocks.9._project_conv.conv',),
    # layer  10
    'efficientnet_features._blocks.10._bn0': ('efficientnet_features._blocks.10._expand_conv.conv',),
    'efficientnet_features._blocks.10._depthwise_conv.conv': ('efficientnet_features._blocks.10._expand_conv.conv',),
    'efficientnet_features._blocks.10._bn1': ('efficientnet_features._blocks.10._depthwise_conv.conv',),
    'efficientnet_features._blocks.10._se_reduce.conv': ('efficientnet_features._blocks.10._depthwise_conv.conv',),
    'efficientnet_features._blocks.10._se_expand.conv': ('efficientnet_features._blocks.10._se_reduce.conv',),
    'efficientnet_features._blocks.10._project_conv.conv': ('efficientnet_features._blocks.10._depthwise_conv.conv',),
    'efficientnet_features._blocks.10._bn2': ('efficientnet_features._blocks.10._project_conv.conv',),
    'efficientnet_features._blocks.11._expand_conv.conv': ('efficientnet_features._blocks.10._project_conv.conv',),
    # layer  11
    'efficientnet_features._blocks.11._bn0': ('efficientnet_features._blocks.11._expand_conv.conv',),
    'efficientnet_features._blocks.11._depthwise_conv.conv': ('efficientnet_features._blocks.11._expand_conv.conv',),
    'efficientnet_features._blocks.11._bn1': ('efficientnet_features._blocks.11._depthwise_conv.conv',),
    'efficientnet_features._blocks.11._se_reduce.conv': ('efficientnet_features._blocks.11._depthwise_conv.conv',),
    'efficientnet_features._blocks.11._se_expand.conv': ('efficientnet_features._blocks.11._se_reduce.conv',),
    'efficientnet_features._blocks.11._project_conv.conv': ('efficientnet_features._blocks.11._depthwise_conv.conv',),
    'efficientnet_features._blocks.11._bn2': ('efficientnet_features._blocks.11._project_conv.conv',),
    'efficientnet_features._blocks.12._expand_conv.conv': ('efficientnet_features._blocks.11._project_conv.conv',),
    # layer  12
    'efficientnet_features._blocks.12._bn0': ('efficientnet_features._blocks.12._expand_conv.conv',),
    'efficientnet_features._blocks.12._depthwise_conv.conv': ('efficientnet_features._blocks.12._expand_conv.conv',),
    'efficientnet_features._blocks.12._bn1': ('efficientnet_features._blocks.12._depthwise_conv.conv',),
    'efficientnet_features._blocks.12._se_reduce.conv': ('efficientnet_features._blocks.12._depthwise_conv.conv',),
    'efficientnet_features._blocks.12._se_expand.conv': ('efficientnet_features._blocks.12._se_reduce.conv',),
    'efficientnet_features._blocks.12._project_conv.conv': ('efficientnet_features._blocks.12._depthwise_conv.conv',),
    'efficientnet_features._blocks.12._bn2': ('efficientnet_features._blocks.12._project_conv.conv',),
    'efficientnet_features._blocks.13._expand_conv.conv': ('efficientnet_features._blocks.12._project_conv.conv',),
    # layer  13
    'efficientnet_features._blocks.13._bn0': ('efficientnet_features._blocks.13._expand_conv.conv',),
    'efficientnet_features._blocks.13._depthwise_conv.conv': ('efficientnet_features._blocks.13._expand_conv.conv',),
    'efficientnet_features._blocks.13._bn1': ('efficientnet_features._blocks.13._depthwise_conv.conv',),
    'efficientnet_features._blocks.13._se_reduce.conv': ('efficientnet_features._blocks.13._depthwise_conv.conv',),
    'efficientnet_features._blocks.13._se_expand.conv': ('efficientnet_features._blocks.13._se_reduce.conv',),
    'efficientnet_features._blocks.13._project_conv.conv': ('efficientnet_features._blocks.13._depthwise_conv.conv',),
    'efficientnet_features._blocks.13._bn2': ('efficientnet_features._blocks.13._project_conv.conv',),
    'efficientnet_features._blocks.14._expand_conv.conv': ('efficientnet_features._blocks.13._project_conv.conv',),
    # layer  14
    'efficientnet_features._blocks.14._bn0': ('efficientnet_features._blocks.14._expand_conv.conv',),
    'efficientnet_features._blocks.14._depthwise_conv.conv': ('efficientnet_features._blocks.14._expand_conv.conv',),
    'efficientnet_features._blocks.14._bn1': ('efficientnet_features._blocks.14._depthwise_conv.conv',),
    'efficientnet_features._blocks.14._se_reduce.conv': ('efficientnet_features._blocks.14._depthwise_conv.conv',),
    'efficientnet_features._blocks.14._se_expand.conv': ('efficientnet_features._blocks.14._se_reduce.conv',),
    'efficientnet_features._blocks.14._project_conv.conv': ('efficientnet_features._blocks.14._depthwise_conv.conv',),
    'efficientnet_features._blocks.14._bn2': ('efficientnet_features._blocks.14._project_conv.conv',),
    'efficientnet_features._blocks.15._expand_conv.conv': ('efficientnet_features._blocks.14._project_conv.conv',),
    # layer  15
    'efficientnet_features._blocks.15._bn0': ('efficientnet_features._blocks.15._expand_conv.conv',),
    'efficientnet_features._blocks.15._depthwise_conv.conv': ('efficientnet_features._blocks.15._expand_conv.conv',),
    'efficientnet_features._blocks.15._bn1': ('efficientnet_features._blocks.15._depthwise_conv.conv',),
    'efficientnet_features._blocks.15._se_reduce.conv': ('efficientnet_features._blocks.15._depthwise_conv.conv',),
    'efficientnet_features._blocks.15._se_expand.conv': ('efficientnet_features._blocks.15._se_reduce.conv',),
    'efficientnet_features._blocks.15._project_conv.conv': ('efficientnet_features._blocks.15._depthwise_conv.conv',),
    'efficientnet_features._blocks.15._bn2': ('efficientnet_features._blocks.15._project_conv.conv',),
    'efficientnet_features._blocks.16._expand_conv.conv': ('efficientnet_features._blocks.15._project_conv.conv',),
    # layer  16
    'efficientnet_features._blocks.16._bn0': ('efficientnet_features._blocks.16._expand_conv.conv',),
    'efficientnet_features._blocks.16._depthwise_conv.conv': ('efficientnet_features._blocks.16._expand_conv.conv',),
    'efficientnet_features._blocks.16._bn1': ('efficientnet_features._blocks.16._depthwise_conv.conv',),
    'efficientnet_features._blocks.16._se_reduce.conv': ('efficientnet_features._blocks.16._depthwise_conv.conv',),
    'efficientnet_features._blocks.16._se_expand.conv': ('efficientnet_features._blocks.16._se_reduce.conv',),
    'efficientnet_features._blocks.16._project_conv.conv': ('efficientnet_features._blocks.16._depthwise_conv.conv',),
    'efficientnet_features._blocks.16._bn2': ('efficientnet_features._blocks.16._project_conv.conv',),
    'efficientnet_features._blocks.17._expand_conv.conv': ('efficientnet_features._blocks.16._project_conv.conv',),
    # layer  17
    'efficientnet_features._blocks.17._bn0': ('efficientnet_features._blocks.17._expand_conv.conv',),
    'efficientnet_features._blocks.17._depthwise_conv.conv': ('efficientnet_features._blocks.17._expand_conv.conv',),
    'efficientnet_features._blocks.17._bn1': ('efficientnet_features._blocks.17._depthwise_conv.conv',),
    'efficientnet_features._blocks.17._se_reduce.conv': ('efficientnet_features._blocks.17._depthwise_conv.conv',),
    'efficientnet_features._blocks.17._se_expand.conv': ('efficientnet_features._blocks.17._se_reduce.conv',),
    'efficientnet_features._blocks.17._project_conv.conv': ('efficientnet_features._blocks.17._depthwise_conv.conv',),
    'efficientnet_features._blocks.17._bn2': ('efficientnet_features._blocks.17._project_conv.conv',),
    'efficientnet_features._blocks.18._expand_conv.conv': ('efficientnet_features._blocks.17._project_conv.conv',),
    # layer  18
    'efficientnet_features._blocks.18._bn0': ('efficientnet_features._blocks.18._expand_conv.conv',),
    'efficientnet_features._blocks.18._depthwise_conv.conv': ('efficientnet_features._blocks.18._expand_conv.conv',),
    'efficientnet_features._blocks.18._bn1': ('efficientnet_features._blocks.18._depthwise_conv.conv',),
    'efficientnet_features._blocks.18._se_reduce.conv': ('efficientnet_features._blocks.18._depthwise_conv.conv',),
    'efficientnet_features._blocks.18._se_expand.conv': ('efficientnet_features._blocks.18._se_reduce.conv',),
    'efficientnet_features._blocks.18._project_conv.conv': ('efficientnet_features._blocks.18._depthwise_conv.conv',),
    'efficientnet_features._blocks.18._bn2': ('efficientnet_features._blocks.18._project_conv.conv',),
    'efficientnet_features._blocks.19._expand_conv.conv': ('efficientnet_features._blocks.18._project_conv.conv',),
    # layer  19
    'efficientnet_features._blocks.19._bn0': ('efficientnet_features._blocks.19._expand_conv.conv',),
    'efficientnet_features._blocks.19._depthwise_conv.conv': ('efficientnet_features._blocks.19._expand_conv.conv',),
    'efficientnet_features._blocks.19._bn1': ('efficientnet_features._blocks.19._depthwise_conv.conv',),
    'efficientnet_features._blocks.19._se_reduce.conv': ('efficientnet_features._blocks.19._depthwise_conv.conv',),
    'efficientnet_features._blocks.19._se_expand.conv': ('efficientnet_features._blocks.19._se_reduce.conv',),
    'efficientnet_features._blocks.19._project_conv.conv': ('efficientnet_features._blocks.19._depthwise_conv.conv',),
    'efficientnet_features._blocks.19._bn2': ('efficientnet_features._blocks.19._project_conv.conv',),
    'efficientnet_features._blocks.20._expand_conv.conv': ('efficientnet_features._blocks.19._project_conv.conv',),
    # layer  20
    'efficientnet_features._blocks.20._bn0': ('efficientnet_features._blocks.20._expand_conv.conv',),
    'efficientnet_features._blocks.20._depthwise_conv.conv': ('efficientnet_features._blocks.20._expand_conv.conv',),
    'efficientnet_features._blocks.20._bn1': ('efficientnet_features._blocks.20._depthwise_conv.conv',),
    'efficientnet_features._blocks.20._se_reduce.conv': ('efficientnet_features._blocks.20._depthwise_conv.conv',),
    'efficientnet_features._blocks.20._se_expand.conv': ('efficientnet_features._blocks.20._se_reduce.conv',),
    'efficientnet_features._blocks.20._project_conv.conv': ('efficientnet_features._blocks.20._depthwise_conv.conv',),
    'efficientnet_features._blocks.20._bn2': ('efficientnet_features._blocks.20._project_conv.conv',),
    'efficientnet_features._blocks.21._expand_conv.conv': ('efficientnet_features._blocks.20._project_conv.conv',),
    # layer  21
    'efficientnet_features._blocks.21._bn0': ('efficientnet_features._blocks.21._expand_conv.conv',),
    'efficientnet_features._blocks.21._depthwise_conv.conv': ('efficientnet_features._blocks.21._expand_conv.conv',),
    'efficientnet_features._blocks.21._bn1': ('efficientnet_features._blocks.21._depthwise_conv.conv',),
    'efficientnet_features._blocks.21._se_reduce.conv': ('efficientnet_features._blocks.21._depthwise_conv.conv',),
    'efficientnet_features._blocks.21._se_expand.conv': ('efficientnet_features._blocks.21._se_reduce.conv',),
    'efficientnet_features._blocks.21._project_conv.conv': ('efficientnet_features._blocks.21._depthwise_conv.conv',),
    'efficientnet_features._blocks.21._bn2': ('efficientnet_features._blocks.21._project_conv.conv',),
    'efficientnet_features._blocks.22._expand_conv.conv': ('efficientnet_features._blocks.21._project_conv.conv',),
    # layer  22
    'efficientnet_features._blocks.22._bn0': ('efficientnet_features._blocks.22._expand_conv.conv',),
    'efficientnet_features._blocks.22._depthwise_conv.conv': ('efficientnet_features._blocks.22._expand_conv.conv',),
    'efficientnet_features._blocks.22._bn1': ('efficientnet_features._blocks.22._depthwise_conv.conv',),
    'efficientnet_features._blocks.22._se_reduce.conv': ('efficientnet_features._blocks.22._depthwise_conv.conv',),
    'efficientnet_features._blocks.22._se_expand.conv': ('efficientnet_features._blocks.22._se_reduce.conv',),
    'efficientnet_features._blocks.22._project_conv.conv': ('efficientnet_features._blocks.22._depthwise_conv.conv',),
    'efficientnet_features._blocks.22._bn2': ('efficientnet_features._blocks.22._project_conv.conv',),
    'efficientnet_features._blocks.23._expand_conv.conv': ('efficientnet_features._blocks.22._project_conv.conv',),
    # layer  23
    'efficientnet_features._blocks.23._bn0': ('efficientnet_features._blocks.23._expand_conv.conv',),
    'efficientnet_features._blocks.23._depthwise_conv.conv': ('efficientnet_features._blocks.23._expand_conv.conv',),
    'efficientnet_features._blocks.23._bn1': ('efficientnet_features._blocks.23._depthwise_conv.conv',),
    'efficientnet_features._blocks.23._se_reduce.conv': ('efficientnet_features._blocks.23._depthwise_conv.conv',),
    'efficientnet_features._blocks.23._se_expand.conv': ('efficientnet_features._blocks.23._se_reduce.conv',),
    'efficientnet_features._blocks.23._project_conv.conv': ('efficientnet_features._blocks.23._depthwise_conv.conv',),
    'efficientnet_features._blocks.23._bn2': ('efficientnet_features._blocks.23._project_conv.conv',),
    'efficientnet_features._blocks.24._expand_conv.conv': ('efficientnet_features._blocks.23._project_conv.conv',),
    # layer  24
    'efficientnet_features._blocks.24._bn0': ('efficientnet_features._blocks.24._expand_conv.conv',),
    'efficientnet_features._blocks.24._depthwise_conv.conv': ('efficientnet_features._blocks.24._expand_conv.conv',),
    'efficientnet_features._blocks.24._bn1': ('efficientnet_features._blocks.24._depthwise_conv.conv',),
    'efficientnet_features._blocks.24._se_reduce.conv': ('efficientnet_features._blocks.24._depthwise_conv.conv',),
    'efficientnet_features._blocks.24._se_expand.conv': ('efficientnet_features._blocks.24._se_reduce.conv',),
    'efficientnet_features._blocks.24._project_conv.conv': ('efficientnet_features._blocks.24._depthwise_conv.conv',),
    'efficientnet_features._blocks.24._bn2': ('efficientnet_features._blocks.24._project_conv.conv',),
    'efficientnet_features._blocks.25._expand_conv.conv': ('efficientnet_features._blocks.24._project_conv.conv',),
    # layer  25
    'efficientnet_features._blocks.25._bn0': ('efficientnet_features._blocks.25._expand_conv.conv',),
    'efficientnet_features._blocks.25._depthwise_conv.conv': ('efficientnet_features._blocks.25._expand_conv.conv',),
    'efficientnet_features._blocks.25._bn1': ('efficientnet_features._blocks.25._depthwise_conv.conv',),
    'efficientnet_features._blocks.25._se_reduce.conv': ('efficientnet_features._blocks.25._depthwise_conv.conv',),
    'efficientnet_features._blocks.25._se_expand.conv': ('efficientnet_features._blocks.25._se_reduce.conv',),
    'efficientnet_features._blocks.25._project_conv.conv': ('efficientnet_features._blocks.25._depthwise_conv.conv',),
    'efficientnet_features._blocks.25._bn2': ('efficientnet_features._blocks.25._project_conv.conv',),
    'efficientnet_features._conv_head.conv': ('efficientnet_features._blocks.25._project_conv.conv',),
    'efficientnet_features._bn1': ('efficientnet_features._conv_head.conv',),
    'last_conv.0': ('efficientnet_features._blocks.0._project_conv.conv', ),
    'last_conv.1': ('last_conv.0', ),
    'last_conv.3': ('last_conv.0', ),
    'last_conv.4': ('last_conv.3', ),
    'last_conv.6': ('last_conv.3', ),
}

if __name__ == '__main__':
    from models.EfficientNet.deeplab_efficiennetb1 import DeepLabv3_plus
    model = DeepLabv3_plus(nInputChannels=3, n_classes=6, os=16, pretrained=False, _print=False)
    last_conv = None
    last_depthwise_conv = None
    last_layer = -1
    for name, _, in model.named_modules():
        if '.conv' in name or '_bn' in name:
            if len(name.split('.')) > 3:
                layer = int(name.split('.')[2])
                if layer != last_layer:
                    print('# layer ', layer)
                    last_layer = layer

            if '_project_conv' in name:
                print('\'%s\': (\'%s\', ),' % (name, last_depthwise_conv))
            else:
                if last_conv is None:
                    print('\'%s\': (%s, ),' % (name, last_conv))
                else:
                    print('\'%s\': (\'%s\', ),' % (name, last_conv))
            if '.conv' in name:
                last_conv = name
            if '_depthwise_conv' in name:
                last_depthwise_conv = name

