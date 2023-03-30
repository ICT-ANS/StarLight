CONFIG = {
    'skip_module': ['skip', 'none'],  # must have skip in 1st place
    'normopact_module': {
        'norm': ['skip', 'BN', 'GN', 'LN'],  # must have skip in 1st place
        'bread_op': ['skip', 'dwise_3x3', 'conv_1x1', 'avgpool', 'self_atten', 'spatial_mlp'],  # must have skip in 1st place, self_atten and spatial_mlp must in the end
        'meat_op': ['skip', 'dwise_3x3'],  # must have skip in 1st place, self_atten and spatial_mlp must in the end
        'act': ['skip', 'relu6', 'gelu', 'silu'],  # must have skip in 1st place
        'has_atten_mlp': [False, False, True, True],  # whether search atten & mlp in one stage
    },
    'depth': {
        'stage-1': [1, 2, 3, 4],
        'stage-2': [1, 2, 3, 4],
        'stage-3': [1, 2, 3, 4, 5, 6, 7, 8],
        'stage-4': [1, 2, 3, 4],
    },
    'width': {
        'stage-1': [32, 64, 96],
        'stage-2': [64, 96, 128],
        'stage-3': [128, 192, 256, 320],  # 384, 448
        'stage-4': [128, 256, 384, 512],  # 640, 768
    },
    'ratio': {
        'stage-1': [1, 2, 4, 6],
        'stage-2': [1, 2, 4, 6],
        'stage-3': [1, 2, 4, 6],
        'stage-4': [1, 2, 4, 6],
    }
}
