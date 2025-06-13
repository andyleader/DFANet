


config_pavia = dict(
        type='FreeNet',
        config=dict(
            in_channels=103,
            num_classes=9,
            block_channels=(96, 128, 192, 256),
            inner_dim=128,
            reduction_ratio=1.0,
            num_blocks=(1, 1, 1, 1)
        )
    )

config_trento = dict(
        type='FreeNet',
        config=dict(
            img_size=(176, 608),#(166,600)->(176,600)
            in_channels=63,
            in_channels_lidar=1,
            change_lidar_channels=12,
            num_classes=6,
            layer_num=4,
            block_channels=(64, 64, 128, 128),
            # block_channels=(96, 96, 128, 128),
            lidar_block_channels=(4, 4, 8, 8),
            # lidar_block_channels=(64, 64, 64, 64),
            inner_dim=128,
            reduction_ratio=1.0,
            num_blocks=(1, 1, 1, 1),

            # patch_config
            middle_dim = 96
        )
    )

config_muufl = dict(
        type='FreeNet',
        config=dict(
            img_size=(336, 224),#(325,220)->(328,224)
            in_channels=64,
            in_channels_lidar=2,
            change_lidar_channels = 12,
            num_classes=11,
            layer_num=4,
            block_channels=(64, 64, 128, 128),
            # block_channels=(96, 96, 128, 128),
            lidar_block_channels=(4, 4, 8, 8),
            # lidar_block_channels=(64, 64, 64, 64),
            inner_dim=128,
            reduction_ratio=1.0,
            num_blocks=(1, 1, 1, 1),

            # patch_config
            middle_dim = 96
        )
    )

config_houston2013 = dict(
        type='FreeNet',
        config=dict(
            img_size=(352, 1920),#(349,1905)->(352,1920)
            # img_size=(352, 448),#(349,1905)->(224,448)
            in_channels=144,
            in_channels_lidar=1,
            change_lidar_channels=12,
            num_classes=15,
            layer_num=4,
            # block_channels=(96, 128, 192, 256),
            block_channels=(64, 64, 128, 128),
            # block_channels=(96, 96, 128, 128),
            lidar_block_channels=(8, 8, 16, 16),
            # lidar_block_channels=(16, 16, 32, 32),
            inner_dim=128,
            reduction_ratio=1.0,
            num_blocks=(1, 1, 1, 1),

            # patch_config
            middle_dim = 96
        )
    )

config_houston2018 = dict(
        type='FreeNet',
        config=dict(
            img_size=(608, 2384),#(601,2384)->(608,2384)
            # img_size=(608, 2048),#(601,2384)->(608, 896)
            # img_size=(448, 448),
            in_channels=48,
            in_channels_lidar=3,
            change_lidar_channels=12,
            num_classes=20,
            layer_num=4,
            # block_channels=(96, 128, 192, 256),
            block_channels=(64, 64, 128, 128),
            # block_channels=(96, 96, 128, 128),
            lidar_block_channels=(8, 8, 16, 16),
            # lidar_block_channels=(16, 16, 32, 32),
            inner_dim=128,
            reduction_ratio=1.0,
            num_blocks=(1, 1, 1, 1),

            # patch_config
            middle_dim = 96
        )
    )
