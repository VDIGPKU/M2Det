# Only m2det_configs are supported
# For debug when configuring m2det.py file
# you can modify them in m2det_config in ./configs/m2detxxx.py

m2det_configs = dict(
    vgg16 = dict(
        backbone = 'vgg16',
        net_family = 'vgg',
        base_out = [22,34], # [22,34] for vgg, [2,4] or [3,4] for res families
        planes = 256,
        num_levels = 8,
        num_scales = 6,
        sfam = True,
        smooth = True,
        num_classes = 81,
        ),
    resnet50 = dict(
        backbone = 'resnet50',
        net_family = 'res',
        base_out = [2,4], # [22,34] for vgg, [2,4] or [3,4] for res families
        planes = 256,
        num_levels = 8,
        num_scales = 6,
        sfam = True,
        smooth = True,
        num_classes = 81,
        )
    )

