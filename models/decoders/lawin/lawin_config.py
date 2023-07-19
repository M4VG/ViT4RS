lawin_config = dict(

	b0 = dict(
		in_channels=[32, 64, 160, 256],
		in_index=[0, 1, 2, 3],
		# channels=128,
		dropout_ratio=0.1,
		num_classes=2,
		norm_cfg=dict(type='BN', requires_grad=True),
		align_corners=False,
		concat_fuse=True,
		depth=1,
		decoder_params=dict(embed_dim=128, in_dim=128, reduction=1, proj_type='conv', use_scale=True, mixing=True),
		input_transform='multiple_select',
		ignore_index=255
	),

	b1 = dict(
		in_channels=[64, 128, 320, 512],
		in_index=[0, 1, 2, 3],
		# channels=128,
		dropout_ratio=0.1,
		num_classes=2,
		norm_cfg=dict(type='BN', requires_grad=True),
		align_corners=False,
		concat_fuse=False,
		depth=1,
		decoder_params=dict(embed_dim=128, in_dim=128, reduction=1, proj_type='conv', use_scale=True, mixing=False),
		input_transform='multiple_select',
		ignore_index=255
	),

	b2 = dict(
		in_channels=[64, 128, 320, 512],
		in_index=[0, 1, 2, 3],
		# channels=128,
		dropout_ratio=0.1,
		num_classes=2,
		norm_cfg=dict(type='BN', requires_grad=True),
		align_corners=False,
		concat_fuse=True,
		depth=1,
		decoder_params=dict(embed_dim=512, in_dim=512, reduction=2, proj_type='conv', use_scale=True, mixing=True),
		input_transform='multiple_select',
		ignore_index=255
	),

	b3 = dict(
		in_channels=[64, 128, 320, 512],
		in_index=[0, 1, 2, 3],
		# channels=128,
		dropout_ratio=0.1,
		num_classes=2,
		norm_cfg=dict(type='BN', requires_grad=True),
		align_corners=False,
		concat_fuse=True,
		depth=1,
		decoder_params=dict(embed_dim=512, in_dim=512, reduction=2, proj_type='conv', use_scale=True, mixing=True),
		input_transform='multiple_select',
		ignore_index=255
	),

	b4 = dict(
		in_channels=[64, 128, 320, 512],
		in_index=[0, 1, 2, 3],
		# channels=128,
		dropout_ratio=0.1,
		num_classes=2,
		norm_cfg=dict(type='BN', requires_grad=True),
		align_corners=False,
		concat_fuse=True,
		depth=1,
		decoder_params=dict(embed_dim=768, in_dim=512, reduction=2, proj_type='conv', use_scale=True, mixing=True),
		input_transform='multiple_select',
		ignore_index=255
	),

	b5 = dict(
		in_channels=[64, 128, 320, 512],
		in_index=[0, 1, 2, 3],
		# channels=128,
		dropout_ratio=0.1,
		num_classes=2,
		norm_cfg=dict(type='BN', requires_grad=True),
		align_corners=False,
		concat_fuse=True,
		depth=1,
		decoder_params=dict(embed_dim=512, in_dim=512, reduction=2, proj_type='conv', use_scale=True, mixing=True),
		input_transform='multiple_select',
		ignore_index=255
	)
)
