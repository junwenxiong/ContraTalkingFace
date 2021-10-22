sudo /usr/local/anaconda3/envs/tf_1/bin/python trainWav2Lip.py \
	--data_root  lrs2_preprocessed/ \
	--name 10_22_test \
	--syncnet_wt 0.03 \
	--eval_interval 100 \
	--checkpoint_interval 100 
	# --checkpoint_path /data_8T/xjw/DeepFake/checkpoints/10_19_vgg_with_weight_0.1/checkpoint_step000110000.pth \
	# --disc_checkpoint_path /data_8T/xjw/DeepFake/checkpoints/10_19_vgg_with_weight_0.1/disc_checkpoint_step000110000.pth


