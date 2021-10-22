sudo /usr/local/anaconda3/envs/tf_1/bin/python hq_wav2lip_train.py \
	--data_root  lrs2_preprocessed/ \
	--checkpoint_dir /data_8T/xjw/DeepFake/1018_with_vgg_130000/ \
	--syncnet_checkpoint_path checkpoints/lipsync_expert.pth \
	--checkpoint_path  /data_8T/xjw/DeepFake/1017_with_vgg/checkpoint_step000130000.pth \
	--disc_checkpoint_path /data_8T/xjw/DeepFake/1017_with_vgg/disc_checkpoint_step000130000.pth
