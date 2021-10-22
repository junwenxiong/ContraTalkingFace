sudo /usr/local/anaconda3/envs/tf_1/bin/python train_contrastive_loss.py \
	--data_root  lrs2_preprocessed/ \
	--checkpoint_dir /data_8T/xjw/DeepFake/1017_contrastive_loss/ \
	--syncnet_checkpoint_path checkpoints/lipsync_expert.pth 
	# --checkpoint_path trained_checkpoints/1012_original/checkpoint_step000114000.pth \
	# --disc_checkpoint_path trained_checkpoints/1012_original/disc_checkpoint_step000114000.pth
