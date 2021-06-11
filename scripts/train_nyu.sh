DATA_ROOT=/cluster/scratch/semilk
TRAIN_SET=$DATA_ROOT/NYU/rectified_nyu/
python train.py $TRAIN_SET \
--folder-type pair \
--resnet-layers 18 \
--num-scales 1 \
-b16 -s0 -c0 --epoch-size 0 --epochs 50 \
--lr 1e-4 \
--with-ssim 1 \
--with-mask 0 \
--with-auto-mask 0 \
--with-pretrain 1 \
--log-output --with-gt \
--dataset nyu \
--name r18_rectified_nyu \
--uncertainty-training 1 \
--pretrained-disp /cluster/home/semilk/3DVision/git/uncertainty_training/SC-SfMLearner-Release/checkpoints/r18_rectified_nyu/06-06-22:22/dispnet_checkpoint.pth.tar \
--pretrained-pose /cluster/home/semilk/3DVision/git/uncertainty_training/SC-SfMLearner-Release/checkpoints/r18_rectified_nyu/06-06-22:22/exp_pose_checkpoint.pth.tar






