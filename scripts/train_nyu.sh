DATA_ROOT=/cluster/scratch/semilk
TRAIN_SET=$DATA_ROOT/NYU/rectified_nyu/
python train.py $TRAIN_SET \
--folder-type pair \
--resnet-layers 18 \
--num-scales 1 \
-b16 -s0.2 -c0.1 --epoch-size 0 --epochs 50 \
--lr 3e-5 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output --with-gt \
--dataset nyu \
--name r18_rectified_nyu \
--uncertainty-training 1






