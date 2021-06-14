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
--uncertainty-training 1






