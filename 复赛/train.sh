
python main.py  --mode train --img_size 184\
     --arch resnet101 --num_segments 24 \
     --gd 20 --lr 0.01 --lr_type cos --epochs 330 \
     --batch-size 6 -j 16 --dropout 0.8 \
     --resume ./save_mode/_resnet101_torch.pth.tar\
