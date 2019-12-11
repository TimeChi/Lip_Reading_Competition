python main.py  --mode test --sub sub\
     --img_size 184 --arch resnet101 --num_segments 24 \
     --gd 20 --lr 0.01 --lr_type cos --epochs 300 \
     --batch-size 12 -j 8 --dropout 0.5 \
     --resume ./save_mode/_resnet101_torch.pth.tar
