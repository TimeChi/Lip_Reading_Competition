# 基础操作（修改数据集路径）：

数据集的路径需要更改一下，具体要求：
更改opts.py里面第一行代码 RAW_PATH变量改为数据具体所在的文件路径,要保证下一级文件夹是‘1.训练集’和‘2.测试集’

# 训练:
**（训练）：**
可以直接在opts.py里更改默认参数运行，或直接运行根目录下的train.sh文件 

# 推断预测:
**（推断预测）：**
训练完成后执行infer.sh文件即可得到生成在result文件夹下的sub_2.csv文件

# 权重文件：
仅提供初赛resnet101+tsm的权重文件作为测试 权重文件百度云链接：https://pan.baidu.com/s/1EeVVZ5W6fx0Dbqf6q916qw 密码：fthm

# 运行环境:
硬件：GTX1080Ti或RTX2080Ti双卡环境,CUDA10, cudnn，ubuntu16.04。
软件：conda一个环境，具体见requirement.txt,可能罗列不全，执行时缺什么就装什么

# 说明
因为对代码做了一些删减，不免可能出现问题，如果有任何问题欢迎交流

本代码都是基于TSM: Temporal Shift Module for Efficient Video Understanding
这篇论文的开源代码 https://github.com/mit-han-lab/temporal-shift-module 进行修改，最后感谢一下大佬的贡献。


