# audio_convTasnet

recombination of audio_contasnet
### 将audio项目里的语音分离项目，重新组合了下
还没写run.sh
目前就先用这行来直接运行吧
```java
------------------------------------------------------
step1：创建conda环境并进入该虚拟环境
conda create -n zzypython3.8 python=3.8
conda activate zzypython3.8

step2:下载pytorch环境
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

step3:进入到conv-tasnet项目
cd audio_convTasnet

step4:启动
wsj0-mix版本：
python train.py --dataset wsj0mix --root_dir /home/zzy/data/min/ --epochs 160 --batch_size 16 --resume /home/zzy/audio_convTasnet/exp/model/epoch_0.pt --tensorboard_dir /home/zzy/audio_convTasnet/exp/tensorboard

或者

Librimix版本：
python train.py --dataset librimix --root_dir /home/zzy/data/libri/Libri2Mix/wav8k/min/ --librimix-tr-split train-100 --epochs 120 --batch_size 16  --tensorboard_dir /home/zzy/audio_convTasnet/exp/tensorboard
在训练完成后，可以用以下命令进行loss可视化
python -m tensorboard.main --logdir exp/tensorboard/
```
数据可选wsj0 也可以选librimix ，目前我主要用的是wsj0

这个readme就算v1吧，还没完工
