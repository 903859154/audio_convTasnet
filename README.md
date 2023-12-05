# audio_convTasnet

recombination of audio_contasnet
### 将audio项目里的语音分离项目，重新组合了下
还没写run.sh
目前就先用这行来直接运行吧
```java
python audio_convTasnet/train.py --dataset wsj0mix --root_dir /home/zzy/data/min/ --batch_size 16 --resume /home/zzy/audio_convTasnet/exp/model/epoch_159.pt 

------------------------------------------------------

python train.py --dataset wsj0mix --root_dir /home/zzy/data/min/ --epochs 160 --batch_size 16 --resume /home/zzy/audio_convTasnet/exp/model/epoch
```
数据可选wsj0 也可以选librimix ，目前我主要用的是wsj0

这个readme就算v1吧，还没完工
