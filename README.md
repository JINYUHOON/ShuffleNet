# ShuffleNet

### ShuffleNet in Facial Landmark Task

```
python train.py --epoch --image_size --batch_size --data_path --label_count --learning_rate

optional arguments:
--epoch                   training epoch number
--image_size              input image_size
--batch_size              training image batch_size
--data_path               data path ( csv data / columns : image_path , coordinates, fold )
--label_count             output label count ( facial landmark )
--learning_rate           optimizer lr
```

데이터 로더에서 csv 파일로 받으며 이미지 경로와 GroundTruth 좌표(x,y) , Data split을 위한 fold로 구성하여 학습 진행

