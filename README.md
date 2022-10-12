# ShuffleNet

ShuffleNet in Facial Landmark Task

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

