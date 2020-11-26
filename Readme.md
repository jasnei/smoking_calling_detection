### Smoking calling demo

#### 1、copy your images to => data/Images

#### 2、copy xml file to => data/Annotations

```
images file name and annotations file name must same
```

#### 3、run python voc_label.py

```
cd yolov5_calling_smoking
python voc_label.py
```

#### 4、run train.py

```
python train.py --batch 16 --data data/smoke.yaml --cfg models/yolov5l.yaml --weights weights/yolov5l.pt --epochs 100

Note：
--batch is batch size, default is 16
--data  config your data
--cfg   config for yolov5
--weights weights of yolov5
--epochs 
```

#### 5. run detect.py for demo

```
python detect.py --source 0 --weights runs/exp2052/weights/best.pt

Note:
--source 0 will load your 0 camera for video detect, if you pass a path of images, then will detect all the images 
-- weights load weigths of the model
-- output  when you pass an output, then all detect images will be save to the output
```



