
##  Show file size date time name
```
!ls -laSh
```

**Sample ouptut**
```
total 428M
-rw-r--r-- 1 root root 245M Oct 29 15:02 yolov4-416.tflite
-rw-r--r-- 1 root root 123M Oct 29 15:47 yolov4-416-fp16.tflite
-rw-r--r-- 1 root root  62M Oct 29 16:07 yolov4-416-int8.tflite
drwxr-xr-x 3 root root 4.0K Oct 29 16:07 .
drwxr-xr-x 9 root root 4.0K Oct 29 16:06 ..
drwxr-xr-x 4 root root 4.0K Oct 29 15:01 yolov4-416
```


## Show files in directory with size
```
!du -a "/content/path"
```

**Sample ouptut**
```
62800	/content/tensorflow-yolov4-tflite/checkpoints/yolov4-416-int8.tflite
4	/content/tensorflow-yolov4-tflite/checkpoints/yolov4-416/assets
36	/content/tensorflow-yolov4-tflite/checkpoints/yolov4-416/variables/variables.index
250312	/content/tensorflow-yolov4-tflite/checkpoints/yolov4-416/variables/variables.data-00000-of-00001
250352	/content/tensorflow-yolov4-tflite/checkpoints/yolov4-416/variables
11568	/content/tensorflow-yolov4-tflite/checkpoints/yolov4-416/saved_model.pb
261928	/content/tensorflow-yolov4-tflite/checkpoints/yolov4-416
125152	/content/tensorflow-yolov4-tflite/checkpoints/yolov4-416-fp16.tflite
249908	/content/tensorflow-yolov4-tflite/checkpoints/yolov4-416.tflite
699792	/content/tensorflow-yolov4-tflite/checkpoints
```
