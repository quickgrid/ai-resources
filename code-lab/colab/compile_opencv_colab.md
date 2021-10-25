## Source, https://answers.opencv.org/question/233476/how-to-make-opencv-use-gpu-on-google-colab/

```
%cd /content
!git clone https://github.com/opencv/opencv
!git clone https://github.com/opencv/opencv_contrib
!mkdir /content/build
%cd /content/build
```

```
!cmake -DOPENCV_EXTRA_MODULES_PATH=/content/opencv_contrib/modules -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DWITH_OPENEXR=OFF -DWITH_CUDA=ON -DWITH_CUBLAS=ON -DWITH_CUDNN=ON  -DOPENCV_DNN_CUDA=ON  /content/opencv
```

```
!make -j8 install
```

```
import cv2
cv2.__version__
```

## Copy compiled file to drive for reuse

```
!mkdir  "/content/drive/My Drive/cv2_cuda"
!cp  /content/build/lib/python3/cv2.cpython-36m-x86_64-linux-gnu.so   "/content/drive/My Drive/cv2_cuda"
```

## Get it back

```
!cp "/content/drive/My Drive/cv2_cuda/cv2.cpython-36m-x86_64-linux-gnu.so" .
```
