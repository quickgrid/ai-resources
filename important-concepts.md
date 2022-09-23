# Important Concepts

Some useful links for understanding various concepts.

### Programming

| Concept | Links |
| --- | --- |
| **CMake build system** <br> Tag: `CMakeLists.txt` | https://www.jetbrains.com/help/clion/cmakelists-txt-file.html |
|  | https://www.jetbrains.com/help/clion/quick-cmake-tutorial.html#new-project |
|  |  |
| **Python source code protection** <br> Tag: `Python Compilation`, `Cython` | https://medium.com/@xpl/protecting-python-sources-using-cython-dcd940bb188e |
| | https://stackoverflow.com/questions/62388701/are-executables-produced-with-cython-really-free-of-the-source-code |
|  |  |
| **Static libraies vs Dynamic/shared libraries** <br> Tag: `Static Linking vs Dynmaic Linking`, `*.so, *.dll vs *.a, *.lib files` | https://medium.com/@StueyGK/static-libraries-vs-dynamic-libraries-af78f0b5f1e4 |
|  | https://stackoverflow.com/questions/2649334/difference-between-static-and-shared-libraries |
|  | https://stackoverflow.com/questions/30186256/what-is-the-difference-between-o-a-and-so-files |
|  |  |


### NLP

| Concept | Links |
| --- | --- |
| **Byte Pair Encoding (BPE)** <br> Tag: `Data Compression`, `Word vs Character vs Subword Tokenization` | https://en.wikipedia.org/wiki/Byte_pair_encoding |
|  | https://www.freecodecamp.org/news/evolution-of-tokenization/ |
|  |  |

<br>

# Commands

- In linux use `df -h` to get information like disk size, free space, used space, mounted on etc.

- In windows `nvidia-smi` can be found under, `C:\Program Files\NVIDIA Corporation\NVSMI`. In `colab` or `aws` jupyter notebooks add `!` before the command below. 
  ```
  nvidia-smi
  ```
  Specific details in CSV format,
  ```
  nvidia-smi --format=csv --query-gpu=power.draw,utilization.gpu,fan.speed,temperature.gpu,memory.used,memory.free
  ```

- Get file types/extensions in given directory,
  ```
  !find "my_dir/dataset/train" -type f -name '*.*' | sed 's|.*\.||' | sort -u
  ```
  
- Move all files in a directory to another folder/directory,
  ```
  !mv my_dir/dataset/train/* another_dir/dataset/train/*
  ```

- Get file and folder size in given directory,
  ```
  du -h "custom_dir/another_dir"
  ```
  
- Running Tensorboard locally on brower on Windows,
  ```
  tensorboard --logdir=C:\DeepLearning\GAN\runs --host localhost --port 8088
  ```
  Running above command on chosen conda enviroment on conda prompt will start Tensorboard on port 8088. It can be accessed from browser on, `http://localhost:8088`.
  Here, I have used absolute path. Reference, https://stackoverflow.com/questions/40106949/unable-to-open-tensorboard-in-browser.


### Example move processed zip file from kaggle to google colab and then from colab move to google drive.

#### From Kaggle

```
from IPython.display import FileLink
FileLink(r'processed_file.zip')
```

This will generate a link, 

```
https://....kaggle.net/...../processed_file.zip
```


#### From Colab

```
!wget "https://....kaggle.net/...../processed_file.zip"
```


#### Mount Google Drive

```
from google.colab import drive
drive.mount('/content/drive')
```

#### Move to Google Drive

```
!cp "/content/processed_file.zip" "/content/drive/My Drive/workspace"
```
