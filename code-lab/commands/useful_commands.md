## Commands

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
