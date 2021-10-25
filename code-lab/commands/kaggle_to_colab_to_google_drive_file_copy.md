Example move processed zip file from kaggle to google colab and then from colab move to google drive.

## From Kaggle

```
from IPython.display import FileLink
FileLink(r'processed_file.zip')
```

This will generate a link, 

```
https://....kaggle.net/...../processed_file.zip
```


## From Colab

```
!wget "https://....kaggle.net/...../processed_file.zip"
```


## Mount Google Drive

```
from google.colab import drive
drive.mount('/content/drive')
```

## Move to Google Drive

```
!cp "/content/processed_file.zip" "/content/drive/My Drive/workspace"
```
