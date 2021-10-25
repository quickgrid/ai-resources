"""AWS S3 boto3 upload and download image from memory without writing to file using BytesIO and show.

"""

import io

import boto3
from botocore.exceptions import ClientError
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


access_key = ''
accrss_secret = ''
bucket_name = ''
region_name = ''
where_to_save = 'fox/fox-upload-21.jpg'


client_s3 = boto3.client(
    's3',
    aws_access_key_id=access_key,
    aws_secret_access_key=accrss_secret,
    region_name=region_name
)

resource_s3 = boto3.resource(
    's3',
    aws_access_key_id=access_key,
    aws_secret_access_key=accrss_secret,
    region_name=region_name
)


# Upload image from memory.
try:
    tmp_img = Image.open('upload_images/fox4.jpg')
    fs = io.BytesIO()
    tmp_img.save(fs, 'JPEG')
    fs.seek(0)

    client_s3.put_object(Body=fs, Bucket=bucket_name, Key=where_to_save)
except ClientError as e:
    print('Incorrect credential')
    print(e)
except Exception as e:
    print(e)


# Download the uploaded image to memory and show.
bucket = resource_s3.Bucket(bucket_name)
img_object = bucket.Object(where_to_save)

file_stream = io.BytesIO()
img_object.download_fileobj(file_stream)
img = mpimg.imread(file_stream, format='jpeg')

plt.imshow(img)
plt.show()
