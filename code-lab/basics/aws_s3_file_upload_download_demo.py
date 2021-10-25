"""AWS S3 file upload sample.

References:
  https://www.youtube.com/watch?v=JmrYZPjSDl4
"""

import os
import boto3
from botocore.exceptions import ClientError


access_key = ''
accrss_secret = ''
bucket_name = ''
region_name = ''


client_s3 = boto3.client(
    's3',
    aws_access_key_id=access_key,
    aws_secret_access_key=accrss_secret,
    region_name=region_name
)


image_upload_folder = os.path.join(os.getcwd(), 'upload_images')
for f in os.listdir(image_upload_folder):
    try:
        client_s3.upload_file(os.path.join(image_upload_folder, f), bucket_name, f)
    except ClientError as e:
        print('Incorrect credential')
        print(e)
    except Exception as e:
        print(e)

        
image_upload_folder = os.path.join(os.getcwd(), 'download_images')
client_s3.download_file(bucket_name, 'fox/fox-wild-11.jpg', os.path.join(image_upload_folder, 'fox-down.jpg'))        
