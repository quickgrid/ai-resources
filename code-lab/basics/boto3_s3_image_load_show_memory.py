"""Boto3 S3 image file stream into memory to show via matplotlib without writing.

"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
import boto3

access_key = ''
accrss_secret = ''
bucket_name = ''
region_name = ''


client_s3 = boto3.resource(
    's3',
    aws_access_key_id=access_key,
    aws_secret_access_key=accrss_secret,
    region_name=region_name
)

bucket = client_s3.Bucket(bucket_name)
img_object = bucket.Object('fox/fox-wild-22.jpg')

file_stream = io.BytesIO()
img_object.download_fileobj(file_stream)
img = mpimg.imread(file_stream, format='jpeg')

plt.imshow(img)
plt.show()
