"""Read files asynchronously and upload to a specific folder with given name to S3 using aioboto3.

"""

import os

import asyncio
import aioboto3
import aiofiles


access_key = '...'
accrss_secret = '...'
bucket_name = '...'
region_name = '...'


image_upload_folder = os.path.join(os.getcwd(), 'upload_images')


async def upload():
    session = aioboto3.Session()
    async with session.client(
        service_name="s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=accrss_secret,
        region_name=region_name
    ) as s3:
        try:
            file_path = os.path.join(image_upload_folder, 'fox23456.jpg')
            print(file_path)
            async with aiofiles.open(file_path, mode='rb') as f:
                await s3.upload_fileobj(f, bucket_name, 'fox/fox-wild.jpg')
        except Exception as e:
            print(e)
            return ""

    return 'WORK DONE'


loop = asyncio.get_event_loop()
loop.run_until_complete(upload())
