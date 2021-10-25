"""Aiohttp asyncio task many api call, post and receive data asynchronously.

Also use time at start and end of a function call to measure time taken between blocking calls. [Not shown]
"""

import aiohttp
import asyncio
import time


results = []

start = time.time()


class ApiUtils():
    apikey = '...'
    authtoken = '...'

    @classmethod
    def get_api_request_dict_list(cls):
        api_request_dict_list = [
            {
                'endpoint': 'GetTotal',
                'url': '',
                'body': {
                    "apikey": cls.apikey,
                    "authtoken": cls.authtoken,
                }
            },
            {
                'endpoint': 'GetActivity',
                'url': '',
                'body': {
                    "apikey": cls.apikey,
                    "authtoken": cls.authtoken,
                    "StartDate": "8/24/2021",
                    "EndDate": "8/25/2021"
                }
            },
        ] * 30 # Send N duplicate requests to check if there is delay
        return api_request_dict_list


def get_tasks(session):
    tasks = []
    for api_data in ApiUtils.get_api_request_dict_list():
        tasks.append(asyncio.create_task(session.post(api_data['url'], data=api_data['body'])))
    return tasks


async def get_api_responses():
    async with aiohttp.ClientSession() as session:
        tasks = get_tasks(session)
        responses = await asyncio.gather(*tasks)
        for response in responses:
            results.append(await response.json())


print('blocking 1')

asyncio.run(get_api_responses())
print(results)

print('blocking 2')
print('?' * 60)

asyncio.run(get_api_responses())
print(results)

print('blocking 3')
print('#' * 60)

asyncio.run(get_api_responses())
print(results)


total_time = time.time() - start
print(f"{total_time} seconds needed for {len(ApiUtils.get_api_request_dict_list())} API calls")
