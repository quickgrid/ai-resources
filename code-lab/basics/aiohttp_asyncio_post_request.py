"""An example of making post request using aiohttp and asyncio to get some data.

References:
  https://docs.aiohttp.org/en/stable/client_quickstart.html#make-a-request
"""

import aiohttp
import asyncio


api_url = 'http://sample.example.com/api/SomethingDemo/GetSomethingTotal'
json_body_dict = {
    "apikey": "abcde1243590469",
    "authtoken": "qwertyu12345678-12324354-243",
    "some_param": 123
}


async def main():
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, data=json_body_dict) as resp:
            print(resp.status)
            print(await resp.text())


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
