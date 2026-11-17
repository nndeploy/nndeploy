"""Call both sample algorithms using the official MCP Python client."""
import asyncio
import json
import os

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main():
    headers = {}
    if os.environ.get('NNDEPLOY_MCP_TOKEN'):
        headers['Authorization'] = 'Bearer ' + os.environ['NNDEPLOY_MCP_TOKEN']
    async with httpx.AsyncClient(headers=headers) as http:
        async with streamable_http_client(os.environ.get('NNDEPLOY_MCP_URL', 'http://127.0.0.1:8000/mcp'),
                                          http_client=http) as (read, write, _):
            async with ClientSession(read, write) as client:
                await client.initialize()
                print('Tools:', [tool.name for tool in (await client.list_tools()).tools])
                for tool in ('uppercase_text', 'count_characters'):
                    accepted = await client.call_tool(tool, {'inputs': {'text': 'Hello nndeploy'}})
                    if accepted.isError:
                        raise RuntimeError(accepted.content)
                    task_id = accepted.structuredContent['task_id']
                    async with asyncio.timeout(60):
                        while True:
                            result = await client.call_tool('get_task', {'task_id': task_id})
                            if result.isError:
                                raise RuntimeError(result.content)
                            record = result.structuredContent
                            if record['state'] in ('succeeded', 'failed', 'cancelled', 'expired'):
                                print(tool, json.dumps(record, ensure_ascii=False))
                                break
                            await asyncio.sleep(.2)


if __name__ == '__main__':
    asyncio.run(main())
