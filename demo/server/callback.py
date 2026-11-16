"""Run a local callback receiver: python demo/server/callback.py"""
import json

from fastapi import FastAPI, Response
import uvicorn

app = FastAPI()


@app.post('/callback')
async def callback(event: dict):
    # Production receivers should deduplicate by task_id before side effects.
    print(json.dumps(event, ensure_ascii=False, indent=2), flush=True)
    return Response(status_code=204)


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=9000)
