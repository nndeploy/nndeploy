# log_broadcaster.py

import asyncio
import logging
import re
from typing import Optional

class LogBroadcaster(logging.Handler):
    def __init__(self, get_loop, get_ws_map):
        super().__init__()
        self.get_loop = get_loop
        self.get_ws_map = get_ws_map
        self.task_log_map = {}

    def emit(self, record):
        try:
            if not self.formatter:
                self.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
            msg = self.format(record)
            task_id = self._extract_task_id(msg)
            if not task_id:
                return

            self.task_log_map.setdefault(task_id, []).append(msg)
            if len(self.task_log_map[task_id]) > 1000:
                self.task_log_map[task_id] = self.task_log_map[task_id][-1000:]

            loop = self.get_loop()
            ws_map = self.get_ws_map()
            if task_id in ws_map and loop and loop.is_running():
                for ws in ws_map[task_id].copy():
                    asyncio.run_coroutine_threadsafe(
                        self._send_log(ws, task_id, msg), loop
                    )
        except Exception as e:
            logging.warning("[LogBroadcaster] emit error: %s", e)

    async def _send_log(self, ws, task_id: str, msg: str):
        payload = {
            "flag": "success",
            "message": "log update",
            "result": {
                "task_id": task_id,
                "type": "log",
                "log": msg,
            }
        }
        try:
            await ws.send_json(payload)
        except Exception as e:
            logging.warning("[LogBroadcaster] send error: %s", e)

    def _extract_task_id(self, msg: str) -> Optional[str]:
            match = re.search(r'\[task_id=([a-f0-9\-]+)\]', msg)
            if match:
                return match.group(1)
            return None

    def get_logs(self, task_id: str) -> list[str]:
        return self.task_log_map.get(task_id, [])
