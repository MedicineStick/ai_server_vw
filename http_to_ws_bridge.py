from fastapi import FastAPI, Request
from pydantic import BaseModel
import websockets
import asyncio
import json

app = FastAPI()

WS_SERVER_URL = "ws://localhost:9501/ws"  # <-- Your actual WebSocket server URL

class Payload(BaseModel):
    message: str

@app.post("/api/send")
async def send_to_websocket(payload: Payload):
    try:
        async with websockets.connect(WS_SERVER_URL) as websocket:
            await websocket.send(payload.message)
            response = await websocket.recv()
            return response
    except Exception as e:
        return response
