"""WebSocket endpoints for real-time inference streaming and live dashboard updates."""
import time
import json
import logging
from typing import Set, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("apex_analytics.websocket")

router = APIRouter(tags=["Real-Time WebSockets"])


class ConnectionManager:
    """Tracks active client connections and broadcasts real-time telemetry."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast payload to all connected clients."""
        payload_str = json.dumps(message)
        dead_connections = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(payload_str)
            except Exception:
                dead_connections.add(connection)
        for dead in dead_connections:
            self.active_connections.discard(dead)


manager = ConnectionManager()


@router.websocket("/ws/predictions")
@router.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    """Real-time bi-directional WebSocket stream for live predictions and model audit feeds."""
    await manager.connect(websocket)
    try:
        # Send initial handshake message
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to Apex Real-Time Academic Intelligence Stream",
            "timestamp": time.time(),
            "active_subscribers": len(manager.active_connections),
        })

        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                payload = {"action": data}

            action = payload.get("action", "").lower()

            if action == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": time.time(),
                })
            elif action == "subscribe":
                channel = payload.get("channel", "inferences")
                await websocket.send_json({
                    "type": "subscribed",
                    "channel": channel,
                    "message": f"Subscribed to real-time events for {channel}",
                })
            elif action == "status":
                await websocket.send_json({
                    "type": "status",
                    "active_connections": len(manager.active_connections),
                    "server_status": "healthy",
                    "timestamp": time.time(),
                })
            else:
                await websocket.send_json({
                    "type": "ack",
                    "received": payload,
                    "timestamp": time.time(),
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        manager.disconnect(websocket)

