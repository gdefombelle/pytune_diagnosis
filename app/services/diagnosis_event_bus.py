# app/services/diagnosis_event_bus.py
import asyncio
import json

# Mode simple : queue locale (process unique)
diagnosis_queue = asyncio.Queue()


async def publish_event(event: dict):
    """Called by WS analysis to broadcast an event."""
    await diagnosis_queue.put(event)


async def consume_events():
    """Used by SSE to stream events."""
    while True:
        data = await diagnosis_queue.get()
        yield f"data: {json.dumps(data)}\n\n"