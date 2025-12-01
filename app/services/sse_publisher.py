from pytune_configuration.redis_config import get_redis_client
import json

CHANNEL = "pytune:diagnosis_events"

async def publish_event(event: dict):
    redis = await get_redis_client()
    await redis.publish(CHANNEL, json.dumps(event))