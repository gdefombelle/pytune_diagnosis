from pytune_configuration.redis_config import get_redis_client
import json
import numpy as np

CHANNEL = "pytune:events"

def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if obj is Ellipsis:
        return "..."

    if isinstance(obj, set):
        return list(obj)

    return str(obj)

async def publish_event(event: dict):
    redis = await get_redis_client()

    try:
        payload = json.dumps(event, default=_json_default, allow_nan=True)
        print("📤 REDIS PUBLISH", CHANNEL, payload[:500])
    except Exception as e:
        print("❌ JSON DUMPS FAILED:", e)
        print("EVENT RAW:", repr(event))
        raise

    result = await redis.publish(CHANNEL, payload)
    print("✅ REDIS PUBLISH RESULT =", result)