from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from pytune_llm.task_reporting.task_pubsub import get_queue
import asyncio

router = APIRouter(prefix="/diag")

