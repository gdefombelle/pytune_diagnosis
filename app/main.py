import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import toml
from pathlib import Path
import os

from .routers.rt_diagnosis import router as rt_diagnosis_router

from simple_logger.logger import get_logger, SimpleLogger
from pytune_configuration.sync_config_singleton import config, SimpleConfig

# 🚀 Importer les routers
from .routers import chat_router

# 📜 Initialisation
if config is None:
    config = SimpleConfig()

# 📦 Lecture de pyproject.toml
pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
pyproject_data = toml.load(pyproject_path)

project_metadata = pyproject_data["tool"]["poetry"]
PROJECT_TITLE = project_metadata.get("name", "pytune_diagnosis")
PROJECT_VERSION = project_metadata.get("version", "0.1.0")
PROJECT_DESCRIPTION = project_metadata.get("description", "PyTune RT Diagnosis")

# 📄 Logger
print("ENV LOG_DIR:", os.getenv("LOG_DIR"))
logger = get_logger("pytune_diagnosis")
logger.info("✅ Logger actif", log_dir=os.getenv("LOG_DIR"))
logger.info("********** STARTING PYTUNE DIAGNOSIS ********")

# 🛡️ Rate Limiting Middleware
from pytune_auth_common.services.rate_middleware import RateLimitMiddleware, RateLimitConfig

try:
    rate_limit_config = RateLimitConfig(
        rate_limit=int(config.RATE_MIDDLEWARE_RATE_LIMIT),
        time_window=int(config.RATE_MIDDLEWARE_TIME_WINDOW),
        block_time=int(config.RATE_MIDDLEWARE_LOCK_TIME),
    )
    logger.info("✅ Rate middleware configuration ready")
except Exception as e:
    logger.critical("❌ Failed to set RateLimit", error=e)
    raise RuntimeError("Failed to set RateLimit") from e

# 🌟 Lifespan
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     try:
#         await logger.asuccess("PYTUNE AI ROUTER READY!")
#         yield
#     except asyncio.CancelledError:
#         await logger.acritical("❌ Lifespan cancelled")
#         raise
#     finally:
#         await logger.asuccess("✅ Lifespan finished without errors")

# 🚀 FastAPI app
app = FastAPI(
    title=PROJECT_TITLE,
    version=PROJECT_VERSION,
    description=PROJECT_DESCRIPTION,
    # lifespan=lifespan,
)

# 🔗 Middleware CORS
allowed_origins = config.ALLOWED_CORS_ORIGINS
logger.info(f"Allowed CORS origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["Authorization"],
)

# 🔗 Middleware Rate Limit
if config.USE_RATE_MIDDLEWARE:
    logger.info("Applying RATE_MIDDLEWARE")
    try:
        app.add_middleware(
            RateLimitMiddleware,
            config=rate_limit_config,
        )
    except Exception as e:
        logger.critical("Erreur lors de l'application des middlewares", error=e)
        raise RuntimeError("Failed to load middlewares") from e
else:
    logger.info("NO RATE_MIDDLEWARE applied")

# 🔗 Inclure les routers
app.include_router(rt_diagnosis_router)

# 📄 Gestion des erreurs FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request
import json
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        raw_body = await request.body()
        try:
            decoded_body = raw_body.decode("utf-8")
        except Exception:
            decoded_body = repr(raw_body)  # ✅ safe

        # ✅ DEBUG : log en console pour dev
        print("❌ Validation error:", exc.errors())
        print("📦 Raw body:", decoded_body)

        return JSONResponse(
            status_code=422,
            content={
                "detail": exc.errors(),
                "body": decoded_body
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": "Exception handler failed", "error": str(e)}
        )

# 📂 Fichiers statiques (optionnel si besoin)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ❤️ Healthcheck route
@app.get("/")
async def health_check():
    return {"status": "ok", "service": PROJECT_TITLE, "version": PROJECT_VERSION}
