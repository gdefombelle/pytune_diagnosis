import uvicorn

def run_uvicorn():
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8008,
        reload=False,  # Mettre à True si dev local uniquement
        log_level="info",
    )

if __name__ == "__main__":
    run_uvicorn()

