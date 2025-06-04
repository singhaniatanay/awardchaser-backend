from fastapi import FastAPI
import importlib
import pkgutil
from pathlib import Path

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}

def load_routers(package: str) -> None:
    """Dynamically import all modules from *package* and include their `router` if present."""
    try:
        module = importlib.import_module(package)
    except ModuleNotFoundError:
        return

    package_path = Path(module.__file__).parent
    for finder, name, ispkg in pkgutil.iter_modules([str(package_path)]):
        full_name = f"{package}.{name}"
        if ispkg:
            load_routers(full_name)
        else:
            mod = importlib.import_module(full_name)
            router = getattr(mod, "router", None)
            if router is not None:
                app.include_router(router)


# Autoload routers from app.api.v1
load_routers("app.api.v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
