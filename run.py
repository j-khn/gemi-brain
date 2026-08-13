"""Entry point — run with: python run.py  or  uvicorn app.main:app"""

import uvicorn

from main import HOST, PORT, RELOAD

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=RELOAD)
