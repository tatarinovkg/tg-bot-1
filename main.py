import subprocess
import asyncio
from config import bot
from aiogram import Dispatcher

from app.handlers import router

dp = Dispatcher()

async def main():
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == '__main__':
    subprocess.run(["python", "sql.py"])
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Exit')
