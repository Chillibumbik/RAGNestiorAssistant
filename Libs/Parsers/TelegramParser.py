from telethon import TelegramClient

from dotenv import load_dotenv
import os

from langchain.schema import Document

from pathlib import Path

from tqdm import tqdm

env_path = Path(__file__).resolve().parents[2] / "Config.env"
load_dotenv(dotenv_path=env_path)


api_id = int(os.getenv('TEL_API_ID'))
api_hash = os.getenv('TEL_API_HASH')
phone = os.getenv('TEL_ACC_NUMBER')


async def get_messages_from_chats(client: TelegramClient, chats: list[str],
                                  messages_limit: int = 1000) -> list[Document]:
    """
    chats - список чатов @username или id
    messages_limit - кол-во сообщений

    Возвращает список документов
    !!! Важно. Список докуметов имеет тип не только text. При загрузке в БД нужно фильтровать
    """
    docs = []
    for chat in chats:
        async for msg in tqdm(client.iter_messages(chat, limit=messages_limit)):
            text = msg.text if msg.text is not None else ""
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": msg.sender_id,
                        "source_path": chat,
                        "message_id": msg.id,
                        "date": str(msg.date),
                        "type": str(msg.media.__class__.__name__) if msg.media else "text"
                    }
                )
            )
    return docs

def parse_telegram_chats(chats: list, messages_limit: int = 1000) -> list[Document]:
    """
    Обёртка: создаёт клиента, получает сообщения, возвращает список Document.
    """
    with TelegramClient("session_name", api_id, api_hash) as client:
        docs = client.loop.run_until_complete(
            get_messages_from_chats(client, chats=chats, messages_limit=messages_limit)
        )
    return docs

# тест
if __name__ == "__main__":
    res = parse_telegram_chats(["https://t.me/+p3EpGIYGVcJlNmUy"], messages_limit=100)
    print(res[:5])   # покажем первые 5 сообщений