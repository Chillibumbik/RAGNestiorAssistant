"""
VKParser: извлечение истории сообщений из личных и групповых диалогов ВКонтакте
-----------------------------------------------------------------------------

Что умеет:
- Подключаться к VK API по user token (личные диалоги и беседы) или по group token (сообщения сообщества).
- Получать историю сообщений по списку peer_id (лички, беседы) с пагинацией.
- Преобразовывать сообщения в список langchain.schema.Document с метаданными.

Ограничения/примечания:
- Для чтения ЛИЧНЫХ сообщений нужен **user access token** с правами: messages, offline.
- Для чтения сообщений СООБЩЕСТВА используйте **group access token** с правами messages; будут доступны только диалоги этого сообщества.
- Максимум 200 сообщений за один вызов messages.getHistory — используем цикл с offset.
- Для бесед peer_id = 2000000000 + chat_id. Для пользователей peer_id = user_id.
- Резолвинг screen name ограниченно поддержан (только пользователи/сообщества); для бесед передавайте integer peer_id напрямую.


Примеры использования:
    from VKParser import parse_vk_dialogs
    docs = parse_vk_dialogs(mode="user", peer_ids=[123456789, 2000000123], limit_per_dialog=500)
    # вернёт list[Document]

"""

from __future__ import annotations

import os
import time
from typing import Iterable, List, Dict, Any

from pathlib import Path
from dotenv import load_dotenv

from langchain.schema import Document

import vk_api

from tqdm import tqdm


ENV_PATH = Path(__file__).resolve().parents[2] / "Config.env"
load_dotenv(dotenv_path=ENV_PATH)


# --- Внутренние утилиты ---

def _build_vk_client(mode: str = "user", user_token: str | None = None,
                     group_token: str | None = None) -> vk_api.VkApi:
    """Создаёт VK API клиент.

    Args:
        mode: "user" или "group". Определяет, какой токен использовать.
        user_token: Явно переданный user access token (если None — берём из VK_USER_TOKEN).
        group_token: Явно переданный group access token (если None — берём из VK_GROUP_TOKEN).

    Returns:
        vk_api.VkApi с авторизацией по токену.
    """
    mode = mode.lower().strip()
    if mode not in {"user", "group"}:
        raise ValueError("mode должен быть 'user' или 'group'")

    token = user_token if mode == "user" else group_token
    if token is None:
        token = os.getenv("VK_USER_TOKEN" if mode == "user" else "VK_GROUP_TOKEN")

    if not token:
        raise RuntimeError(
            f"Не найден токен для режима {mode}. Передайте явный токен или задайте переменную окружения "
            f"{'VK_USER_TOKEN' if mode == 'user' else 'VK_GROUP_TOKEN'}."
        )

    return vk_api.VkApi(token=token)


def _resolve_screen_name(vk: vk_api.VkApiMethod, name: str) -> int | None:
    """Пытается резолвить screen name (vk.com/xxxx или просто 'xxxx') в id.

    Возвращает:
        positive int user_id / group_id (для групп возвращает отрицательный id в VK стандарте не нужен,
        но для peer_id лички с группой он не применяется), либо None если не удалось.

    Примечание: для бесед (chat) screen name не работает — используйте integer peer_id.
    """
    name = name.strip()
    if name.startswith("https://vk.com/"):
        name = name.rsplit("/", 1)[-1]
    try:
        data = vk.utils.resolveScreenName(screen_name=name)
        if not data:
            return None
        if data.get("type") == "user":
            return int(data["object_id"])  # user_id
        if data.get("type") == "group":
            return -int(data["object_id"])  # группы обычно как отрицательные owner_id; для peer_id лички это не применяется
    except Exception:
        return None
    return None


def _normalize_peer_ids(vk: vk_api.VkApiMethod, peer_ids: Iterable[int | str]) -> List[int]:
    """Приводит входные идентификаторы к integer peer_id.

    Поддерживает:
        - int (считается уже готовым peer_id)
        - 'https://vk.com/username' или 'username' (только для пользователей/сообществ → вернёт user_id; для бесед не сработает)

    Для бесед передавайте сразу integer: 2000000000 + chat_id.
    """
    out: List[int] = []
    for pid in peer_ids:
        if isinstance(pid, int):
            out.append(pid)
            continue
        # строка: попробуем резолвить как screen name
        resolved = _resolve_screen_name(vk, pid)
        if resolved is None:
            raise ValueError(f"Не удалось распознать peer_id из '{pid}'. Для бесед передавайте целочисленный peer_id.")
        # Для пользователей peer_id == user_id (положительный). Для групп личных сообщений peer_id == -group_id не используется.
        if resolved > 0:
            out.append(resolved)
        else:
            # Для сообщества личных сообщений peer_id обычно формируется иначе;
            # большинство задач парсинга сводится к пользователям и беседам.
            raise ValueError(
                f"Экранное имя '{pid}' указывает на сообщество. Передайте integer peer_id явным образом или используйте диалоги из messages.getConversations."
            )
    return out


def _fetch_history(vk: vk_api.VkApiMethod, peer_id: int, limit: int = 1000, pause: float = 0.34) -> List[Dict[str, Any]]:
    """Тянет историю сообщений по peer_id с пагинацией.

    Args:
        vk: vk_session.get_api()
        peer_id: целевой диалог (личка или беседа). Для беседы: 2000000000 + chat_id.
        limit: максимум сообщений, которые хотим получить.
        pause: задержка между запросами, чтобы не уткнуться в rate limit (~3 rps).

    Returns:
        Список "сырых" сообщений (dict), в порядке от новых к старым (как возвращает VK).
    """
    items: List[Dict[str, Any]] = []
    got = 0
    offset = 0
    per_call = 200 if limit > 200 else limit

    while got < limit:
        count = min(per_call, limit - got)
        resp = vk.messages.getHistory(peer_id=peer_id, count=count, offset=offset)
        batch = resp.get("items", [])
        if not batch:
            break
        items.extend(batch)
        got += len(batch)
        offset += len(batch)
        time.sleep(pause)
        if len(batch) < count:
            break  # дошли до конца истории
    return items


def _vk_message_to_document(msg: Dict[str, Any], peer_id: int) -> Document:
    """Преобразует vk message dict в LangChain Document.

    Поля VK message: id, date, from_id, text, attachments, fwd_messages, etc.
    """
    text = msg.get("text") or ""

    # Краткая сводка вложений (без скачивания)
    attachments = msg.get("attachments") or []
    att_summ = [att.get("type", "unknown") for att in attachments]

    metadata = {
        "platform": "vk",
        "peer_id": peer_id,
        "message_id": msg.get("id"),
        "date": msg.get("date"),
        "from_id": msg.get("from_id"),
        "attachments": att_summ,
        "has_forward": bool(msg.get("fwd_messages")),
    }
    return Document(page_content=text, metadata=metadata)


def get_messages_from_vk(
    vk_session: vk_api.VkApi,
    peer_ids: Iterable[int | str],
    limit_per_dialog: int = 1000,
) -> List[Document]:
    """Извлекает сообщения из заданных диалогов и возвращает список Document.

    Args:
        vk_session: авторизованная vk_api.VkApi (через _build_vk_client или вручную).
        peer_ids: список целевых диалогов. Поддерживаются int peer_id; для пользователей можно передать screen name.
                  Для бесед ОБЯЗАТЕЛЕН integer peer_id (2000000000 + chat_id).
        limit_per_dialog: лимит сообщений на каждый диалог (верхняя граница). Будет идти пагинация.

    Returns:
        list[Document]
    """
    vk = vk_session.get_api()

    # Нормализуем peer_ids
    normalized_peers = _normalize_peer_ids(vk, peer_ids)

    docs: List[Document] = []
    for pid in normalized_peers:
        raw_msgs = _fetch_history(vk, pid, limit=limit_per_dialog)
        for m in raw_msgs:
            docs.append(_vk_message_to_document(m, pid))
    return docs


def parse_vk_dialogs(
    mode: str = "user",
    peer_ids: Iterable[int | str] = (),
    limit_per_dialog: int = 1000,
    user_token: str | None = None,
    group_token: str | None = None,
) -> List[Document]:
    """Высокоуровневая обёртка: создаёт VK клиент, тянет историю и возвращает list[Document].

    Args:
        mode: "user" или "group" — какой токен использовать.
        peer_ids: список диалогов (peer_id int, либо screen name для пользователей/сообществ). Для бесед — только int peer_id.
        limit_per_dialog: лимит сообщений на каждый диалог.
        user_token: опционально, user access token (иначе возьмётся из VK_USER_TOKEN).
        group_token: опционально, group access token (иначе возьмётся из VK_GROUP_TOKEN).

    Returns:
        list[Document]
    """
    session = _build_vk_client(mode=mode, user_token=user_token, group_token=group_token)
    return get_messages_from_vk(session, peer_ids=peer_ids, limit_per_dialog=limit_per_dialog)


# --- Тестовый запуск ---
if __name__ == "__main__":
    # Примеры (раскомментируйте и подставьте реальные значения):

    # 1) Личные диалоги пользователя (нужен VK_USER_TOKEN в .env)
    # peer_ids можно передавать user_id (int) или screen name пользователя ('durov').
    # docs = parse_vk_dialogs(mode="user", peer_ids=[123456789, "durov"], limit_per_dialog=300)

    # 2) Беседа: peer_id = 2000000000 + chat_id
    # docs = parse_vk_dialogs(mode="user", peer_ids=[2000000123], limit_per_dialog=500)

    # 3) Сообщество (сообщения в ЛС группы), нужен VK_GROUP_TOKEN в .env
    # В этом случае передавайте peer_id пользователей, которые переписывались с сообществом.
    # docs = parse_vk_dialogs(mode="group", peer_ids=[123456789], limit_per_dialog=200)

    # print(len(docs), "documents")
    pass
