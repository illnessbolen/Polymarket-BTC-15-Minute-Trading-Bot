"""
Minimal Telegram Bot API client (long polling + sendMessage) on top of httpx.

Only chats listed in TELEGRAM_CHAT_ID may control the bot. Anyone else who
writes to it just gets their chat id back, which is how you find the value
to put into TELEGRAM_CHAT_ID in the first place.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

import httpx
from loguru import logger

API = "https://api.telegram.org"
MAX_MESSAGE = 4000  # Telegram's hard limit is 4096 characters

CommandHandler = Callable[[str, str, int], Awaitable[Optional[str]]]


class Telegram:
    def __init__(
        self,
        token: Optional[str],
        chat_ids: tuple[int, ...] = (),
        http: Optional[httpx.AsyncClient] = None,
        api_url: str = API,
    ):
        self.token = token
        self.chat_ids = tuple(chat_ids)
        self._api = api_url.rstrip("/")
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(40.0, connect=10.0))
        self._offset: Optional[int] = None

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _call(self, method: str, **params) -> dict:
        # httpx errors embed the request URL, which contains the bot token —
        # re-raise without it so the token never ends up in logs.
        try:
            resp = await self._http.post(f"{self._api}/bot{self.token}/{method}", json=params)
        except httpx.HTTPError as e:
            raise RuntimeError(f"Telegram {method} failed: {type(e).__name__}") from None
        try:
            data = resp.json()
        except ValueError:
            raise RuntimeError(f"Telegram {method} failed: HTTP {resp.status_code}") from None
        if not data.get("ok"):
            raise RuntimeError(f"Telegram {method} failed: {data.get('description', resp.status_code)}")
        return data["result"]

    async def get_me(self) -> dict:
        return await self._call("getMe")

    async def send(self, text: str, chat_id: Optional[int] = None) -> None:
        """Send to one chat, or broadcast to every allowed chat. Never raises."""
        if not self.enabled:
            return
        targets = (chat_id,) if chat_id is not None else self.chat_ids
        if len(text) > MAX_MESSAGE:
            text = text[: MAX_MESSAGE - 1] + "…"
        for target in targets:
            try:
                await self._call(
                    "sendMessage",
                    chat_id=target,
                    text=text,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
            except Exception as e:
                logger.warning(f"Telegram send to {target} failed: {e}")

    async def poll_once(self, handler: CommandHandler, timeout: int = 25) -> None:
        params = {"timeout": timeout, "allowed_updates": ["message"]}
        if self._offset is not None:
            params["offset"] = self._offset
        updates = await self._call("getUpdates", **params)
        for update in updates:
            self._offset = update["update_id"] + 1
            message = update.get("message") or {}
            text = (message.get("text") or "").strip()
            chat_id = (message.get("chat") or {}).get("id")
            if not text.startswith("/") or chat_id is None:
                continue

            if chat_id not in self.chat_ids:
                logger.info(f"Ignoring command from unauthorised chat {chat_id}")
                await self.send(
                    f"⛔ Нет доступа. Ваш chat_id: <code>{chat_id}</code>\n"
                    f"Добавьте его в TELEGRAM_CHAT_ID в .env и перезапустите бота.",
                    chat_id=chat_id,
                )
                continue

            # "/status@MyBot arg" -> command "status", argument "arg"
            head, _, arg = text[1:].partition(" ")
            command = head.split("@", 1)[0].lower()
            try:
                reply = await handler(command, arg.strip(), chat_id)
            except Exception as e:
                logger.exception(f"Telegram command /{command} failed")
                reply = f"⚠️ Ошибка: {e}"
            if reply:
                await self.send(reply, chat_id=chat_id)

    async def run(self, handler: CommandHandler, stop: asyncio.Event) -> None:
        if not self.enabled:
            return
        backoff = 1.0
        while not stop.is_set():
            try:
                await self.poll_once(handler)
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Telegram polling error: {e}; retrying in {backoff:.0f}s")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 60.0)
