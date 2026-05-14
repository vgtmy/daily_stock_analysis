# -*- coding: utf-8 -*-
"""
Notification channel dispatcher with registry pattern.

Each configured channel registers a callable that sends the report.
The dispatcher fans out to all registered channels concurrently.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Protocol, Tuple
from enum import Enum

from src.enums import ReportType

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    WECHAT = "wechat"
    FEISHU = "feishu"
    TELEGRAM = "telegram"
    EMAIL = "email"
    PUSHOVER = "pushover"
    NTFY = "ntfy"
    GOTIFY = "gotify"
    PUSHPLUS = "pushplus"
    SERVERCHAN3 = "serverchan3"
    CUSTOM = "custom"
    DISCORD = "discord"
    SLACK = "slack"
    ASTRBOT = "astrbot"


class ChannelSender(Protocol):
    """Protocol for a channel-specific report sender."""
    def send_report(self, content: str, **kwargs) -> bool:
        ...


ChannelDispatchFn = Callable[..., bool]


class NotificationDispatcher:
    """Registry-based dispatcher that fans out reports to all registered channels.

    Channels are registered lazily based on config availability.
    The ``send_all`` method fans out concurrently via ThreadPoolExecutor.
    """

    def __init__(self, notification_service):
        self._ns = notification_service
        self._config = notification_service._config
        self._registry: Dict[NotificationChannel, ChannelDispatchFn] = {}
        self._build_registry()

    def _build_registry(self) -> None:
        """Register callables for each configured channel."""
        ns = self._ns
        available = set(ns.get_available_channels())

        if NotificationChannel.WECHAT in available:
            self._registry[NotificationChannel.WECHAT] = self._send_wechat
        if NotificationChannel.FEISHU in available:
            self._registry[NotificationChannel.FEISHU] = self._send_feishu
        if NotificationChannel.TELEGRAM in available:
            self._registry[NotificationChannel.TELEGRAM] = self._send_telegram
        if NotificationChannel.EMAIL in available:
            self._registry[NotificationChannel.EMAIL] = self._send_email
        if NotificationChannel.PUSHOVER in available:
            self._registry[NotificationChannel.PUSHOVER] = self._send_pushover
        if NotificationChannel.NTFY in available:
            self._registry[NotificationChannel.NTFY] = self._send_ntfy
        if NotificationChannel.GOTIFY in available:
            self._registry[NotificationChannel.GOTIFY] = self._send_gotify
        if NotificationChannel.PUSHPLUS in available:
            self._registry[NotificationChannel.PUSHPLUS] = self._send_pushplus
        if NotificationChannel.SERVERCHAN3 in available:
            self._registry[NotificationChannel.SERVERCHAN3] = self._send_serverchan3
        if NotificationChannel.CUSTOM in available:
            self._registry[NotificationChannel.CUSTOM] = self._send_custom
        if NotificationChannel.DISCORD in available:
            self._registry[NotificationChannel.DISCORD] = self._send_discord
        if NotificationChannel.SLACK in available:
            self._registry[NotificationChannel.SLACK] = self._send_slack
        if NotificationChannel.ASTRBOT in available:
            self._registry[NotificationChannel.ASTRBOT] = self._send_astrbot

    @property
    def channels(self) -> List[NotificationChannel]:
        return list(self._registry.keys())

    @property
    def channel_count(self) -> int:
        return len(self._registry)

    def _resolve_image_bytes(self, content: str, channel: NotificationChannel) -> Optional[bytes]:
        """Resolve markdown-to-image bytes for channels that need it."""
        ns = self._ns
        image_channels = getattr(ns, '_markdown_to_image_channels', set())
        if channel.value not in image_channels:
            return None
        if channel in (NotificationChannel.NTFY, NotificationChannel.GOTIFY):
            return None

        from src.md2img import markdown_to_image

        max_chars = getattr(ns, '_markdown_to_image_max_chars', 15000)
        return markdown_to_image(content, max_chars=max_chars)

    # ---- per-channel send implementations ----

    def _send_wechat(self, content: str, results: List, report_type: ReportType, **kwargs) -> bool:
        ns = self._ns
        if report_type == ReportType.BRIEF:
            dashboard = ns.generate_brief_report(results)
        else:
            dashboard = ns.generate_wechat_dashboard(results)

        image_bytes = self._resolve_image_bytes(dashboard, NotificationChannel.WECHAT)
        if image_bytes and ns._should_use_image_for_channel(NotificationChannel.WECHAT, image_bytes):
            return ns._send_wechat_image(image_bytes)
        return ns.send_to_wechat(dashboard)

    def _send_feishu(self, content: str, **kwargs) -> bool:
        return self._ns.send_to_feishu(content)

    def _send_telegram(self, content: str, **kwargs) -> bool:
        ns = self._ns
        image_bytes = self._resolve_image_bytes(content, NotificationChannel.TELEGRAM)
        if image_bytes and ns._should_use_image_for_channel(NotificationChannel.TELEGRAM, image_bytes):
            return ns._send_telegram_photo(image_bytes)
        return ns.send_to_telegram(content)

    def _send_email(self, content: str, **kwargs) -> bool:
        ns = self._ns
        image_bytes = self._resolve_image_bytes(content, NotificationChannel.EMAIL)
        if image_bytes and ns._should_use_image_for_channel(NotificationChannel.EMAIL, image_bytes):
            return ns._send_email_with_inline_image(image_bytes)
        return ns.send_to_email(content)

    def _send_pushover(self, content: str, **kwargs) -> bool:
        return self._ns.send_to_pushover(content)

    def _send_ntfy(self, content: str, **kwargs) -> bool:
        return self._ns.send_to_ntfy(content)

    def _send_gotify(self, content: str, **kwargs) -> bool:
        return self._ns.send_to_gotify(content)

    def _send_pushplus(self, content: str, **kwargs) -> bool:
        return self._ns.send_to_pushplus(content)

    def _send_serverchan3(self, content: str, **kwargs) -> bool:
        return self._ns.send_to_serverchan3(content)

    def _send_custom(self, content: str, **kwargs) -> bool:
        ns = self._ns
        image_bytes = self._resolve_image_bytes(content, NotificationChannel.CUSTOM)
        if image_bytes and ns._should_use_image_for_channel(NotificationChannel.CUSTOM, image_bytes):
            return ns._send_custom_webhook_image(image_bytes, fallback_content=content)
        return ns.send_to_custom(content)

    def _send_discord(self, content: str, **kwargs) -> bool:
        return self._ns.send_to_discord(content)

    def _send_slack(self, content: str, **kwargs) -> bool:
        ns = self._ns
        image_bytes = self._resolve_image_bytes(content, NotificationChannel.SLACK)
        if image_bytes and ns._should_use_image_for_channel(NotificationChannel.SLACK, image_bytes):
            bot_token = getattr(ns, '_slack_bot_token', None)
            channel_id = getattr(ns, '_slack_channel_id', None)
            if bot_token and channel_id:
                return ns._send_slack_image(image_bytes, fallback_content=content)
        return ns.send_to_slack(content)

    def _send_astrbot(self, content: str, **kwargs) -> bool:
        return self._ns.send_to_astrbot(content)

    # ---- concurrent fan-out ----

    def send_all(
        self,
        content: str,
        results: Optional[List] = None,
        report_type: ReportType = ReportType.SIMPLE,
        max_workers: int = 6,
    ) -> Dict[str, bool]:
        """Fan out the report to all registered channels concurrently.

        Returns a dict mapping channel name → success.
        """
        if not self._registry:
            logger.info("No notification channels registered, skipping push")
            return {}

        results_list = results or []
        futures: Dict[str, Tuple[NotificationChannel, ...]] = {}

        with ThreadPoolExecutor(max_workers=min(max_workers, len(self._registry))) as executor:
            for channel, send_fn in self._registry.items():
                if channel == NotificationChannel.WECHAT:
                    fut = executor.submit(send_fn, content, results=results_list, report_type=report_type)
                else:
                    fut = executor.submit(send_fn, content)
                futures[fut] = channel

            outcomes: Dict[str, bool] = {}
            for future in as_completed(futures):
                channel = futures[future]
                try:
                    success = bool(future.result())
                    outcomes[channel.value] = success
                except Exception as exc:
                    logger.exception("Channel %s raised exception: %s", channel.value, exc)
                    outcomes[channel.value] = False
                else:
                    if success:
                        logger.info("Channel %s sent successfully", channel.value)
                    else:
                        logger.warning("Channel %s send returned False", channel.value)

        return outcomes
