"""
Alerting Rules and Webhook Integration.

Provides alerting for:
- High error rates
- Latency threshold violations
- Service health issues
- Resource exhaustion

Supports webhook notifications to Slack, PagerDuty, Discord, and custom endpoints.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


class AlertChannel(str, Enum):
    """Alert notification channels."""
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class AlertRule:
    """
    Definition of an alert rule.

    Attributes:
        name: Unique rule identifier
        description: Human-readable description
        condition: Function that returns True when alert should fire
        severity: Alert severity level
        cooldown_seconds: Minimum time between alerts
        channels: Notification channels to use
    """

    name: str
    description: str
    condition: Callable[[], bool]
    severity: AlertSeverity = AlertSeverity.WARNING
    cooldown_seconds: int = 300  # 5 minutes
    channels: list[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG])
    labels: dict[str, str] = field(default_factory=dict)

    # State
    _last_fired: datetime | None = field(default=None, repr=False)
    _firing: bool = field(default=False, repr=False)

    def should_fire(self) -> bool:
        """Check if alert should fire based on condition and cooldown."""
        if not self.condition():
            self._firing = False
            return False

        now = datetime.utcnow()

        # Check cooldown
        if self._last_fired:
            cooldown_end = self._last_fired + timedelta(seconds=self.cooldown_seconds)
            if now < cooldown_end:
                return False

        self._last_fired = now
        self._firing = True
        return True


@dataclass
class Alert:
    """An alert instance."""

    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    description: str
    fired_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "status": self.status.value,
            "description": self.description,
            "fired_at": self.fired_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "labels": self.labels,
            "annotations": self.annotations,
        }


@dataclass
class WebhookConfig:
    """Configuration for webhook notifications."""

    channel: AlertChannel
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 10

    # Channel-specific settings
    slack_channel: str | None = None
    pagerduty_routing_key: str | None = None
    discord_username: str | None = None


class AlertNotifier:
    """
    Sends alert notifications to configured channels.
    """

    def __init__(self):
        self._webhooks: dict[AlertChannel, WebhookConfig] = {}
        self._http_client = None

    def configure_webhook(self, config: WebhookConfig) -> None:
        """Configure a webhook for a channel."""
        self._webhooks[config.channel] = config
        logger.info(
            "Webhook configured",
            channel=config.channel.value,
            url=config.url[:50] + "..." if len(config.url) > 50 else config.url,
        )

    async def send_alert(self, alert: Alert, channels: list[AlertChannel]) -> None:
        """Send alert to specified channels."""
        for channel in channels:
            try:
                if channel == AlertChannel.LOG:
                    self._log_alert(alert)
                elif channel in self._webhooks:
                    await self._send_webhook(alert, self._webhooks[channel])
                else:
                    logger.warning(
                        "No webhook configured for channel",
                        channel=channel.value,
                    )
            except Exception as e:
                logger.error(
                    "Failed to send alert",
                    channel=channel.value,
                    error=str(e),
                )

    def _log_alert(self, alert: Alert) -> None:
        """Log alert to structured logger."""
        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical,
        }.get(alert.severity, logger.warning)

        log_func(
            f"ALERT [{alert.severity.value.upper()}]: {alert.rule_name}",
            description=alert.description,
            status=alert.status.value,
            labels=alert.labels,
        )

    async def _send_webhook(self, alert: Alert, config: WebhookConfig) -> None:
        """Send alert via webhook."""
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed, cannot send webhook")
            return

        payload = self._format_payload(alert, config)

        async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
            response = await client.post(
                config.url,
                json=payload,
                headers=config.headers,
            )
            response.raise_for_status()

        logger.debug(
            "Webhook sent",
            channel=config.channel.value,
            status_code=response.status_code,
        )

    def _format_payload(self, alert: Alert, config: WebhookConfig) -> dict[str, Any]:
        """Format alert payload for specific channel."""
        if config.channel == AlertChannel.SLACK:
            return self._format_slack(alert, config)
        elif config.channel == AlertChannel.PAGERDUTY:
            return self._format_pagerduty(alert, config)
        elif config.channel == AlertChannel.DISCORD:
            return self._format_discord(alert, config)
        else:
            return alert.to_dict()

    def _format_slack(self, alert: Alert, config: WebhookConfig) -> dict[str, Any]:
        """Format alert for Slack webhook."""
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffcc00",
            AlertSeverity.ERROR: "#ff6600",
            AlertSeverity.CRITICAL: "#ff0000",
        }.get(alert.severity, "#808080")

        return {
            "channel": config.slack_channel,
            "attachments": [{
                "color": color,
                "title": f"[{alert.severity.value.upper()}] {alert.rule_name}",
                "text": alert.description,
                "fields": [
                    {"title": "Status", "value": alert.status.value, "short": True},
                    {"title": "Time", "value": alert.fired_at.isoformat(), "short": True},
                ] + [
                    {"title": k, "value": str(v), "short": True}
                    for k, v in alert.labels.items()
                ],
                "footer": "Ontology Reasoning System",
                "ts": int(alert.fired_at.timestamp()),
            }],
        }

    def _format_pagerduty(self, alert: Alert, config: WebhookConfig) -> dict[str, Any]:
        """Format alert for PagerDuty Events API v2."""
        severity_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical",
        }

        return {
            "routing_key": config.pagerduty_routing_key,
            "event_action": "trigger" if alert.status == AlertStatus.FIRING else "resolve",
            "dedup_key": alert.rule_name,
            "payload": {
                "summary": f"{alert.rule_name}: {alert.description}",
                "severity": severity_map.get(alert.severity, "warning"),
                "source": "ontology-reasoning-system",
                "timestamp": alert.fired_at.isoformat(),
                "custom_details": {
                    **alert.labels,
                    **alert.annotations,
                },
            },
        }

    def _format_discord(self, alert: Alert, config: WebhookConfig) -> dict[str, Any]:
        """Format alert for Discord webhook."""
        color = {
            AlertSeverity.INFO: 0x36a64f,
            AlertSeverity.WARNING: 0xffcc00,
            AlertSeverity.ERROR: 0xff6600,
            AlertSeverity.CRITICAL: 0xff0000,
        }.get(alert.severity, 0x808080)

        return {
            "username": config.discord_username or "Ontology Alerts",
            "embeds": [{
                "title": f"[{alert.severity.value.upper()}] {alert.rule_name}",
                "description": alert.description,
                "color": color,
                "fields": [
                    {"name": k, "value": str(v), "inline": True}
                    for k, v in alert.labels.items()
                ],
                "timestamp": alert.fired_at.isoformat(),
            }],
        }


class AlertManager:
    """
    Manages alert rules and notifications.

    Features:
    - Rule registration and evaluation
    - Alert deduplication
    - Notification routing
    - Alert history
    """

    def __init__(self):
        self._rules: dict[str, AlertRule] = {}
        self._active_alerts: dict[str, Alert] = {}
        self._alert_history: list[Alert] = []
        self._notifier = AlertNotifier()
        self._max_history = 1000

        # Register default rules
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default alerting rules."""
        # These rules need access to metrics, so they're defined as lambdas
        # that will be evaluated at check time
        pass  # Rules are registered externally with actual metric access

    def register_rule(self, rule: AlertRule) -> None:
        """Register an alert rule."""
        self._rules[rule.name] = rule
        logger.info(
            "Alert rule registered",
            rule_name=rule.name,
            severity=rule.severity.value,
        )

    def configure_webhook(self, config: WebhookConfig) -> None:
        """Configure a notification webhook."""
        self._notifier.configure_webhook(config)

    async def check_rules(self) -> list[Alert]:
        """Check all rules and fire alerts as needed."""
        fired_alerts = []

        for rule in self._rules.values():
            try:
                if rule.should_fire():
                    alert = Alert(
                        rule_name=rule.name,
                        severity=rule.severity,
                        status=AlertStatus.FIRING,
                        description=rule.description,
                        labels=rule.labels.copy(),
                    )

                    self._active_alerts[rule.name] = alert
                    self._alert_history.append(alert)
                    fired_alerts.append(alert)

                    # Send notifications
                    await self._notifier.send_alert(alert, rule.channels)

                elif rule.name in self._active_alerts and not rule._firing:
                    # Alert resolved
                    alert = self._active_alerts.pop(rule.name)
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_at = datetime.utcnow()

                    # Send resolved notification
                    await self._notifier.send_alert(alert, rule.channels)

            except Exception as e:
                logger.error(
                    "Rule evaluation failed",
                    rule_name=rule.name,
                    error=str(e),
                )

        # Trim history
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history:]

        return fired_alerts

    def get_active_alerts(self) -> list[Alert]:
        """Get all currently active alerts."""
        return list(self._active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> list[Alert]:
        """Get recent alert history."""
        return self._alert_history[-limit:]

    def acknowledge_alert(self, rule_name: str) -> bool:
        """Acknowledge an active alert."""
        if rule_name in self._active_alerts:
            self._active_alerts[rule_name].status = AlertStatus.ACKNOWLEDGED
            return True
        return False

    def get_status(self) -> dict[str, Any]:
        """Get alert manager status."""
        return {
            "rules_count": len(self._rules),
            "active_alerts": len(self._active_alerts),
            "history_count": len(self._alert_history),
            "rules": [
                {
                    "name": r.name,
                    "severity": r.severity.value,
                    "firing": r._firing,
                    "last_fired": r._last_fired.isoformat() if r._last_fired else None,
                }
                for r in self._rules.values()
            ],
            "active": [a.to_dict() for a in self._active_alerts.values()],
        }


# Global alert manager
_alert_manager: AlertManager | None = None


def get_alert_manager() -> AlertManager:
    """Get the global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def create_metric_alert_rules(metrics_registry) -> list[AlertRule]:
    """
    Create alert rules based on metrics.

    Args:
        metrics_registry: The metrics registry to monitor

    Returns:
        List of alert rules to register
    """
    rules = []

    # High error rate alert
    rules.append(AlertRule(
        name="high_error_rate",
        description="Error rate exceeds 10%",
        condition=lambda: (
            metrics_registry.query_metrics._total_queries > 10 and
            metrics_registry.query_metrics._failed_queries /
            metrics_registry.query_metrics._total_queries > 0.10
        ),
        severity=AlertSeverity.ERROR,
        cooldown_seconds=300,
        labels={"component": "query"},
    ))

    # High latency alert (p95 > 10s)
    rules.append(AlertRule(
        name="high_latency_p95",
        description="Query latency p95 exceeds 10 seconds",
        condition=lambda: (
            metrics_registry.query_metrics._latency_stats.count > 10 and
            (metrics_registry.query_metrics._latency_stats.compute() or True) and
            metrics_registry.query_metrics._latency_stats.p95 > 10000
        ),
        severity=AlertSeverity.WARNING,
        cooldown_seconds=600,
        labels={"component": "query", "metric": "latency"},
    ))

    # Very high latency alert (p99 > 30s)
    rules.append(AlertRule(
        name="critical_latency_p99",
        description="Query latency p99 exceeds 30 seconds",
        condition=lambda: (
            metrics_registry.query_metrics._latency_stats.count > 10 and
            (metrics_registry.query_metrics._latency_stats.compute() or True) and
            metrics_registry.query_metrics._latency_stats.p99 > 30000
        ),
        severity=AlertSeverity.CRITICAL,
        cooldown_seconds=300,
        labels={"component": "query", "metric": "latency"},
    ))

    # Low confidence scores
    rules.append(AlertRule(
        name="low_confidence_scores",
        description="Average confidence score below 0.5",
        condition=lambda: (
            metrics_registry.query_metrics._confidence_stats.count > 10 and
            (metrics_registry.query_metrics._confidence_stats.compute() or True) and
            metrics_registry.query_metrics._confidence_stats.mean < 0.5
        ),
        severity=AlertSeverity.WARNING,
        cooldown_seconds=900,
        labels={"component": "query", "metric": "confidence"},
    ))

    # High MACER iterations
    rules.append(AlertRule(
        name="high_macer_iterations",
        description="Average MACER iterations exceeds 7",
        condition=lambda: (
            metrics_registry.query_metrics._iteration_stats.count > 10 and
            (metrics_registry.query_metrics._iteration_stats.compute() or True) and
            metrics_registry.query_metrics._iteration_stats.mean > 7
        ),
        severity=AlertSeverity.WARNING,
        cooldown_seconds=600,
        labels={"component": "macer", "metric": "iterations"},
    ))

    return rules


async def start_alert_checker(
    interval_seconds: int = 60,
    metrics_registry=None,
) -> asyncio.Task:
    """
    Start background task to periodically check alert rules.

    Args:
        interval_seconds: Check interval
        metrics_registry: Metrics registry to monitor

    Returns:
        Background task
    """
    manager = get_alert_manager()

    # Register metric-based rules if registry provided
    if metrics_registry:
        for rule in create_metric_alert_rules(metrics_registry):
            manager.register_rule(rule)

    async def checker():
        while True:
            try:
                await manager.check_rules()
            except Exception as e:
                logger.error("Alert check failed", error=str(e))

            await asyncio.sleep(interval_seconds)

    task = asyncio.create_task(checker())
    logger.info(
        "Alert checker started",
        interval_seconds=interval_seconds,
        rules_count=len(manager._rules),
    )
    return task
