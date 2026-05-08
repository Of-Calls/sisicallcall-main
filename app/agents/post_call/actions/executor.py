from __future__ import annotations

from app.agents.post_call.actions.result import action_failed, action_skipped, action_success
from app.repositories.mcp_action_log_repo import find_successful_action
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ActionExecutor:
    """action_plan.actions 를 MCPGatewayConnector 로 라우팅하고 표준 6-key
    결과 list 를 반환한다.

    Post-call action 은 MCP-only 다 — direct registry handler 는 호출하지 않는다.
    실행 흐름:

        ActionExecutor
        → MCPGatewayConnector.execute()
        → MCPProtocolClient.call_tool(mcp_tool_name, payload)
        → stdio transport
        → 자체 MCP Server (별도 process)
        → MCP Server tool
        → 외부 provider API

    실행 결과의 ``result`` 에는 source=mcp_server / via_mcp=true /
    execution_mode=mcp / mcp_tool=<dotted> metadata 가 포함되며 그대로
    mcp_action_logs.response_payload 로 저장된다.

    새 tool 추가 시 executor.py 는 수정하지 않는다 — MCP gateway tool name
    map (resolve_mcp_tool_name) 과 MCP Server provider tool 만 갱신하면 된다.
    """

    async def execute_actions(
        self,
        call_id: str,
        tenant_id: str,
        actions: list[dict] | None,
    ) -> list[dict]:
        if not actions:
            return []
        results: list[dict] = []
        for action in actions:
            results.append(
                await self._execute_one(action, call_id=call_id, tenant_id=tenant_id)
            )
        return results

    async def execute_all(self, actions: list[dict], *, call_id: str) -> list[dict]:
        """후방 호환 인터페이스 — action_router_node 가 호출한다."""
        return await self.execute_actions(
            call_id=call_id,
            tenant_id="",
            actions=actions,
        )

    async def _execute_one(
        self,
        action: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        from app.services.mcp.connectors.mcp_gateway_connector import (
            MCPClientTransportError,
            get_default_gateway,
            resolve_mcp_tool_name,
        )

        tool_key = action.get("tool", "")
        action_type = action.get("action_type", "")

        # ── idempotency check ───────────────────────────────────────────────
        previous = await find_successful_action(
            call_id=call_id,
            action_type=action_type,
            tool=tool_key,
        )
        if previous:
            logger.info(
                "action idempotency skip call_id=%s tool=%s action_type=%s previous_external_id=%s",
                call_id,
                tool_key,
                action_type,
                previous.get("external_id"),
            )
            skip_result: dict = {
                "idempotency": "already_succeeded",
                "previous_external_id": previous.get("external_id"),
                "previous_status": previous.get("status"),
                "source": "mcp_server",
                "via_mcp": True,
                "execution_mode": "mcp",
            }
            resolved_mcp_tool = resolve_mcp_tool_name(tool_key, action_type)
            if resolved_mcp_tool:
                skip_result["mcp_tool"] = resolved_mcp_tool
            return action_skipped(
                action,
                reason="already_succeeded",
                result=skip_result,
            )

        # ── unknown tool 은 gateway 를 부르지 않고 즉시 실패 ────────────────
        if resolve_mcp_tool_name(tool_key, action_type) is None:
            logger.warning(
                "MCP unknown mapping call_id=%s tool=%s action_type=%s",
                call_id, tool_key, action_type,
            )
            return action_failed(
                action,
                error=f"unknown_mcp_tool:{tool_key}.{action_type}",
                result={
                    "source": "mcp_server",
                    "via_mcp": True,
                    "execution_mode": "mcp",
                },
            )

        # ── MCP gateway 로 위임 (direct fallback 없음) ───────────────────────
        gateway = get_default_gateway()
        try:
            raw = await gateway.execute(action, call_id=call_id, tenant_id=tenant_id)
            return self._raw_to_action_result(action, raw)
        except MCPClientTransportError as exc:
            logger.error(
                "MCP transport 오류 call_id=%s tool=%s action_type=%s err=%s",
                call_id, tool_key, action_type, exc,
            )
            return action_failed(
                action,
                error=f"mcp_transport_failed:{exc}",
                result={
                    "source": "mcp_server",
                    "via_mcp": True,
                    "execution_mode": "mcp",
                    "transport_error": str(exc),
                },
            )
        except Exception as exc:
            logger.error(
                "MCP gateway 예외 call_id=%s tool=%s action_type=%s err=%s",
                call_id, tool_key, action_type, exc,
            )
            return action_failed(action, error=str(exc))

    @staticmethod
    def _raw_to_action_result(action: dict, raw: dict) -> dict:
        status = raw.get("status", "success")
        if status == "failed":
            return action_failed(
                action,
                error=raw.get("error") or "handler returned failed",
                result=raw.get("result"),
            )
        if status == "skipped":
            return action_skipped(
                action,
                reason=raw.get("error") or "handler returned skipped",
                result=raw.get("result"),
            )
        return action_success(
            action,
            external_id=raw.get("external_id"),
            result=raw.get("result"),
        )


# ── 모듈 레벨 편의 함수 ───────────────────────────────────────────────────────

_default_executor = ActionExecutor()


async def execute_actions(
    call_id: str,
    tenant_id: str,
    actions: list[dict] | None,
) -> list[dict]:
    """모듈 레벨 편의 함수 — ActionExecutor().execute_actions() 와 동일."""
    return await _default_executor.execute_actions(
        call_id=call_id,
        tenant_id=tenant_id,
        actions=actions,
    )
