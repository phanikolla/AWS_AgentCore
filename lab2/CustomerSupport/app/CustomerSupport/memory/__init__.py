"""AgentCore Memory integration for Customer Support Agent."""
from .manager import (
    REGION,
    MEMORY_NAME,
    SSM_PARAM_MEMORY_ID,
    get_or_create_memory_resource,
    create_memory_session_manager,
)

__all__ = [
    "REGION",
    "MEMORY_NAME",
    "SSM_PARAM_MEMORY_ID",
    "get_or_create_memory_resource",
    "create_memory_session_manager",
]
