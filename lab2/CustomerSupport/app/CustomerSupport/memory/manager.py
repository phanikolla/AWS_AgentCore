import os
import uuid
import logging
import boto3
from typing import Optional

from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.constants import StrategyType
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig,
    RetrievalConfig,
)
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)

logger = logging.getLogger("CustomerSupport.Memory")

REGION = os.getenv("AWS_REGION", "us-east-1")
MEMORY_NAME = "CustomerSupportMemory"
SSM_PARAM_MEMORY_ID = "/app/customersupport/agentcore/memory_id"


def get_ssm_parameter(param_name: str, region_name: str = REGION) -> Optional[str]:
    """Retrieve parameter from AWS Systems Manager Parameter Store."""
    try:
        ssm = boto3.client("ssm", region_name=region_name)
        response = ssm.get_parameter(Name=param_name, WithDecryption=False)
        return response.get("Parameter", {}).get("Value")
    except Exception as e:
        logger.debug(f"Parameter {param_name} not found in SSM: {e}")
        return None


def put_ssm_parameter(param_name: str, value: str, region_name: str = REGION) -> bool:
    """Store parameter in AWS Systems Manager Parameter Store."""
    try:
        ssm = boto3.client("ssm", region_name=region_name)
        ssm.put_parameter(
            Name=param_name,
            Value=value,
            Type="String",
            Overwrite=True,
            Description="Bedrock AgentCore Memory ID for Customer Support Agent",
        )
        logger.info(f"Stored {param_name} in SSM: {value}")
        return True
    except Exception as e:
        logger.warning(f"Failed to store parameter in SSM: {e}")
        return False


def get_or_create_memory_resource(region_name: str = REGION) -> Optional[str]:
    """Get existing or create new AgentCore Memory resource in the specified region.
    
    Checks in order:
    1. AGENTCORE_MEMORY_ID environment variable
    2. SSM Parameter Store (/app/customersupport/agentcore/memory_id)
    3. Existing memories via list_memories()
    4. Creates new CustomerSupportMemory if not found
    """
    # 1. Environment variable
    env_memory_id = os.getenv("AGENTCORE_MEMORY_ID")
    if env_memory_id:
        logger.info(f"Using memory ID from environment: {env_memory_id}")
        return env_memory_id

    # 2. SSM Parameter
    ssm_memory_id = get_ssm_parameter(SSM_PARAM_MEMORY_ID, region_name=region_name)
    client = MemoryClient(region_name=region_name)

    if ssm_memory_id:
        try:
            memory_info = client.get_memory(memory_id=ssm_memory_id)
            status = memory_info.get("status")
            if status in ("ACTIVE", "CREATING"):
                logger.info(f"Found active memory ID in SSM: {ssm_memory_id} (status: {status})")
                return ssm_memory_id
        except Exception as e:
            logger.warning(f"Memory ID {ssm_memory_id} from SSM is invalid or deleted: {e}")

    # 3. Check existing memories by name
    try:
        existing_memories = client.list_memories()
        for mem in existing_memories:
            name = mem.get("name") or mem.get("id")
            if name == MEMORY_NAME or (isinstance(name, str) and name.startswith(MEMORY_NAME)):
                found_id = mem.get("id")
                logger.info(f"Found existing memory '{name}' with ID: {found_id}")
                put_ssm_parameter(SSM_PARAM_MEMORY_ID, found_id, region_name=region_name)
                return found_id
    except Exception as e:
        logger.warning(f"Could not list memories: {e}")

    # 4. Create new memory resource with USER_PREFERENCE and SEMANTIC strategies
    try:
        logger.info(f"Creating new AgentCore Memory '{MEMORY_NAME}' in region {region_name}...")
        strategies = [
            {
                StrategyType.USER_PREFERENCE.value: {
                    "name": "CustomerPreferences",
                    "description": "Captures customer preferences and behavior",
                    "namespaces": ["support/customer/{actorId}/preferences/"],
                }
            },
            {
                StrategyType.SEMANTIC.value: {
                    "name": "CustomerSupportSemantic",
                    "description": "Stores facts from conversations",
                    "namespaces": ["support/customer/{actorId}/semantic/"],
                }
            },
        ]
        response = client.create_memory_and_wait(
            name=MEMORY_NAME,
            description="Customer Support Agent Memory (Lab 2)",
            strategies=strategies,
            event_expiry_days=90,
            max_wait=360,
            poll_interval=10,
        )
        created_id = response.get("id")
        if created_id:
            logger.info(f"Successfully created AgentCore Memory: {created_id}")
            put_ssm_parameter(SSM_PARAM_MEMORY_ID, created_id, region_name=region_name)
            return created_id
    except Exception as e:
        logger.error(f"Failed to create AgentCore Memory: {e}")
        raise

    return None


def create_memory_session_manager(
    memory_id: str,
    actor_id: str = "customer_001",
    session_id: Optional[str] = None,
    region_name: str = REGION,
) -> AgentCoreMemorySessionManager:
    """Create a Strands AgentCoreMemorySessionManager configured for customer preferences and semantics."""
    if not session_id:
        session_id = str(uuid.uuid4())

    memory_config = AgentCoreMemoryConfig(
        memory_id=memory_id,
        session_id=session_id,
        actor_id=actor_id,
        retrieval_config={
            "support/customer/{actorId}/semantic/": RetrievalConfig(top_k=3, relevance_score=0.2),
            "support/customer/{actorId}/preferences/": RetrievalConfig(top_k=3, relevance_score=0.2),
        },
    )
    return AgentCoreMemorySessionManager(agentcore_memory_config=memory_config, region_name=region_name)
