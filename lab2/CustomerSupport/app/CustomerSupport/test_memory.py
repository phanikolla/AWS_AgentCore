"""Test and verify Bedrock AgentCore Memory with the Customer Support Agent (Lab 2).

This script:
1. Resolves or creates the AgentCore Memory resource 'CustomerSupportMemory' in us-east-1.
2. Seeds prior customer conversations (Short-Term Memory).
3. Demonstrates Long-Term Memory (LTM) retrieval of preferences and semantic facts.
4. Invokes the Customer Support Agent to verify personalized responses.
"""

import sys
import io
import time
import uuid

# Fix Windows console encoding for Unicode/emojis
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from bedrock_agentcore.memory import MemoryClient
from memory.manager import (
    REGION,
    MEMORY_NAME,
    get_or_create_memory_resource,
)
from main import create_agent

ACTOR_ID = "customer_001"
SESSION_ID = f"session-{uuid.uuid4().hex[:8]}"

# Seed customer interactions
PREVIOUS_INTERACTIONS = [
    ("I'm having issues with my MacBook Pro overheating during video editing.", "USER"),
    (
        "I can help with that thermal issue. For video editing workloads, let's check your Activity Monitor and adjust performance settings. Your MacBook Pro order #MB-78432 is still under warranty.",
        "ASSISTANT",
    ),
    (
        "What's the return policy on gaming headphones? I need low latency for competitive FPS games",
        "USER",
    ),
    (
        "For gaming headphones, you have 30 days to return. Since you're into competitive FPS, I'd recommend checking the audio latency specs - most gaming models have <40ms latency.",
        "ASSISTANT",
    ),
    (
        "I need a laptop under $1200 for programming. Prefer 16GB RAM minimum and good Linux compatibility. I like ThinkPad models.",
        "USER",
    ),
    (
        "Perfect! For development work, I'd suggest looking at our ThinkPad E series or Dell XPS models. Both have excellent Linux support and 16GB RAM options within your budget.",
        "ASSISTANT",
    ),
]


def step1_ensure_memory() -> str:
    print(f"\n==================================================")
    print(f"Step 1: Get or Create AgentCore Memory in {REGION}")
    print(f"==================================================")
    memory_id = get_or_create_memory_resource(region_name=REGION)
    print(f"✅ Active Memory ID: {memory_id}")
    return memory_id


def step2_seed_history(memory_id: str, client: MemoryClient):
    print(f"\n==================================================")
    print(f"Step 2: Seed Customer Interactions into STM")
    print(f"==================================================")
    try:
        client.create_event(
            memory_id=memory_id,
            actor_id=ACTOR_ID,
            session_id=SESSION_ID,
            messages=PREVIOUS_INTERACTIONS,
        )
        print("✅ Seeded customer history successfully")
        print("📝 Interactions saved to Short-Term Memory (STM)")
        print("⏳ Long-Term Memory (LTM) processing will run asynchronously...")
    except Exception as e:
        print(f"⚠️ Error seeding history: {e}")


def step3_check_ltm(memory_id: str, client: MemoryClient):
    print(f"\n==================================================")
    print(f"Step 3: Check Long-Term Memory (Preferences & Facts)")
    print(f"==================================================")

    print("🔍 Checking customer preferences...")
    retries = 0
    max_retries = 6
    pref_memories = []

    while retries < max_retries:
        try:
            pref_memories = client.retrieve_memories(
                memory_id=memory_id,
                namespace=f"support/customer/{ACTOR_ID}/preferences/",
                query="customer preferences and habits",
            )
            if pref_memories:
                print(f"✅ Found {len(pref_memories)} customer preference memories:")
                for i, mem in enumerate(pref_memories, 1):
                    content = mem.get("content", {})
                    text = content.get("text", "") if isinstance(content, dict) else str(content)
                    print(f"   {i}. {text}")
                break
        except Exception as e:
            print(f"   Note during preference retrieval: {e}")

        retries += 1
        if retries < max_retries:
            print(f"⏳ Waiting for LTM processing... (attempt {retries}/{max_retries})")
            time.sleep(10)

    print("\n🔍 Checking semantic facts...")
    try:
        fact_memories = client.retrieve_memories(
            memory_id=memory_id,
            namespace=f"support/customer/{ACTOR_ID}/semantic/",
            query="MacBook order or laptop specifications",
        )
        if fact_memories:
            print(f"✅ Found {len(fact_memories)} semantic factual memories:")
            for i, mem in enumerate(fact_memories, 1):
                content = mem.get("content", {})
                text = content.get("text", "") if isinstance(content, dict) else str(content)
                print(f"   {i}. {text}")
        else:
            print("ℹ️ Semantic memories still processing or not yet extracted.")
    except Exception as e:
        print(f"   Note during semantic retrieval: {e}")


def step4_test_agent_personalization():
    print(f"\n==================================================")
    print(f"Step 4: Test Agent with Persistent Personalization")
    print(f"==================================================")

    agent = create_agent(actor_id=ACTOR_ID, session_id=f"test-{uuid.uuid4().hex[:6]}")

    test_queries = [
        "Which headphones would you recommend?",
        "What is my preferred laptop brand and requirements?",
    ]

    for q in test_queries:
        print(f"\n--- Customer Query: \"{q}\" ---")
        try:
            response = agent(q)
            print(f"Agent Response:\n{response}")
        except Exception as e:
            print(f"Error querying agent: {e}")


def main():
    print("🚀 Starting Lab 2 Bedrock AgentCore Memory Verification...")
    client = MemoryClient(region_name=REGION)
    memory_id = step1_ensure_memory()
    step2_seed_history(memory_id, client)
    step3_check_ltm(memory_id, client)
    step4_test_agent_personalization()
    print("\n🎉 Lab 2 Memory test complete!")


if __name__ == "__main__":
    main()
