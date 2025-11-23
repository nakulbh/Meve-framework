#!/usr/bin/env python3
"""
Test script for MCP save/get context functionality.

Demonstrates how to use the MeVe MCP server with context storage and ChromaDB.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from meve import MeVeConfig, ContextChunk
from meve.core.engine import load_hotpotqa_data
from meve.services import VectorDBClient
from integrations.mcp import create_meve_mcp_server, FileContextStore


def test_mcp_server():
    """Test basic MCP server functionality with save/get context using ChromaDB."""

    print("=" * 70)
    print("🧪 Testing MeVe MCP Server - Save/Get Context with ChromaDB")
    print("=" * 70)

    # Load data
    print("\n📂 Loading HotpotQA data...")
    try:
        context_chunks, questions_list = load_hotpotqa_data(max_examples=50)
        print(f"✅ Loaded {len(questions_list)} questions and {len(context_chunks)} context chunks")
    except FileNotFoundError:
        print("❌ HotpotQA data not found. Run 'make download-data' first.")
        return False

    # Create chunks list for VectorDBClient
    chunks_list = list(context_chunks.values())
    print(f"✅ Created {len(chunks_list)} context chunks for vector DB")

    # Create config
    config = MeVeConfig(k_init=5, tau_relevance=0.3, n_min=2, t_max=512)
    print(f"✅ Config: k_init={config.k_init}, tau_relevance={config.tau_relevance}")

    # Initialize ChromaDB vector client
    print("\n🔧 Initializing ChromaDB vector store...")
    vector_db_client = VectorDBClient(
        chunks=chunks_list,
        is_persistent=False,  # In-memory for testing
        collection_name="meve_test",
    )
    print(f"✅ ChromaDB vector store initialized with {len(chunks_list)} chunks")

    # Create MCP server with file-based storage
    print("\n🔧 Creating MCP server with file-based storage...")
    storage_dir = "result/mcp_test_contexts"
    storage = FileContextStore(storage_dir)
    server = create_meve_mcp_server(
        vector_store=context_chunks,
        bm25_index=context_chunks,
        config=config,
        storage=storage,
        vector_db_client=vector_db_client,
    )
    print(f"✅ MCP server created with storage at: {storage_dir}")

    # Test 1: Get context
    print("\n" + "=" * 70)
    print("TEST 1: Get Context (meve_get_context)")
    print("=" * 70)
    test_query = questions_list[0]["question"] if questions_list else "What is machine learning?"
    print(f"Query: {test_query[:80]}...")

    context = server.get_context(test_query)
    print(f"\n✅ Retrieved context ({len(context)} chars):")
    print(f"   {context[:200]}...")

    # Test 2: Save context
    print("\n" + "=" * 70)
    print("TEST 2: Save Context (meve_save_context)")
    print("=" * 70)
    result = server.save_context(query=test_query, context=context)
    context_id = result["context_id"]
    print(f"✅ Saved context with ID: {context_id}")

    # Test 3: Retrieve saved context
    print("\n" + "=" * 70)
    print("TEST 3: Get Saved Context (meve_get_saved_context)")
    print("=" * 70)
    saved = server.get_saved_context(context_id)
    print(f"✅ Retrieved saved context:")
    print(f"   ID: {saved['context_id']}")
    print(f"   Query: {saved['query'][:80]}...")
    print(f"   Timestamp: {saved['timestamp']}")
    print(f"   Context length: {len(saved['context'])} chars")

    # Test 4: Save another context
    print("\n" + "=" * 70)
    print("TEST 4: Save Another Context")
    print("=" * 70)
    test_query_2 = (
        questions_list[1]["question"] if len(questions_list) > 1 else "What is deep learning?"
    )
    print(f"Query: {test_query_2[:80]}...")
    context_2 = server.get_context(test_query_2)
    result_2 = server.save_context(query=test_query_2, context=context_2)
    context_id_2 = result_2["context_id"]
    print(f"✅ Saved second context with ID: {context_id_2}")

    # Test 5: List saved contexts
    print("\n" + "=" * 70)
    print("TEST 5: List Saved Contexts (meve_list_saved_contexts)")
    print("=" * 70)
    contexts_list = server.list_saved_contexts(limit=10)
    print(f"✅ Found {contexts_list['count']} saved contexts:")
    for ctx in contexts_list["contexts"]:
        print(f"   - {ctx['context_id']}: {ctx['query'][:60]}...")

    # Test 6: Delete context
    print("\n" + "=" * 70)
    print("TEST 6: Delete Saved Context (meve_delete_saved_context)")
    print("=" * 70)
    delete_result = server.delete_saved_context(context_id)
    print(f"✅ {delete_result['message']}")

    # Verify deletion
    verify_contexts = server.list_saved_contexts(limit=10)
    print(f"✅ After deletion: {verify_contexts['count']} saved contexts remain")

    # Test 7: Test MCP tool routing
    print("\n" + "=" * 70)
    print("TEST 7: MCP Tool Routing (handle_request)")
    print("=" * 70)

    # Test get_context via handle_request
    result = server.handle_request("meve_get_context", {"query": "test query"})
    print(f"✅ meve_get_context: returned {len(result)} chars")

    # Test save_context via handle_request
    result = server.handle_request(
        "meve_save_context",
        {"query": "routed query", "context": "test context data"},
    )
    print(f"✅ meve_save_context: {result['message']}")

    # Test list via handle_request
    result = server.handle_request("meve_list_saved_contexts", {"limit": 5})
    print(f"✅ meve_list_saved_contexts: returned {result['count']} contexts")

    # Test 8: Get available tools
    print("\n" + "=" * 70)
    print("TEST 8: Available MCP Tools")
    print("=" * 70)
    tools = server.get_all_tools()
    print(f"✅ Server exposes {len(tools)} tools:")
    for tool in tools:
        print(f"   - {tool['name']}: {tool['description'][:60]}...")

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_mcp_server()
    sys.exit(0 if success else 1)
