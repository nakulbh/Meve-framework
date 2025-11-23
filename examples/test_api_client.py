"""
Example: Testing MeVe REST API client.

This example demonstrates how to use the MeVeClient to interact with the API.

Prerequisites:
    1. Start API server: python examples/api_server.py
    2. In another terminal, run this example: python examples/test_api_client.py

Usage:
    python examples/test_api_client.py
"""

import sys
from pathlib import Path
import time
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.api.client import MeVeClient
from meve.utils import get_logger

logger = get_logger(__name__)


def test_health_check():
    """Test health endpoint."""
    logger.info("Testing health check...")

    with MeVeClient() as client:
        try:
            health = client.health()
            logger.info(f"Health Status: {health['status']}")
            logger.info(f"  Vector DB Ready: {health['vector_db_ready']}")
            logger.info(f"  BM25 Index Ready: {health['bm25_index_ready']}")
            logger.info(f"  Message: {health.get('message', 'N/A')}")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


def test_retrieval():
    """Test retrieval endpoint."""
    logger.info("Testing retrieval...")

    with MeVeClient() as client:
        try:
            queries = [
                "What is photosynthesis?",
                "How does machine learning work?",
                "What is the light reaction?",
            ]

            for query in queries:
                logger.info(f"Query: {query}")

                result = client.retrieve(query)

                logger.info(f"  Request ID: {result['request_id']}")
                logger.info(f"  Status: {result['status']}")
                logger.info(f"  Processing Time: {result['processing_time_ms']:.2f}ms")
                logger.info(f"  Chunks Retrieved: {len(result['chunks'])}")
                logger.info(f"  Context Length: {len(result['context'])} chars")

                # Print metrics
                metrics = result["metrics"]
                logger.info(f"  Metrics:")
                logger.info(f"    - Phase 1 Candidates: {metrics['phase_1_candidates']}")
                logger.info(f"    - Phase 2 Verified: {metrics['phase_2_verified']}")
                logger.info(f"    - Phase 3 Triggered: {metrics['phase_3_triggered']}")
                logger.info(f"    - Phase 4 Deduplicated: {metrics['phase_4_deduplicated']}")
                logger.info(f"    - Phase 5 Final Tokens: {metrics['phase_5_final_tokens']}")
                logger.info("")

                time.sleep(0.5)

            return True
        except Exception as e:
            logger.error(f"Retrieval test failed: {e}")
            return False


def test_config():
    """Test configuration endpoints."""
    logger.info("Testing configuration...")

    with MeVeClient() as client:
        try:
            # Get current config
            logger.info("Getting current configuration...")
            config = client.get_config()
            logger.info(f"Current Config: {json.dumps(config, indent=2)}")

            # Update config
            logger.info("Updating configuration...")
            updated = client.update_config(k_init=10, tau_relevance=0.6)
            logger.info(f"Updated Config: {json.dumps(updated['config'], indent=2)}")

            # Reset config
            logger.info("Resetting configuration...")
            reset = client.reset_config()
            logger.info(f"Reset Config: {json.dumps(reset['config'], indent=2)}")

            return True
        except Exception as e:
            logger.error(f"Config test failed: {e}")
            return False


def test_data_sources():
    """Test data source endpoints."""
    logger.info("Testing data sources...")

    with MeVeClient() as client:
        try:
            # Get data source info
            logger.info("Getting data source info...")
            info = client.data_source_info()
            logger.info(f"Data Source Info: {json.dumps(info, indent=2)}")

            # List chunks
            logger.info("Listing chunks...")
            chunks = client.list_chunks(limit=5)
            logger.info(f"Total Available: {chunks['total_available']}")
            logger.info(f"Returned: {chunks['returned']}")
            logger.info("Chunks:")
            for chunk in chunks["chunks"]:
                logger.info(f"  - {chunk['doc_id']}: {chunk['content'][:60]}...")

            return True
        except Exception as e:
            logger.error(f"Data sources test failed: {e}")
            return False


def test_metrics():
    """Test metrics retrieval."""
    logger.info("Testing metrics retrieval...")

    with MeVeClient() as client:
        try:
            # Perform a retrieval
            logger.info("Performing retrieval to get metrics...")
            result = client.retrieve("What is photosynthesis?")
            request_id = result["request_id"]

            # Get metrics
            logger.info(f"Getting metrics for request {request_id}...")
            metrics = client.get_metrics(request_id)
            logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

            return True
        except Exception as e:
            logger.error(f"Metrics test failed: {e}")
            return False


def test_config_override():
    """Test per-query config override."""
    logger.info("Testing per-query config override...")

    with MeVeClient() as client:
        try:
            # Retrieve with config override
            logger.info("Retrieving with config override...")
            result = client.retrieve(
                "What is photosynthesis?",
                k_init=5,
                tau_relevance=0.4,
                t_max=256,
            )

            logger.info(f"Query executed with custom config")
            logger.info(f"  Request ID: {result['request_id']}")
            logger.info(f"  Chunks: {len(result['chunks'])}")
            logger.info(f"  Processing Time: {result['processing_time_ms']:.2f}ms")

            return True
        except Exception as e:
            logger.error(f"Config override test failed: {e}")
            return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("MeVe REST API Client Tests")
    logger.info("=" * 60)
    logger.info("")

    tests = [
        ("Health Check", test_health_check),
        ("Retrieval", test_retrieval),
        ("Configuration", test_config),
        ("Data Sources", test_data_sources),
        ("Metrics", test_metrics),
        ("Config Override", test_config_override),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"Running: {test_name}")
        logger.info("-" * 40)

        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test crashed: {e}")
            results.append((test_name, False))

        logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("")
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
