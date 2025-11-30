"""
Example: Running MeVe REST API server.

This example demonstrates:
1. Loading data from HotpotQA
2. Initializing MeVeEngine
3. Creating and running FastAPI server
4. Making requests via HTTP client

Usage:
    python examples/api_server.py
    
Then in another terminal:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/retrieve \\
        -H "Content-Type: application/json" \\
        -d '{"query": "What is photosynthesis?"}'
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from meve import MeVeEngine, MeVeConfig, ContextChunk
from meve.core.engine import load_hotpotqa_data
from integrations.api import create_app
from meve.utils import get_logger

logger = get_logger(__name__)


def load_sample_data():
    """Load sample data from HotpotQA."""
    try:
        # Load HotpotQA data
        data_dir = Path(__file__).parent.parent / "data"
        hotpot_file = data_dir / "hotpot_dev_distractor_v1.json"

        if not hotpot_file.exists():
            logger.warn(f"HotpotQA file not found at {hotpot_file}")
            logger.info("Creating sample data for demonstration...")
            return create_sample_chunks()

        # Load from HotpotQA
        chunks, _ = load_hotpotqa_data(str(hotpot_file), max_examples=100)
        logger.info(f"Loaded {len(chunks)} chunks from HotpotQA")
        return chunks

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return create_sample_chunks()


def create_sample_chunks():
    """Create sample chunks for demonstration."""
    sample_data = [
        {
            "content": "Photosynthesis is a process used by plants, algae, and certain bacteria to harness energy from sunlight and turn it into chemical energy. Light-dependent reactions occur in the thylakoid membrane of the chloroplast and light-independent reactions occur in the stroma.",
            "doc_id": "bio_1",
        },
        {
            "content": "The photosynthetic process consists of two main stages: the light reactions and the Calvin cycle. The light reactions capture energy from sunlight, while the Calvin cycle uses that energy to synthesize glucose.",
            "doc_id": "bio_2",
        },
        {
            "content": "Chlorophyll is the primary pigment in photosynthesis, absorbing light energy and converting it to chemical energy stored in ATP and NADPH. These energy carriers then power the Calvin cycle.",
            "doc_id": "bio_3",
        },
        {
            "content": "The equation for photosynthesis is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂. This fundamental process is the basis of most food chains on Earth.",
            "doc_id": "bio_4",
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that focuses on training algorithms to learn patterns from data without explicit programming. It powers recommendation systems, natural language processing, and computer vision.",
            "doc_id": "ml_1",
        },
    ]

    chunks = {}
    for item in sample_data:
        chunk = ContextChunk(
            content=item["content"],
            doc_id=item["doc_id"],
        )
        chunks[item["doc_id"]] = chunk

    return chunks


def main():
    """Initialize and run the API server."""
    logger.info("Initializing MeVe REST API server...")

    # Load data
    chunks = load_sample_data()

    # Create engine
    config = MeVeConfig(
        k_init=5,
        tau_relevance=0.5,
        n_min=2,
        theta_redundancy=0.85,
        lambda_mmr=0.6,
        t_max=512,
    )

    engine = MeVeEngine(
        config=config,
        vector_store=chunks,
        bm25_index=chunks,
    )

    logger.info(f"Engine initialized with {len(chunks)} chunks")

    # Create FastAPI app
    app = create_app(engine)

    logger.info("=" * 60)
    logger.info("MeVe REST API Server Starting")
    logger.info("=" * 60)
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("Alternative Docs:  http://localhost:8000/redoc")
    logger.info("")
    logger.info("Example Requests:")
    logger.info("  1. Health check: curl http://localhost:8000/health")
    logger.info("  2. Retrieve:     curl -X POST http://localhost:8000/retrieve \\")
    logger.info('       -H "Content-Type: application/json" \\')
    logger.info('       -d \'{"query": "What is photosynthesis?"}\'')
    logger.info("  3. Get config:   curl http://localhost:8000/config")
    logger.info("")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 60)
    logger.info("")

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
