# ChromaDB Test Server for MeVe MCP

This folder contains scripts to quickly set up a ChromaDB server with test datasets for testing the MeVe MCP integration.

## Quick Start

### 1. Populate Test Data

**Option A: Simple Test Data (Recommended for Quick Testing)**

```bash
cd test_chromadb_server
uv run python populate_simple_test_data.py
```

This creates sample collections instantly:
- **tech_articles** - 20 technology and AI articles
- **business_concepts** - 15 startup and business concepts

No external dependencies, embeddings computed automatically!

**Option B: Chroma Datasets (For Realistic Testing)**

```bash
uv run python populate_test_datasets.py
```

This imports pre-built datasets with OpenAI embeddings:
- **State of the Union** (51kb, ~40 chunks) - Small, perfect for quick tests
- **Paul Graham Essays** (1.3mb, ~300 chunks) - Medium, good for real-world testing
- **Huberman Podcasts** (4.3mb, ~1000 chunks) - Large, for performance testing

*Note: Requires `chroma-datasets` package compatibility*

### 2. Start ChromaDB Server

**Option A: Using ChromaDB CLI (Recommended)**

```bash
uv run start_test_server.py
```

**Option B: Using Docker**

```bash
uv run start_test_server.py --docker
```

The server will start at `http://localhost:8000`

### 3. Configure MeVe MCP Server

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "meve-rag": {
      "command": "/Users/YOUR_USERNAME/.local/bin/uv",
      "args": [
        "--directory",
        "/path/to/Meve-framework",
        "run",
        "meve_mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "/path/to/Meve-framework",
        "CHROMADB_URL": "http://localhost:8000"
      }
    }
  }
}
```

### 4. Restart Claude Desktop

### 5. Test in Claude Desktop

```
"List all ChromaDB collections"
"Load the state_of_the_union collection"
"Search for information about infrastructure using the MeVe pipeline"
```

## Available Datasets

### State of the Union
- **Size**: 51kb (~40 documents)
- **Content**: 2022 State of the Union address
- **Collection**: `state_of_the_union`
- **Use case**: Quick testing, demonstrations
- **Example queries**:
  - "What did the president say about infrastructure?"
  - "Tell me about economic policy"
  - "What were the main topics discussed?"

### Paul Graham Essays
- **Size**: 1.3mb (~300 documents)
- **Content**: Essays from http://www.paulgraham.com
- **Collection**: `paul_graham_essays`
- **Use case**: Startup/tech content, medium-scale testing
- **Example queries**:
  - "What does Paul Graham say about startups?"
  - "How should founders approach product development?"
  - "What is the importance of growth?"

### Huberman Podcasts
- **Size**: 4.3mb (~1000 documents)
- **Content**: Huberman Lab podcast transcripts
- **Collection**: `huberman_podcasts`
- **Use case**: Health/science content, performance testing
- **Example queries**:
  - "What does Huberman say about sleep optimization?"
  - "How does exercise affect the brain?"
  - "What are the benefits of cold exposure?"

## Scripts

### populate_simple_test_data.py (Recommended)

Creates simple test collections with synthetic data. Perfect for quick testing!

**Usage:**
```bash
# Interactive mode (recommended)
uv run python populate_simple_test_data.py

# Specify custom path
uv run python populate_simple_test_data.py --db-path /path/to/data

# List existing collections
uv run python populate_simple_test_data.py --list
```

**Features:**
- No external dependencies
- Fast embedding computation using sentence-transformers
- 35 total documents across 2 collections
- Technology and business content
- Metadata included

### populate_test_datasets.py (Alternative)

Populates ChromaDB with pre-built datasets from `chroma_datasets` package.

**Usage:**
```bash
# Interactive mode
uv run python populate_test_datasets.py

# Custom path
uv run python populate_test_datasets.py --db-path /path/to/data

# List collections
uv run python populate_test_datasets.py --list
```

**Features:**
- Real-world datasets (State of Union, Paul Graham, Huberman)
- Pre-computed OpenAI embeddings
- 40-1000+ documents
- May require additional dependencies

### start_test_server.py

Starts a ChromaDB server using either the CLI or Docker.

**Usage:**
```bash
# Start with CLI (default)
uv run start_test_server.py

# Custom host/port
uv run start_test_server.py --host 0.0.0.0 --port 9000

# Use Docker
uv run start_test_server.py --docker

# Custom data path
uv run start_test_server.py --db-path /path/to/data
```

**Features:**
- Auto-installs ChromaDB CLI if needed
- Lists available collections
- Tests server connectivity
- Provides next steps

## Complete Workflow Example

### Scenario: Test MeVe with State of the Union

**1. Populate data:**
```bash
cd test_chromadb_server
uv run populate_test_datasets.py
# Select: 1 (State of the Union)
```

**2. Start server:**
```bash
uv run start_test_server.py
# Server starts at http://localhost:8000
```

**3. In Claude Desktop:**
```
User: "List all ChromaDB collections"
Claude: [Uses list_chromadb_collections]
        Shows: state_of_the_union (40 documents)

User: "Load the state_of_the_union collection for MeVe"
Claude: [Uses use_external_collection_for_meve]
        Loaded 40 documents with embeddings

User: "What did the president say about infrastructure?"
Claude: [Uses query_with_meve]
        Returns relevant context from the speech
```

## Advanced Usage

### Using with Custom Data

1. Start the server with an empty database:
```bash
uv run start_test_server.py --db-path ./my_custom_data
```

2. Populate with your own data using Python:
```python
import chromadb

client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection("my_docs")

collection.add(
    documents=["Your document text here", "Another document"],
    ids=["doc1", "doc2"],
    metadatas=[{"source": "custom"}, {"source": "custom"}]
)
```

3. Connect MeVe to your collection:
```
"Connect to http://localhost:8000"
"Load the my_docs collection"
"Query your data with MeVe"
```

### Using with Chroma CLI Directly

If you prefer to use the Chroma CLI directly:

```bash
# Install ChromaDB with CLI
pip install chromadb

# Start server
chroma run --path ./chroma_test_data --host localhost --port 8000
```

### Using Docker Compose

Create `docker-compose.yml` in this folder:

```yaml
version: '3.8'

services:
  chromadb:
    image: chromadb/chroma:latest
    container_name: meve-chromadb-test
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_test_data:/data
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_HTTP_PORT=8000
```

Then run:
```bash
docker-compose up -d
```

## Troubleshooting

### "Database directory not found"
Run `populate_test_datasets.py` first to create the test data.

### "ChromaDB CLI not installed"
The script will prompt to install it automatically, or run:
```bash
pip install chromadb
```

### "Port 8000 already in use"
Either stop the existing service or use a different port:
```bash
uv run start_test_server.py --port 9000
```

### Docker issues
Make sure Docker is running:
```bash
docker ps
```

### Can't connect from MeVe MCP
1. Verify server is running: `curl http://localhost:8000/api/v1/heartbeat`
2. Check CHROMADB_URL in Claude Desktop config
3. Restart Claude Desktop after config changes

## File Structure

```
test_chromadb_server/
├── README.md                    # This file
├── populate_test_datasets.py    # Script to load test data
├── start_test_server.py         # Script to start ChromaDB server
└── chroma_test_data/            # Created by populate script
    └── [ChromaDB database files]
```

## Resources

- [ChromaDB Documentation](https://docs.trychroma.com)
- [Chroma Datasets Package](https://github.com/chroma-core/chroma-datasets)
- [MeVe Framework README](../README.md)
- [MeVe MCP Server README](../MCP_SERVER_README.md)
- [ChromaDB Setup Guide](../CHROMADB_SETUP_GUIDE.md)

## Next Steps

After testing with the test datasets:
1. Try with your own data
2. Experiment with different MeVe parameters (k_init, tau_relevance, etc.)
3. Compare MeVe results with standard RAG using `benchmark_efficiency`
4. Analyze retrieval quality with `analyze_retrieval`

## Support

Issues? Check:
1. Server logs in the terminal
2. Claude Desktop logs (Menu → View Logs)
3. Test connection: `curl http://localhost:8000/api/v1/heartbeat`
