# MeVe MCP Server - Quick Start Guide

## 1. Start ChromaDB (Docker)

```bash
docker run -d --name chromadb -v ./chroma-data:/data -p 8000:8000 chromadb/chroma
```

## 2. Configure Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "meve-rag": {
      "command": "/Users/YOUR_USERNAME/.local/bin/uv",
      "args": [
        "--directory",
        "/absolute/path/to/Meve-framework",
        "run",
        "meve_mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "/absolute/path/to/Meve-framework",
        "CHROMADB_URL": "http://localhost:8000"
      }
    }
  }
}
```

**Find your UV path:**
```bash
which uv
# Use the output in the "command" field
```

## 3. Restart Claude Desktop

Close and reopen Claude Desktop to load the MCP server.

## 4. Use in Claude

**List collections:**
```
"List all available ChromaDB collections"
```

**Connect to a collection:**
```
"Connect to the collection named 'my_documents'"
```

**Query with MeVe:**
```
"Search for information about renewable energy using the MeVe pipeline"
```

## Connection Methods

### Option A: Environment Variable (Easiest)

Set in Claude Desktop config:
```json
"env": {
  "CHROMADB_URL": "http://localhost:8000"
}
```

### Option B: MCP Tool

In Claude Desktop:
```
"Connect to ChromaDB at http://localhost:8000"
```

### Option C: Remote Server

```json
"env": {
  "CHROMADB_URL": "http://your-server.com:8000"
}
```

Or:
```
"Connect to ChromaDB at http://your-server.com:8000"
```

## Supported URL Formats

- `http://localhost:8000`
- `https://my-server.com:8080`
- `my-server.com` (assumes port 8000)
- `192.168.1.100:9000`

## Common Commands

| Task | Command in Claude |
|------|-------------------|
| List collections | "List all ChromaDB collections" |
| Connect to collection | "Load the 'my_docs' collection for MeVe" |
| Query | "Search for [topic] using MeVe pipeline" |
| Check status | "Show MeVe pipeline status" |
| Configure | "Set MeVe k_init to 30 and tau_relevance to 0.6" |

## Troubleshooting

**Can't find UV:**
```bash
which uv
# Copy the full path to claude_desktop_config.json
```

**ChromaDB not connecting:**
```bash
# Check if running
docker ps | grep chroma

# Test connection
curl http://localhost:8000/api/v1/heartbeat
```

**No collections found:**
```python
# Add documents to ChromaDB first
import chromadb
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection("my_docs")
collection.add(
    documents=["Your document text here"],
    ids=["doc1"]
)
```

## Full Documentation

- [Complete ChromaDB Setup Guide](./CHROMADB_SETUP_GUIDE.md)
- [MCP Server README](./MCP_SERVER_README.md)
- [Implementation Summary](./MCP_IMPLEMENTATION_SUMMARY.md)

## Support

- Test server: `uv run meve_mcp_server.py`
- Check logs: Claude Desktop → Menu → View Logs
- ChromaDB logs: `docker logs chromadb`
