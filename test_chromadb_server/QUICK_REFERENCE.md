# MeVe ChromaDB Test Server - Quick Reference

## 🚀 3-Step Setup

```bash
# 1. Populate test data
cd test_chromadb_server
uv run python populate_simple_test_data.py
# Select: all

# 2. Start server
uv run python start_test_server.py

# 3. Configure Claude Desktop
# Edit: ~/Library/Application Support/Claude/claude_desktop_config.json
# Add: "CHROMADB_URL": "http://localhost:8000" in env section
```

## 📋 Commands Cheat Sheet

| Task | Command |
|------|---------|
| Create test data | `uv run python populate_simple_test_data.py` |
| List collections | `uv run python populate_simple_test_data.py --list` |
| Start server (CLI) | `uv run python start_test_server.py` |
| Start server (Docker) | `uv run python start_test_server.py --docker` |
| Check server health | `curl http://localhost:8000/api/v1/heartbeat` |
| Stop Docker server | `docker stop meve-chromadb-test && docker rm meve-chromadb-test` |

## 💬 Claude Desktop Usage

| Action | What to Say in Claude |
|--------|----------------------|
| List collections | "List all ChromaDB collections" |
| Load collection | "Load the tech_articles collection" |
| Query with MeVe | "Search for AI using MeVe pipeline" |
| Check status | "Show MeVe pipeline status" |
| Direct query | "Query ChromaDB for machine learning in tech_articles collection" |

## 📊 Test Collections

### tech_articles (20 documents)
- Topics: AI, cloud, blockchain, IoT, cybersecurity
- Example queries:
  - "What is artificial intelligence?"
  - "Explain blockchain technology"
  - "How does cloud computing work?"

### business_concepts (15 documents)
- Topics: Startups, MVP, growth, venture capital
- Example queries:
  - "What is product-market fit?"
  - "Explain lean startup methodology"
  - "How do network effects work?"

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Database not found" | Run `populate_simple_test_data.py` first |
| "Port 8000 in use" | Use `--port 9000` when starting server |
| "Can't connect from MCP" | Check `CHROMADB_URL` in Claude config, restart Claude Desktop |
| Server won't start | Install ChromaDB: `pip install chromadb` |

## 🎯 Complete Test Workflow

1. **Setup** (one time):
   ```bash
   cd test_chromadb_server
   uv run python populate_simple_test_data.py  # Select 'all'
   ```

2. **Start server**:
   ```bash
   uv run python start_test_server.py
   ```

3. **Configure Claude** (one time):
   ```json
   "env": {
     "CHROMADB_URL": "http://localhost:8000"
   }
   ```

4. **Test in Claude**:
   ```
   "List ChromaDB collections"
   → Shows: tech_articles (20 docs), business_concepts (15 docs)

   "Load tech_articles collection"
   → Loaded 20 documents with embeddings

   "What is blockchain technology?"
   → MeVe pipeline returns relevant context
   ```

## 📁 File Structure

```
test_chromadb_server/
├── README.md                       # Full documentation
├── QUICK_REFERENCE.md              # This file
├── populate_simple_test_data.py    # Create synthetic test data
├── populate_test_datasets.py       # Import chroma_datasets
├── start_test_server.py            # Start ChromaDB server
└── chroma_test_data/               # Database (created by scripts)
```

## 🔗 Links

- Full README: [README.md](./README.md)
- MCP Server: [../MCP_SERVER_README.md](../MCP_SERVER_README.md)
- ChromaDB Setup: [../CHROMADB_SETUP_GUIDE.md](../CHROMADB_SETUP_GUIDE.md)
- Quick Start: [../QUICK_START.md](../QUICK_START.md)
