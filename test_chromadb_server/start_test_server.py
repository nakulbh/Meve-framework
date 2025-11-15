#!/usr/bin/env python3
"""
Start a ChromaDB server for testing the MeVe MCP integration.

This script starts a ChromaDB server using the official CLI, serving the test data
populated by populate_test_datasets.py.
"""

import subprocess
import sys
import os
import time
import requests
from pathlib import Path


def check_chroma_installed():
    """Check if ChromaDB CLI is installed."""
    try:
        result = subprocess.run(
            ["chroma", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def install_chroma():
    """Install ChromaDB with CLI support."""
    print("📦 ChromaDB CLI not found. Installing...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "chromadb[cli]"],
            check=True
        )
        print("✅ ChromaDB CLI installed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install ChromaDB CLI: {e}")
        return False


def start_chromadb_server(
    db_path: str = "./chroma_test_data",
    host: str = "localhost",
    port: int = 8000
):
    """
    Start ChromaDB server using the official CLI.

    Args:
        db_path: Path to ChromaDB data directory
        host: Host to bind to
        port: Port to listen on
    """
    print(f"\n{'='*60}")
    print("ChromaDB Test Server")
    print(f"{'='*60}\n")

    # Check if data directory exists
    db_path_abs = os.path.abspath(db_path)
    if not os.path.exists(db_path):
        print(f"⚠️  Database directory not found: {db_path_abs}")
        print("\nPlease run populate_test_datasets.py first:")
        print(f"  python populate_test_datasets.py --db-path {db_path}\n")
        sys.exit(1)

    # Check if ChromaDB CLI is installed
    if not check_chroma_installed():
        print("⚠️  ChromaDB CLI is not installed.")
        response = input("Install it now? (y/n): ").strip().lower()
        if response == 'y':
            if not install_chroma():
                sys.exit(1)
        else:
            print("\nPlease install ChromaDB CLI manually:")
            print("  pip install chromadb")
            sys.exit(1)

    # List collections in the database
    print(f"📁 Database path: {db_path_abs}")
    print(f"🌐 Server will start at: http://{host}:{port}\n")

    try:
        import chromadb
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()

        if collections:
            print("📚 Available collections:")
            for col in collections:
                count = col.count()
                print(f"  • {col.name}: {count} documents")
        else:
            print("⚠️  No collections found in database.")
            print("Run populate_test_datasets.py to add test data.\n")
    except Exception as e:
        print(f"⚠️  Could not list collections: {e}")

    print(f"\n{'='*60}")
    print("Starting ChromaDB server...")
    print(f"{'='*60}\n")

    # Start the ChromaDB server using CLI
    cmd = [
        "chroma",
        "run",
        "--path", db_path,
        "--host", host,
        "--port", str(port)
    ]

    print(f"Command: {' '.join(cmd)}\n")
    print("Server is running. Press Ctrl+C to stop.\n")
    print("Next steps:")
    print("  1. Configure Claude Desktop with: CHROMADB_URL=http://localhost:8000")
    print("  2. Restart Claude Desktop")
    print("  3. In Claude, use: 'List ChromaDB collections'")
    print("  4. Then: 'Load the state_of_the_union collection'")
    print("  5. Query: 'Search for infrastructure using MeVe'\n")
    print(f"{'='*60}\n")

    try:
        # Start the server
        process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        # Wait a bit for server to start
        time.sleep(2)

        # Test the connection
        try:
            response = requests.get(f"http://{host}:{port}/api/v1/heartbeat", timeout=2)
            if response.status_code == 200:
                print(f"\n✅ Server is running at http://{host}:{port}")
                print(f"   Heartbeat: {response.json()}\n")
        except Exception as e:
            print(f"\n⚠️  Could not verify server status: {e}")

        # Keep the server running
        process.wait()

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping ChromaDB server...")
        process.terminate()
        process.wait()
        print("✅ Server stopped.\n")
    except Exception as e:
        print(f"\n❌ Server error: {e}\n")
        sys.exit(1)


def start_with_docker(
    db_path: str = "./chroma_test_data",
    port: int = 8000
):
    """
    Alternative: Start ChromaDB using Docker.

    Args:
        db_path: Path to ChromaDB data directory
        port: Port to expose
    """
    db_path_abs = os.path.abspath(db_path)

    print(f"\n{'='*60}")
    print("ChromaDB Test Server (Docker)")
    print(f"{'='*60}\n")

    if not os.path.exists(db_path):
        print(f"⚠️  Database directory not found: {db_path_abs}")
        print("\nPlease run populate_test_datasets.py first.\n")
        sys.exit(1)

    print(f"📁 Database path: {db_path_abs}")
    print(f"🌐 Server will start at: http://localhost:{port}\n")

    cmd = [
        "docker", "run",
        "--name", "meve-chromadb-test",
        "-d",
        "-v", f"{db_path_abs}:/data",
        "-p", f"{port}:8000",
        "chromadb/chroma"
    ]

    print(f"Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ ChromaDB server started in Docker!")
        print(f"   Access at: http://localhost:{port}")
        print("\nTo stop the server:")
        print("  docker stop meve-chromadb-test")
        print("  docker rm meve-chromadb-test\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Docker container: {e}")
        print("\nMake sure Docker is running and try again.\n")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start ChromaDB test server")
    parser.add_argument(
        "--db-path",
        default="./chroma_test_data",
        help="Path to ChromaDB data directory (default: ./chroma_test_data)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Use Docker instead of CLI"
    )

    args = parser.parse_args()

    if args.docker:
        start_with_docker(args.db_path, args.port)
    else:
        start_chromadb_server(args.db_path, args.host, args.port)
