#!/usr/bin/env python3
"""
Test script for MeVe MCP Server functionality.
Tests all tools, resources, and prompts to ensure proper operation.
"""

import json
import asyncio
import sys
from meve_mcp_server import (
    mcp, 
    server_state, 
    initialize_meve_data,
    query_with_meve,
    analyze_retrieval,
    execute_phase1_only,
    execute_phase5_only,
    configure_pipeline,
    get_pipeline_status,
    benchmark_efficiency,
    index_documents,
    get_config,
    get_stats,
    get_collections
)

async def test_initialization():
    """Test data initialization."""
    print("🔄 Testing data initialization...")
    try:
        await initialize_meve_data()
        if server_state.is_initialized:
            print(f"✅ Initialization successful: {len(server_state.vector_store)} chunks loaded")
            return True
        else:
            print("❌ Initialization failed")
            return False
    except Exception as e:
        print(f"❌ Initialization error: {str(e)}")
        return False

async def test_resources():
    """Test MCP resources."""
    print("\n📄 Testing MCP Resources...")
    
    # Test config resource
    try:
        config = await get_config()
        config_data = json.loads(config)
        print(f"✅ Config resource: {config_data.get('k_init', 'N/A')} k_init")
    except Exception as e:
        print(f"❌ Config resource error: {str(e)}")
    
    # Test stats resource
    try:
        stats = await get_stats()
        stats_data = json.loads(stats)
        print(f"✅ Stats resource: {stats_data.get('total_queries', 0)} total queries")
    except Exception as e:
        print(f"❌ Stats resource error: {str(e)}")
    
    # Test collections resource
    try:
        collections = await get_collections()
        collections_data = json.loads(collections)
        vector_count = collections_data.get('vector_store', {}).get('chunk_count', 0)
        print(f"✅ Collections resource: {vector_count} vector chunks")
    except Exception as e:
        print(f"❌ Collections resource error: {str(e)}")

async def test_basic_tools():
    """Test basic MCP tools."""
    print("\n🔧 Testing Basic Tools...")
    
    test_query = "What is artificial intelligence?"
    
    # Test pipeline status
    try:
        status = await get_pipeline_status()
        status_data = json.loads(status)
        print(f"✅ Pipeline status: {status_data.get('server_status', 'unknown')}")
    except Exception as e:
        print(f"❌ Pipeline status error: {str(e)}")
    
    # Test configuration update
    try:
        config_result = await configure_pipeline(k_init=15, tau_relevance=0.6)
        config_data = json.loads(config_result)
        if config_data.get('status') == 'success':
            print(f"✅ Configuration update: k_init={config_data['new_config']['k_init']}")
        else:
            print(f"❌ Configuration update failed: {config_data.get('message', 'unknown error')}")
    except Exception as e:
        print(f"❌ Configuration update error: {str(e)}")
    
    # Test Phase 1 only
    try:
        phase1_result = await execute_phase1_only(test_query, k_init=10)
        phase1_data = json.loads(phase1_result)
        candidates = phase1_data.get('results', {}).get('candidates_retrieved', 0)
        print(f"✅ Phase 1 execution: {candidates} candidates retrieved")
    except Exception as e:
        print(f"❌ Phase 1 execution error: {str(e)}")

async def test_phase5_tool():
    """Test Phase 5 token budgeting tool."""
    print("\n🎯 Testing Phase 5 Tool...")
    
    # Create sample chunks for testing
    sample_chunks = [
        {
            "content": "Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.",
            "doc_id": "ai_definition",
            "relevance_score": 0.95
        },
        {
            "content": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence.",
            "doc_id": "ml_definition", 
            "relevance_score": 0.87
        },
        {
            "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
            "doc_id": "dl_definition",
            "relevance_score": 0.82
        }
    ]
    
    try:
        chunks_json = json.dumps(sample_chunks)
        phase5_result = await execute_phase5_only(chunks_json, t_max=200)
        phase5_data = json.loads(phase5_result)
        
        results = phase5_data.get('results', {})
        final_chunks = results.get('final_chunks', 0)
        efficiency = results.get('token_efficiency', 'N/A')
        
        print(f"✅ Phase 5 execution: {final_chunks} final chunks, {efficiency} efficiency")
        
    except Exception as e:
        print(f"❌ Phase 5 execution error: {str(e)}")

async def test_full_pipeline():
    """Test the complete MeVe pipeline."""
    print("\n🚀 Testing Full MeVe Pipeline...")
    
    test_queries = [
        "What is machine learning?",
        "How does neural network training work?", 
        "What are the applications of AI?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        try:
            result = await query_with_meve(
                query=query,
                k_init=15,
                tau_relevance=0.5,
                t_max=300
            )
            
            result_data = json.loads(result)
            
            if result_data.get('status') == 'success':
                context_length = result_data.get('final_context_length', 0)
                print(f"✅ Query {i}: {context_length} chars final context")
            else:
                error = result_data.get('error', 'unknown error')
                print(f"❌ Query {i} failed: {error}")
                
        except Exception as e:
            print(f"❌ Query {i} error: {str(e)}")

async def test_analysis_tool():
    """Test the retrieval analysis tool."""
    print("\n🔍 Testing Retrieval Analysis...")
    
    try:
        analysis = await analyze_retrieval(
            query="What is deep learning?",
            k_init=10
        )
        
        analysis_data = json.loads(analysis)
        phases = analysis_data.get('phases', {})
        
        # Check each phase
        for phase_name, phase_data in phases.items():
            status = phase_data.get('status', 'unknown')
            description = phase_data.get('description', 'No description')
            print(f"   {phase_name}: {status} - {description[:50]}...")
        
        summary = analysis_data.get('summary', {})
        if summary:
            pipeline_success = summary.get('pipeline_success', False)
            final_chunks = summary.get('final_chunks_count', 0)
            print(f"✅ Analysis complete: {final_chunks} final chunks, success={pipeline_success}")
        else:
            print(f"✅ Analysis complete: {len(phases)} phases analyzed")
            
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")

async def test_indexing():
    """Test document indexing functionality."""
    print("\n📚 Testing Document Indexing...")
    
    sample_documents = [
        "The Internet of Things (IoT) describes the network of physical objects that are embedded with sensors, software, and other technologies for the purpose of connecting and exchanging data with other devices and systems over the internet.",
        
        "Blockchain technology is a decentralized, distributed ledger that records the provenance of a digital asset. By inherent design, the data on a blockchain is unable to be modified, which makes it a legitimate disruptor for industries like payments, cybersecurity and healthcare."
    ]
    
    try:
        indexing_result = await index_documents(
            documents=sample_documents,
            collection_name="test_collection",
            metadata={"source": "test_data", "type": "technology"}
        )
        
        indexing_data = json.loads(indexing_result)
        
        if indexing_data.get('indexing_status') == 'success':
            stats = indexing_data.get('statistics', {})
            total_chunks = stats.get('total_chunks', 0)
            total_docs = stats.get('total_documents', 0)
            print(f"✅ Indexing: {total_chunks} chunks from {total_docs} documents")
        else:
            error = indexing_data.get('message', 'unknown error')
            print(f"❌ Indexing failed: {error}")
            
    except Exception as e:
        print(f"❌ Indexing error: {str(e)}")

async def test_benchmarking():
    """Test efficiency benchmarking."""
    print("\n📊 Testing Efficiency Benchmarking...")
    
    try:
        benchmark_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?"
        ]
        
        benchmark_result = await benchmark_efficiency(
            queries=benchmark_queries,
            collection_name="test"
        )
        
        benchmark_data = json.loads(benchmark_result)
        
        summary = benchmark_data.get('summary', {})
        if summary:
            chunk_reduction = summary.get('avg_chunk_reduction', 'N/A')
            token_reduction = summary.get('avg_token_reduction', 'N/A')
            print(f"✅ Benchmark: {chunk_reduction} chunk reduction, {token_reduction} token reduction")
        else:
            results_count = len(benchmark_data.get('results', []))
            print(f"✅ Benchmark: {results_count} queries tested")
            
    except Exception as e:
        print(f"❌ Benchmark error: {str(e)}")

async def run_all_tests():
    """Run comprehensive test suite."""
    print("🧪 MeVe MCP Server Test Suite")
    print("=" * 50)
    
    # Initialize first
    if not await test_initialization():
        print("\n❌ Cannot proceed without data initialization")
        return
    
    # Test all components
    await test_resources()
    await test_basic_tools()
    await test_phase5_tool()
    await test_indexing()
    await test_full_pipeline()
    await test_analysis_tool()
    await test_benchmarking()
    
    print("\n" + "=" * 50)
    print("🎉 Test suite completed!")
    
    # Final status
    try:
        final_status = await get_pipeline_status()
        status_data = json.loads(final_status)
        stats = status_data.get('statistics', {})
        total_queries = stats.get('total_queries', 0)
        successful_queries = stats.get('successful_queries', 0)
        
        print(f"📈 Final Stats: {successful_queries}/{total_queries} successful queries")
        
    except Exception as e:
        print(f"❌ Final status error: {str(e)}")

if __name__ == "__main__":
    # Run the test suite
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n⏹️  Test suite interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {str(e)}")
        sys.exit(1)