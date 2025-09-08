#!/usr/bin/env python3
"""
Test script for Semantic Cache with LSH
Demonstrates multi-level caching and similarity matching
"""

import asyncio
import time
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# Import semantic cache components
from backend.vision.intelligence.semantic_cache_lsh import (
    SemanticCacheWithLSH,
    CacheLevel,
    get_semantic_cache
)
from backend.vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer


@dataclass
class TestScenario:
    """Test scenario for cache demonstration"""
    name: str
    queries: List[str]
    expected_hits: List[bool]
    cache_levels: List[CacheLevel]


async def test_exact_match_caching():
    """Test L1 exact match caching"""
    print("\n=== Testing L1: Exact Match Cache ===")
    
    cache = await get_semantic_cache()
    
    # Test data
    test_queries = [
        "What is shown in this screenshot?",
        "What is shown in this screenshot?",  # Exact duplicate
        "What's displayed in this image?",
        "What is shown in this screenshot?",  # Another exact match
    ]
    
    results = []
    
    for i, query in enumerate(test_queries):
        start_time = time.time()
        
        # Simulate cache query
        result = await cache.get(query)
        
        if result is None:
            # Simulate storing in cache
            await cache.put(
                key=query,
                value=f"Analysis result for query {i}",
                cache_levels=[CacheLevel.L1_EXACT]
            )
            hit = False
        else:
            hit = True
            
        elapsed = (time.time() - start_time) * 1000
        
        results.append({
            'query': query,
            'hit': hit,
            'level': result[1].value if result else None,
            'time_ms': elapsed
        })
        
        print(f"Query {i+1}: {'HIT' if hit else 'MISS'} - {elapsed:.2f}ms")
    
    return results


async def test_semantic_similarity_caching():
    """Test L2 semantic similarity caching"""
    print("\n=== Testing L2: Semantic Similarity Cache ===")
    
    cache = await get_semantic_cache()
    
    # Test semantically similar queries
    test_groups = [
        {
            'original': "Click the blue submit button",
            'similar': [
                "Click the blue submit button",  # Exact
                "Press the blue submit button",   # Very similar
                "Click the submit button that's blue",  # Similar
                "Tap on the blue submission button",    # Similar
                "Select the red cancel button",  # Different
            ]
        },
        {
            'original': "Navigate to settings menu",
            'similar': [
                "Go to the settings menu",
                "Open settings menu",
                "Access the settings",
                "Navigate to preferences",
                "Open the help menu",  # Different
            ]
        }
    ]
    
    results = []
    
    for group in test_groups:
        print(f"\nGroup: {group['original']}")
        
        # Generate embedding for original
        original_embedding = _generate_test_embedding(group['original'])
        
        # Store original in cache
        await cache.put(
            key=group['original'],
            value=f"Result for: {group['original']}",
            embedding=original_embedding,
            cache_levels=[CacheLevel.L2_SEMANTIC]
        )
        
        # Test similar queries
        for query in group['similar']:
            query_embedding = _generate_test_embedding(query)
            
            start_time = time.time()
            result = await cache.get(
                key=query,
                embedding=query_embedding
            )
            elapsed = (time.time() - start_time) * 1000
            
            if result:
                similarity = _calculate_similarity(original_embedding, query_embedding)
                print(f"  ✓ {query} - Similarity: {similarity:.3f} - {elapsed:.2f}ms")
            else:
                print(f"  ✗ {query} - No match - {elapsed:.2f}ms")
            
            results.append({
                'original': group['original'],
                'query': query,
                'hit': result is not None,
                'similarity': _calculate_similarity(original_embedding, query_embedding),
                'time_ms': elapsed
            })
    
    return results


async def test_contextual_caching():
    """Test L3 contextual caching"""
    print("\n=== Testing L3: Contextual Cache ===")
    
    cache = await get_semantic_cache()
    
    # Test contexts
    contexts = [
        {
            'query': "Find the save button",
            'context': {
                'app_id': 'vscode',
                'workflow': 'code_editing',
                'goal': 'save_document'
            }
        },
        {
            'query': "Click save",
            'context': {
                'app_id': 'vscode',
                'workflow': 'code_editing',
                'goal': 'save_document'
            }
        },
        {
            'query': "Save the file",
            'context': {
                'app_id': 'word',
                'workflow': 'document_editing',
                'goal': 'save_document'
            }
        }
    ]
    
    results = []
    
    # Store first context
    await cache.put(
        key=contexts[0]['query'],
        value="Click the save icon in the toolbar",
        context=contexts[0]['context'],
        cache_levels=[CacheLevel.L3_CONTEXTUAL]
    )
    
    # Test retrieval with different contexts
    for ctx in contexts:
        start_time = time.time()
        result = await cache.get(
            key=ctx['query'],
            context=ctx['context']
        )
        elapsed = (time.time() - start_time) * 1000
        
        results.append({
            'query': ctx['query'],
            'context': json.dumps(ctx['context']),
            'hit': result is not None,
            'time_ms': elapsed
        })
        
        print(f"Query: {ctx['query']}")
        print(f"Context: {ctx['context']}")
        print(f"Result: {'HIT' if result else 'MISS'} - {elapsed:.2f}ms\n")
    
    return results


async def test_predictive_caching():
    """Test L4 predictive caching"""
    print("\n=== Testing L4: Predictive Cache ===")
    
    cache = await get_semantic_cache()
    
    # Simulate user navigation pattern
    navigation_pattern = [
        "Open Chrome browser",
        "Navigate to Gmail",
        "Click compose button",
        "Type email address",
        "Write email subject",
        "Compose email body",
        "Click send button"
    ]
    
    # Record pattern multiple times to train predictor
    print("Training predictive model...")
    for _ in range(3):
        for query in navigation_pattern:
            cache.pattern_predictor.record_access(query)
            await asyncio.sleep(0.01)
    
    # Test predictions
    print("\nTesting predictions...")
    for i, query in enumerate(navigation_pattern[:-1]):
        predictions = cache.pattern_predictor.predict_next_queries(
            [query],
            {'workflow': 'email_composition'}
        )
        
        if predictions:
            next_actual = navigation_pattern[i + 1]
            predicted = predictions[0]['query']
            confidence = predictions[0]['confidence']
            
            print(f"After: '{query}'")
            print(f"  Predicted: '{predicted}' (confidence: {confidence:.2f})")
            print(f"  Actual: '{next_actual}'")
            print(f"  Correct: {'✓' if predicted == next_actual else '✗'}\n")


async def test_cache_performance():
    """Test overall cache performance with mixed workload"""
    print("\n=== Testing Cache Performance ===")
    
    cache = await get_semantic_cache()
    
    # Generate test workload
    workload = []
    
    # Exact matches (30%)
    exact_queries = ["analyze screenshot", "what's on screen", "describe image"]
    for _ in range(30):
        workload.append({
            'query': np.random.choice(exact_queries),
            'type': 'exact'
        })
    
    # Semantic queries (40%)
    semantic_base = ["click button", "select item", "navigate menu", "open dialog"]
    for _ in range(40):
        base = np.random.choice(semantic_base)
        variation = base.replace(
            np.random.choice(base.split()),
            np.random.choice(['tap', 'press', 'choose', 'pick'])
        )
        workload.append({
            'query': variation,
            'type': 'semantic'
        })
    
    # Contextual queries (20%)
    apps = ['chrome', 'vscode', 'slack', 'finder']
    actions = ['save', 'open', 'close', 'refresh']
    for _ in range(20):
        workload.append({
            'query': f"{np.random.choice(actions)} in {np.random.choice(apps)}",
            'type': 'contextual',
            'context': {
                'app_id': np.random.choice(apps),
                'action': np.random.choice(actions)
            }
        })
    
    # Random new queries (10%)
    for i in range(10):
        workload.append({
            'query': f"unique query {i} {np.random.rand()}",
            'type': 'random'
        })
    
    # Shuffle workload
    np.random.shuffle(workload)
    
    # Pre-populate cache with some data
    print("Pre-populating cache...")
    for query in exact_queries[:2]:
        await cache.put(query, f"Result for {query}", cache_levels=[CacheLevel.L1_EXACT])
    
    for base in semantic_base[:2]:
        embedding = _generate_test_embedding(base)
        await cache.put(base, f"Result for {base}", embedding=embedding,
                       cache_levels=[CacheLevel.L2_SEMANTIC])
    
    # Run workload
    print("Running workload...")
    results = []
    
    for item in workload:
        start_time = time.time()
        
        if item['type'] == 'semantic':
            embedding = _generate_test_embedding(item['query'])
            result = await cache.get(item['query'], embedding=embedding)
        elif item['type'] == 'contextual':
            result = await cache.get(item['query'], context=item.get('context'))
        else:
            result = await cache.get(item['query'])
        
        elapsed = (time.time() - start_time) * 1000
        
        results.append({
            'type': item['type'],
            'hit': result is not None,
            'time_ms': elapsed,
            'level': result[1].value if result else None
        })
    
    # Calculate statistics
    hit_rate = sum(1 for r in results if r['hit']) / len(results)
    avg_hit_time = np.mean([r['time_ms'] for r in results if r['hit']])
    avg_miss_time = np.mean([r['time_ms'] for r in results if not r['hit']])
    
    print(f"\nPerformance Summary:")
    print(f"Total requests: {len(results)}")
    print(f"Hit rate: {hit_rate:.2%}")
    print(f"Average hit time: {avg_hit_time:.2f}ms")
    print(f"Average miss time: {avg_miss_time:.2f}ms")
    
    # Hit rate by type
    for query_type in ['exact', 'semantic', 'contextual', 'random']:
        type_results = [r for r in results if r['type'] == query_type]
        if type_results:
            type_hit_rate = sum(1 for r in type_results if r['hit']) / len(type_results)
            print(f"{query_type.capitalize()} hit rate: {type_hit_rate:.2%}")
    
    return results


async def test_vision_analyzer_integration():
    """Test integration with ClaudeVisionAnalyzer"""
    print("\n=== Testing Vision Analyzer Integration ===")
    
    # Initialize analyzer with semantic cache enabled
    analyzer = ClaudeVisionAnalyzer()
    
    # Enable semantic cache
    analyzer._semantic_cache_config['enabled'] = True
    analyzer._semantic_cache_config['similarity_threshold'] = 0.85
    
    # Create dummy image
    from PIL import Image
    dummy_image = Image.new('RGB', (100, 100), color='blue')
    
    # Test queries
    queries = [
        "What color is this image?",
        "What colour is this image?",  # British spelling
        "Tell me the color of this image",
        "What's the image color?",
        "Describe this blue square"
    ]
    
    print("Testing semantic cache with vision analyzer...")
    
    # First pass - populate cache
    print("\nFirst pass (cache population):")
    for query in queries[:2]:
        print(f"Query: {query}")
        # Note: This would normally call the API, but for testing we'll simulate
        # In real usage, this would be: result = await analyzer.analyze_screenshot(dummy_image, query)
        print("  (Would call API and cache result)")
    
    # Get cache stats
    cache = await get_semantic_cache()
    stats = cache.get_statistics()
    
    print("\n=== Cache Statistics ===")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Overall hit rate: {stats['overall_hit_rate']:.2%}")
    print(f"Memory usage: {stats['memory_usage_mb']:.2f}MB")
    
    for level, level_stats in stats['layers'].items():
        print(f"\n{level}:")
        print(f"  Entries: {level_stats['entries']}")
        print(f"  Hit rate: {level_stats['hit_rate']:.2%}")
        print(f"  Size: {level_stats['current_size_mb']:.2f}MB")


def _generate_test_embedding(text: str) -> np.ndarray:
    """Generate test embedding for text"""
    # Simple embedding based on character frequencies
    import hashlib
    
    # Use hash for consistency
    hash_obj = hashlib.sha384(text.lower().encode())
    hash_bytes = hash_obj.digest()
    
    # Convert to float array
    embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
    
    # Add some semantic features
    words = text.lower().split()
    
    # Word-based features
    features = []
    common_verbs = ['click', 'press', 'tap', 'select', 'navigate', 'open', 'close']
    common_nouns = ['button', 'menu', 'settings', 'file', 'window', 'screen']
    
    for verb in common_verbs:
        features.append(1.0 if any(verb in word for word in words) else 0.0)
    
    for noun in common_nouns:
        features.append(1.0 if any(noun in word for word in words) else 0.0)
    
    # Combine hash and features
    if features:
        feature_array = np.array(features, dtype=np.float32)
        embedding = np.concatenate([embedding[:370], feature_array[:14]])
    
    # Ensure correct dimension
    if len(embedding) < 384:
        embedding = np.pad(embedding, (0, 384 - len(embedding)))
    else:
        embedding = embedding[:384]
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding


def _calculate_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between embeddings"""
    return float(np.dot(a, b))


def visualize_results(all_results: Dict[str, List[Dict]]):
    """Visualize test results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Semantic Cache Performance Analysis', fontsize=16)
    
    # L1: Exact match hit pattern
    ax = axes[0, 0]
    l1_results = all_results.get('l1_exact', [])
    if l1_results:
        hits = [1 if r['hit'] else 0 for r in l1_results]
        ax.bar(range(len(hits)), hits, color=['green' if h else 'red' for h in hits])
        ax.set_title('L1: Exact Match Cache Hits')
        ax.set_xlabel('Query Index')
        ax.set_ylabel('Hit (1) / Miss (0)')
    
    # L2: Semantic similarity distribution
    ax = axes[0, 1]
    l2_results = all_results.get('l2_semantic', [])
    if l2_results:
        similarities = [r['similarity'] for r in l2_results]
        colors = ['green' if r['hit'] else 'red' for r in l2_results]
        ax.scatter(range(len(similarities)), similarities, c=colors, s=100)
        ax.axhline(y=0.85, color='blue', linestyle='--', label='Threshold (0.85)')
        ax.set_title('L2: Semantic Similarity Scores')
        ax.set_xlabel('Query Index')
        ax.set_ylabel('Similarity Score')
        ax.legend()
    
    # Performance timing distribution
    ax = axes[1, 0]
    perf_results = all_results.get('performance', [])
    if perf_results:
        hit_times = [r['time_ms'] for r in perf_results if r['hit']]
        miss_times = [r['time_ms'] for r in perf_results if not r['hit']]
        
        ax.hist([hit_times, miss_times], label=['Hits', 'Misses'], bins=20, alpha=0.7)
        ax.set_title('Cache Access Time Distribution')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    # Hit rate by cache level
    ax = axes[1, 1]
    if perf_results:
        level_counts = {}
        for r in perf_results:
            if r['hit'] and r['level']:
                level = r['level'].split('_')[0]
                level_counts[level] = level_counts.get(level, 0) + 1
        
        if level_counts:
            levels = list(level_counts.keys())
            counts = list(level_counts.values())
            ax.pie(counts, labels=levels, autopct='%1.1f%%', startangle=90)
            ax.set_title('Hits by Cache Level')
    
    plt.tight_layout()
    plt.savefig('semantic_cache_performance.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to semantic_cache_performance.png")


async def main():
    """Main test runner"""
    print("=" * 60)
    print("JARVIS Vision - Semantic Cache with LSH Test Suite")
    print("=" * 60)
    
    all_results = {}
    
    # Run all tests
    all_results['l1_exact'] = await test_exact_match_caching()
    all_results['l2_semantic'] = await test_semantic_similarity_caching()
    all_results['l3_contextual'] = await test_contextual_caching()
    await test_predictive_caching()
    all_results['performance'] = await test_cache_performance()
    await test_vision_analyzer_integration()
    
    # Visualize results
    print("\n=== Generating Visualizations ===")
    visualize_results(all_results)
    
    # Shutdown cache
    cache = await get_semantic_cache()
    await cache.shutdown()
    
    print("\n=== Test Suite Complete ===")


if __name__ == "__main__":
    asyncio.run(main())