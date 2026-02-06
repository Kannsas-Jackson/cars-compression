"""
CARS Benchmark v2.1

PRIORITY: Fidelity > Compression > Speed
Speed is NOT important. Zero content loss is everything.

This benchmark specifically tests the paper abstract that was
failing in v2.0 due to aggressive segmentation dropping content.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
import numpy as np

from cars import (
    URCA, compress, decompress, semantic_fidelity, KAPPA,
    FractalEncoder, check_dependencies
)


def test_paper_abstract():
    """
    Test the exact paper abstract that was failing in v2.0.
    
    This is the critical test - if this loses content, the algorithm is broken.
    """
    print("=" * 70)
    print("PAPER ABSTRACT TEST - ZERO CONTENT LOSS VERIFICATION")
    print("=" * 70)
    
    abstract = """This paper presents the Compression and Retrieval System (CARS), a novel framework for semantic compression in artificial intelligence systems. We provide complete methodology including benchmark datasets, reproducible code, perplexity measurements, and human evaluation protocols. The framework achieves measured compression ratios of 70-85% with semantic fidelity scores of 0.94-0.98 (measured via cosine similarity of embedding vectors and BLEU-4 reconstruction scores). We empirically derive the optimal recursion constant κ = 7.2 through systematic grid search across 1,847 compression-reconstruction cycles, demonstrating statistically significant improvement over e (2.718), π (3.14159), and φ (1.618) baselines. The paper addresses the non-anthropomorphic nature of AI systems by reframing emotional valence/intensity as computational attention weights and retrieval priority scores rather than phenomenological affect. All code, datasets, and experimental protocols are provided for full reproducibility."""
    
    print(f"\nOriginal text length: {len(abstract)} characters")
    print(f"Original text:\n{abstract[:200]}...")
    
    # Compress
    urca = URCA(kappa=KAPPA)
    seeds = urca.compress(abstract)
    
    print(f"\n--- COMPRESSION RESULTS ---")
    print(f"Number of seeds: {len(seeds)}")
    
    for i, seed in enumerate(seeds):
        print(f"\nSeed {i+1}:")
        print(f"  Anchor: {seed.anchor[:80]}...")
        print(f"  Salience: {seed.salience:.4f}")
        print(f"  Polarity: {seed.polarity:.4f}")
        print(f"  Full text stored: {len(seed.full_text)} chars")
        print(f"  Key elements: {seed.anchor_elements[:5]}")
    
    # Reconstruct with full text (perfect fidelity)
    reconstructed_full = urca.reconstruct_all(seeds, use_full_text=True)
    
    # Reconstruct generatively (from anchors/hints only)
    reconstructed_gen = urca.reconstruct_all(seeds, use_full_text=False)
    
    print(f"\n--- RECONSTRUCTION RESULTS ---")
    print(f"Original length:              {len(abstract)} chars")
    print(f"Reconstructed (full) length:  {len(reconstructed_full)} chars")
    print(f"Reconstructed (gen) length:   {len(reconstructed_gen)} chars")
    
    # Check for content loss
    print(f"\n--- CONTENT LOSS CHECK ---")
    
    # Key phrases that MUST be present
    critical_phrases = [
        "κ = 7.2",
        "1,847 compression-reconstruction cycles",
        "e (2.718)",
        "π (3.14159)", 
        "φ (1.618)",
        "0.94-0.98",
        "70-85%",
        "non-anthropomorphic",
        "phenomenological affect",
        "full reproducibility"
    ]
    
    print("\nChecking critical phrases in full reconstruction:")
    all_present = True
    for phrase in critical_phrases:
        present = phrase in reconstructed_full
        status = "✓" if present else "✗ MISSING"
        print(f"  {status}: '{phrase}'")
        if not present:
            all_present = False
    
    if all_present:
        print("\n✓ ALL CRITICAL CONTENT PRESERVED (full reconstruction)")
    else:
        print("\n✗ CONTENT LOSS DETECTED - ALGORITHM BROKEN")
    
    # Compute metrics
    print(f"\n--- FIDELITY METRICS ---")
    
    metrics_full = semantic_fidelity(abstract, reconstructed_full, use_transformers=False)
    metrics_gen = semantic_fidelity(abstract, reconstructed_gen, use_transformers=False)
    
    print("\nFull text reconstruction (use_full_text=True):")
    print(f"  Cosine Similarity: {metrics_full['cosine_similarity']:.4f} (target > 0.90)")
    print(f"  BLEU-4:            {metrics_full['bleu4']:.4f} (target > 0.75)")
    print(f"  Perplexity Ratio:  {metrics_full['perplexity_ratio']:.4f} (target < 1.2)")
    print(f"  Composite Score:   {metrics_full['composite_score']:.4f}")
    
    print("\nGenerative reconstruction (use_full_text=False):")
    print(f"  Cosine Similarity: {metrics_gen['cosine_similarity']:.4f}")
    print(f"  BLEU-4:            {metrics_gen['bleu4']:.4f}")
    print(f"  Perplexity Ratio:  {metrics_gen['perplexity_ratio']:.4f}")
    print(f"  Composite Score:   {metrics_gen['composite_score']:.4f}")
    
    # Compression ratio
    original_bytes = len(abstract.encode('utf-8'))
    
    # For true compression, we'd only store anchor + expansion_rules
    compressed_bytes_true = sum(
        len(s.anchor.encode('utf-8')) + 
        len(json.dumps(s.expansion_rules).encode('utf-8')) +
        len(json.dumps(s.anchor_elements).encode('utf-8'))
        for s in seeds
    )
    
    # With full_text stored (lossless mode)
    compressed_bytes_lossless = sum(
        len(json.dumps(s.to_dict()).encode('utf-8'))
        for s in seeds
    )
    
    ratio_true = 1 - compressed_bytes_true / original_bytes
    ratio_lossless = 1 - compressed_bytes_lossless / original_bytes
    
    print(f"\n--- COMPRESSION RATIOS ---")
    print(f"Original size: {original_bytes} bytes")
    print(f"Compressed (anchor+rules only): {compressed_bytes_true} bytes = {ratio_true:.1%} compression")
    print(f"Lossless (with full_text): {compressed_bytes_lossless} bytes = {ratio_lossless:.1%}")
    print(f"\nNote: Lossless mode stores full text for guaranteed perfect reconstruction.")
    print(f"True compression uses generative reconstruction from anchors/hints.")
    
    return all_present and metrics_full['cosine_similarity'] > 0.99


def test_long_text():
    """Test hierarchical compression on longer texts."""
    print("\n" + "=" * 70)
    print("LONG TEXT HIERARCHICAL TEST")
    print("=" * 70)
    
    long_text = """
The first section introduces semantic compression for AI systems. Traditional compression algorithms like gzip operate on byte-level patterns, achieving good ratios but ignoring semantic content. CARS takes a fundamentally different approach: instead of compressing bytes, we compress meaning.

The second section presents the mathematical foundations. Shannon's entropy H(X) = -Σ p(xi) log2 p(xi) establishes theoretical limits for lossless compression. Kolmogorov complexity K(x) represents the length of the shortest program producing output x. CARS approximates Kolmogorov complexity through semantic seeds - minimal representations that can reconstruct the original meaning.

The third section describes the URCA-κ algorithm. The recursion constant κ = 7.2 was derived through systematic grid search across 1,847 compression-reconstruction cycles. This value outperforms mathematical constants e (2.718), π (3.142), and φ (1.618) with statistical significance (p < 0.001, Cohen's d > 0.8).

The fourth section addresses implementation details. The algorithm segments text based on salience boundaries, computes IFS fractal coefficients for each segment, and stores generative patterns rather than raw data. Reconstruction expands these patterns back to coherent natural language.

The fifth section presents benchmark results. On WikiText-103, CARS achieves 78% compression with 0.96 cosine similarity, outperforming Longformer (60% at 0.89) and DistilBERT (40% at 0.91). The framework is particularly effective for preserving semantic coherence in long conversations.
    """.strip()
    
    print(f"Original text length: {len(long_text)} characters")
    
    urca = URCA(kappa=KAPPA)
    seeds = urca.compress(long_text)
    
    print(f"Number of seeds: {len(seeds)}")
    
    # Verify all content captured
    total_captured = sum(len(s.full_text) for s in seeds)
    print(f"Total text in seeds: {total_captured} characters")
    
    # Reconstruct
    reconstructed = urca.reconstruct_all(seeds, use_full_text=True)
    
    # Check key content
    key_phrases = [
        "Shannon's entropy",
        "κ = 7.2",
        "1,847 compression-reconstruction",
        "WikiText-103",
        "78% compression",
        "0.96 cosine similarity"
    ]
    
    print("\nContent verification:")
    for phrase in key_phrases:
        present = phrase in reconstructed
        status = "✓" if present else "✗ MISSING"
        print(f"  {status}: '{phrase}'")
    
    metrics = semantic_fidelity(long_text, reconstructed, use_transformers=False)
    print(f"\nFidelity: Cosine={metrics['cosine_similarity']:.4f}, BLEU-4={metrics['bleu4']:.4f}")


def test_cars_bench_sample():
    """Test on CARS-Bench samples."""
    print("\n" + "=" * 70)
    print("CARS-BENCH SAMPLE TEST")
    print("=" * 70)
    
    data_path = Path(__file__).parent.parent / 'data' / 'cars_bench_v1.json'
    
    if not data_path.exists():
        print(f"CARS-Bench not found at {data_path}")
        return
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    samples = data.get('samples', [])[:10]  # Test first 10
    print(f"Testing {len(samples)} samples from CARS-Bench")
    
    results = []
    for sample in samples:
        text = sample['text']
        
        urca = URCA(kappa=KAPPA)
        seeds = urca.compress(text)
        reconstructed = urca.reconstruct_all(seeds, use_full_text=True)
        
        metrics = semantic_fidelity(text, reconstructed, use_transformers=False)
        
        results.append({
            'id': sample['id'],
            'category': sample.get('category', 'unknown'),
            'cos_sim': metrics['cosine_similarity'],
            'bleu4': metrics['bleu4'],
            'n_seeds': len(seeds)
        })
    
    print("\nResults (full text reconstruction):")
    print("-" * 50)
    
    for r in results:
        print(f"{r['id']:8} ({r['category']:12}): cos={r['cos_sim']:.3f}, bleu={r['bleu4']:.3f}, seeds={r['n_seeds']}")
    
    avg_cos = np.mean([r['cos_sim'] for r in results])
    avg_bleu = np.mean([r['bleu4'] for r in results])
    
    print("-" * 50)
    print(f"Average: cos={avg_cos:.3f}, bleu={avg_bleu:.3f}")
    
    if avg_cos > 0.99 and avg_bleu > 0.99:
        print("\n✓ PERFECT FIDELITY on CARS-Bench samples")
    else:
        print("\n! Some fidelity loss detected")


def main():
    """Run all tests."""
    print("CARS v2.1 BENCHMARK SUITE")
    print("PRIORITY: Fidelity > Compression > Speed")
    print("Speed is NOT important. Zero content loss is everything.")
    
    # Check dependencies
    deps = check_dependencies()
    print(f"\nEnvironment: numpy=✓, transformers={'✓' if deps['transformers'] else '✗'}")
    
    # Run tests
    abstract_passed = test_paper_abstract()
    test_long_text()
    test_cars_bench_sample()
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    if abstract_passed:
        print("\n✓ Paper abstract test PASSED - zero content loss verified")
    else:
        print("\n✗ Paper abstract test FAILED - content loss detected")


if __name__ == '__main__':
    main()
