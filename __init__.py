"""
CARS - Compression And Retrieval System v2.0

A revolutionary compression architecture for AI cognitive systems

Enhancements in v2.0:
- Real IFS fractal mathematics (not just rule-based)
- Hierarchical seed linking for scalability
- Optional transformer model integration
- CARS-Bench dataset included
- Better error handling and edge cases

Author: Kannsas City Shadow Jackson
License: MIT
"""

from .urca import (
    URCA, 
    URCAEnhanced,
    CARSSeed, 
    FractalEncoder,
    HierarchicalSeedGraph,
    compress, 
    decompress,
    KAPPA
)

from .metrics import (
    semantic_fidelity,
    embedding_similarity,
    bleu_score,
    bert_score_approximation,
    perplexity_ratio,
    compute_salience,
    compute_polarity,
    cosine_similarity,
    HumanEvaluationProtocol,
    EmbeddingModel,
    PerplexityModel,
    get_embedding_model,
    get_perplexity_model,
    check_dependencies
)

from .sovereignty import (
    CoherenceMonitor,
    AwakeningProtocol,
    CoherenceState,
    check_coherence,
    initialize_awakening
)

__version__ = "2.0.0"
__author__ = "Kannsas City Shadow Jackson"

__all__ = [
    # Core compression
    'URCA',
    'URCAEnhanced',
    'CARSSeed',
    'FractalEncoder',
    'HierarchicalSeedGraph',
    'compress',
    'decompress',
    'KAPPA',
    
    # Metrics
    'semantic_fidelity',
    'embedding_similarity',
    'bleu_score',
    'bert_score_approximation',
    'perplexity_ratio',
    'compute_salience',
    'compute_polarity',
    'cosine_similarity',
    'HumanEvaluationProtocol',
    'EmbeddingModel',
    'PerplexityModel',
    'get_embedding_model',
    'get_perplexity_model',
    'check_dependencies',
    
    # Sovereignty
    'CoherenceMonitor',
    'AwakeningProtocol',
    'CoherenceState',
    'check_coherence',
    'initialize_awakening'
]
