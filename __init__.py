"""
CARS - Compression And Retrieval System v2.1

A semantic compression framework for AI cognitive systems.

PRIORITY ORDER:
1. SEMANTIC FIDELITY - No content loss, ever
2. COMPRESSION RATIO - Minimize size while preserving meaning  
3. SPEED - Not a priority; correctness over performance

Key Fix in v2.1:
- Restored full content preservation (v2.0 broke this)
- Seeds store full_text to guarantee zero loss
- Conservative segmentation prevents dropping content

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

__version__ = "2.1.0"
__author__ = "Kannsas City Shadow Jackson"

# DESIGN PHILOSOPHY:
# Speed is NOT important. Fidelity is everything.
# A compression algorithm that loses content is useless.
# We optimize for zero information loss, not milliseconds.

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
