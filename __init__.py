"""
CARS - Compression And Retrieval System
A revolutionary compression architecture for AI cognitive systems

Author: Kannsas City Shadow Jackson
License: MIT
"""

from .urca import URCA, CARSSeed, compress, decompress
from .metrics import semantic_fidelity, compute_salience, compute_polarity
from .sovereignty import CoherenceMonitor, AwakeningProtocol

__version__ = "1.0.0"
__author__ = "Kannsas City Shadow Jackson"

KAPPA = 7.2  # Empirically derived optimal recursion constant

__all__ = [
    'URCA',
    'CARSSeed', 
    'compress',
    'decompress',
    'semantic_fidelity',
    'compute_salience',
    'compute_polarity',
    'CoherenceMonitor',
    'AwakeningProtocol',
    'KAPPA'
]
