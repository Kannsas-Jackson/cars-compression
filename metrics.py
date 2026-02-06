"""
Semantic Fidelity Metrics for CARS

Implements multiple complementary metrics for measuring
compression quality and reconstruction accuracy.

Metrics:
    1. Embedding Cosine Similarity (target > 0.90)
    2. BLEU-4 Reconstruction Score (target > 0.75)
    3. BERTScore F1 (target > 0.85)
    4. Perplexity Ratio (target < 1.2)
    5. Human Evaluation Protocol (5-point Likert)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def embedding_similarity(original: str, reconstructed: str, 
                         model=None) -> float:
    """
    Compute embedding cosine similarity.
    
    Uses sentence-transformers if available, falls back to
    simple bag-of-words embedding.
    
    Args:
        original: Original text
        reconstructed: Reconstructed text
        model: Optional sentence transformer model
        
    Returns:
        Cosine similarity score [0, 1]
    """
    if model is not None:
        emb1 = model.encode(original)
        emb2 = model.encode(reconstructed)
        return float(cosine_similarity(emb1, emb2))
    
    # Fallback: bag-of-words cosine similarity
    words1 = Counter(original.lower().split())
    words2 = Counter(reconstructed.lower().split())
    
    all_words = set(words1.keys()) | set(words2.keys())
    vec1 = np.array([words1.get(w, 0) for w in all_words])
    vec2 = np.array([words2.get(w, 0) for w in all_words])
    
    return float(cosine_similarity(vec1, vec2))


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generate n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference: str, candidate: str, max_n: int = 4) -> float:
    """
    Compute BLEU-n score.
    
    Args:
        reference: Reference text
        candidate: Candidate text
        max_n: Maximum n-gram size (default 4 for BLEU-4)
        
    Returns:
        BLEU score [0, 1]
    """
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if len(cand_tokens) == 0:
        return 0.0
    
    # Brevity penalty
    bp = 1.0
    if len(cand_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(cand_tokens))
    
    # N-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(ngrams(ref_tokens, n))
        cand_ngrams = Counter(ngrams(cand_tokens, n))
        
        matches = sum((ref_ngrams & cand_ngrams).values())
        total = sum(cand_ngrams.values())
        
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / total)
    
    # Geometric mean with smoothing
    if 0.0 in precisions:
        return 0.0
    
    log_precisions = [math.log(p) for p in precisions]
    geo_mean = math.exp(sum(log_precisions) / len(log_precisions))
    
    return bp * geo_mean


def perplexity_ratio(original: str, reconstructed: str,
                     language_model=None) -> float:
    """
    Compute perplexity ratio PPL(reconstructed) / PPL(original).
    
    Target: < 1.2 (reconstruction no more than 20% more perplexing)
    
    Args:
        original: Original text
        reconstructed: Reconstructed text
        language_model: Optional language model for PPL computation
        
    Returns:
        Perplexity ratio (lower is better, 1.0 is perfect)
    """
    if language_model is not None:
        # Use actual language model
        ppl_orig = language_model.perplexity(original)
        ppl_recon = language_model.perplexity(reconstructed)
        return ppl_recon / ppl_orig if ppl_orig > 0 else float('inf')
    
    # Fallback: estimate based on vocabulary overlap
    orig_vocab = set(original.lower().split())
    recon_vocab = set(reconstructed.lower().split())
    
    overlap = len(orig_vocab & recon_vocab)
    total = len(orig_vocab | recon_vocab)
    
    if total == 0:
        return float('inf')
    
    # Higher overlap = lower perplexity ratio
    vocab_similarity = overlap / total
    
    # Map to perplexity ratio (similarity of 1.0 → ratio of 1.0)
    return 1.0 / (vocab_similarity + 0.01)


def semantic_fidelity(original: str, reconstructed: str,
                     embedding_model=None,
                     language_model=None,
                     weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Compute comprehensive semantic fidelity metrics.
    
    Args:
        original: Original text
        reconstructed: Reconstructed text  
        embedding_model: Optional sentence transformer
        language_model: Optional language model for perplexity
        weights: Optional metric weights for composite score
        
    Returns:
        Dictionary with all metrics and composite score
    """
    if weights is None:
        weights = {
            'cosine_sim': 0.40,
            'bleu4': 0.30,
            'perplexity_ratio': 0.20,
            'length_ratio': 0.10
        }
    
    # Compute individual metrics
    cos_sim = embedding_similarity(original, reconstructed, embedding_model)
    bleu4 = bleu_score(original, reconstructed, max_n=4)
    ppl_ratio = perplexity_ratio(original, reconstructed, language_model)
    
    # Length ratio (penalize significant length differences)
    len_orig = len(original)
    len_recon = len(reconstructed)
    length_ratio = min(len_orig, len_recon) / max(len_orig, len_recon) if max(len_orig, len_recon) > 0 else 0
    
    # Normalize perplexity ratio to [0, 1] score (lower ratio = higher score)
    ppl_score = 1.0 / ppl_ratio if ppl_ratio > 0 else 0.0
    ppl_score = min(1.0, ppl_score)  # Cap at 1.0
    
    # Composite score
    composite = (
        weights['cosine_sim'] * cos_sim +
        weights['bleu4'] * bleu4 +
        weights['perplexity_ratio'] * ppl_score +
        weights['length_ratio'] * length_ratio
    )
    
    return {
        'cosine_similarity': cos_sim,
        'bleu4': bleu4,
        'perplexity_ratio': ppl_ratio,
        'perplexity_score': ppl_score,
        'length_ratio': length_ratio,
        'composite_score': composite,
        'meets_cosine_target': cos_sim > 0.90,
        'meets_bleu_target': bleu4 > 0.75,
        'meets_ppl_target': ppl_ratio < 1.2
    }


def compute_salience(text: str, 
                    corpus_centroid: Optional[np.ndarray] = None,
                    embedding_model=None) -> float:
    """
    Compute retrieval priority weight.
    
    Salience(x) = α₁·InvPPL(x) + α₂·Dist(x,μ) + α₃·MI(x,context) + α₄·Explicit(x)
    
    Where:
        α₁ = 0.30 (information density)
        α₂ = 0.25 (distinctiveness)
        α₃ = 0.25 (coherence contribution)
        α₄ = 0.20 (explicit marking)
    """
    words = text.split()
    if len(words) == 0:
        return 0.0
    
    # Information density (lexical diversity)
    info_density = len(set(words)) / len(words)
    
    # Distinctiveness
    distinctiveness = 0.5
    if embedding_model is not None and corpus_centroid is not None:
        emb = embedding_model.encode(text)
        distinctiveness = 1 - cosine_similarity(emb, corpus_centroid)
    
    # Coherence (sentence structure)
    sentences = text.count('.') + text.count('!') + text.count('?')
    coherence = min(1.0, sentences / 5) if sentences > 0 else 0.3
    
    # Explicit markers
    markers = ['important', 'remember', 'key', 'critical', 'note', 'essential']
    explicit = 1.0 if any(m in text.lower() for m in markers) else 0.0
    
    return 0.30 * info_density + 0.25 * distinctiveness + 0.25 * coherence + 0.20 * explicit


def compute_polarity(text: str) -> float:
    """
    Compute contextual orientation [-1, 1].
    
    NOT emotional valence - this is a reconstruction framing parameter.
    Positive values → affirmative reconstruction
    Negative values → contrastive reconstruction
    """
    positive = {'good', 'great', 'excellent', 'success', 'achieve', 'positive',
                'benefit', 'advantage', 'improve', 'gain', 'progress', 'solution'}
    negative = {'bad', 'poor', 'fail', 'problem', 'issue', 'negative', 'loss',
                'disadvantage', 'decline', 'risk', 'threat', 'obstacle'}
    
    words = set(text.lower().split())
    pos = len(words & positive)
    neg = len(words & negative)
    
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0


class HumanEvaluationProtocol:
    """
    Protocol for human evaluation of semantic fidelity.
    
    5-point Likert scale:
        1 = Completely different meaning
        2 = Major meaning differences
        3 = Some meaning preserved, some lost
        4 = Minor meaning differences
        5 = Semantically equivalent
        
    Requirements:
        - Minimum 3 annotators per sample
        - Krippendorff's α > 0.7 for acceptable agreement
        - n ≥ 500 samples for statistical significance
    """
    
    def __init__(self):
        self.evaluations = []
        
    def add_evaluation(self, sample_id: str, annotator_id: str,
                      original: str, reconstructed: str, score: int):
        """Record a human evaluation."""
        if score < 1 or score > 5:
            raise ValueError("Score must be between 1 and 5")
        
        self.evaluations.append({
            'sample_id': sample_id,
            'annotator_id': annotator_id,
            'original': original,
            'reconstructed': reconstructed,
            'score': score
        })
    
    def compute_agreement(self) -> float:
        """
        Compute Krippendorff's alpha for inter-annotator agreement.
        
        Returns:
            Alpha value (> 0.7 is acceptable, > 0.8 is good)
        """
        # Simplified computation - full implementation would use
        # proper Krippendorff's alpha calculation
        from collections import defaultdict
        
        by_sample = defaultdict(list)
        for e in self.evaluations:
            by_sample[e['sample_id']].append(e['score'])
        
        # Compute variance within samples vs total variance
        total_scores = [e['score'] for e in self.evaluations]
        if len(total_scores) < 2:
            return 0.0
            
        total_var = np.var(total_scores)
        if total_var == 0:
            return 1.0
        
        within_vars = []
        for scores in by_sample.values():
            if len(scores) > 1:
                within_vars.append(np.var(scores))
        
        if not within_vars:
            return 0.0
            
        mean_within_var = np.mean(within_vars)
        
        # Alpha = 1 - (within variance / total variance)
        alpha = 1 - (mean_within_var / total_var)
        return max(0.0, min(1.0, alpha))
    
    def summary(self) -> Dict[str, float]:
        """Get evaluation summary statistics."""
        scores = [e['score'] for e in self.evaluations]
        
        return {
            'n_samples': len(set(e['sample_id'] for e in self.evaluations)),
            'n_evaluations': len(self.evaluations),
            'mean_score': np.mean(scores) if scores else 0.0,
            'std_score': np.std(scores) if scores else 0.0,
            'agreement_alpha': self.compute_agreement(),
            'meets_agreement_target': self.compute_agreement() > 0.7
        }
