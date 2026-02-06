"""
Enhanced Semantic Fidelity Metrics for CARS v2.0

Improvements:
- Optional sentence-transformers integration for real embeddings
- Optional transformers/GPT-2 integration for real perplexity
- Better fallback metrics when models unavailable
- BERTScore approximation
- More robust error handling

Install optional dependencies:
    pip install sentence-transformers transformers torch
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import math
import warnings

# Optional imports with graceful fallback
SENTENCE_TRANSFORMERS_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


class EmbeddingModel:
    """
    Wrapper for sentence embedding models with fallback.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = None
        self.model_name = model_name
        self._initialized = False
        
    def _lazy_init(self):
        """Lazy initialization to avoid loading on import."""
        if self._initialized:
            return
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                print(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                warnings.warn(f"Failed to load {self.model_name}: {e}. Using fallback.")
        
        self._initialized = True
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding vector."""
        self._lazy_init()
        
        if self.model is not None:
            return self.model.encode(text, convert_to_numpy=True)
        
        # Fallback: TF-IDF-like embedding
        return self._fallback_encode(text)
    
    def _fallback_encode(self, text: str, dim: int = 384) -> np.ndarray:
        """
        Fallback embedding using character/word statistics.
        
        Creates a pseudo-embedding that captures:
        - Character n-gram frequencies
        - Word length distribution
        - Punctuation patterns
        """
        vec = np.zeros(dim)
        
        if not text:
            return vec
        
        # Character trigram hashing (first 128 dims)
        for i in range(len(text) - 2):
            trigram = text[i:i+3].lower()
            idx = hash(trigram) % 128
            vec[idx] += 1
        
        # Word features (next 128 dims)
        words = text.split()
        for i, word in enumerate(words[:64]):
            idx = 128 + (hash(word.lower()) % 64)
            vec[idx] += 1
            # Word length feature
            idx = 192 + min(63, len(word))
            vec[idx] += 1
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec


class PerplexityModel:
    """
    Wrapper for perplexity computation with fallback.
    """
    
    def __init__(self, model_name: str = 'gpt2'):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self._initialized = False
        self.device = 'cpu'
        
    def _lazy_init(self):
        """Lazy initialization."""
        if self._initialized:
            return
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
                self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
                self.model.eval()
                
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    self.model = self.model.to(self.device)
                
                print(f"Loaded perplexity model: {self.model_name}")
            except Exception as e:
                warnings.warn(f"Failed to load {self.model_name}: {e}. Using fallback.")
        
        self._initialized = True
    
    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text."""
        self._lazy_init()
        
        if self.model is not None and self.tokenizer is not None:
            return self._model_perplexity(text)
        
        return self._fallback_perplexity(text)
    
    def _model_perplexity(self, text: str) -> float:
        """Compute perplexity using actual language model."""
        try:
            encodings = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            input_ids = encodings.input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
            
            return torch.exp(loss).item()
        except Exception:
            return self._fallback_perplexity(text)
    
    def _fallback_perplexity(self, text: str) -> float:
        """
        Fallback perplexity estimation based on:
        - Vocabulary size / token count ratio
        - Average word frequency (common words = lower PPL)
        - Sentence structure regularity
        """
        if not text:
            return float('inf')
        
        words = text.lower().split()
        if not words:
            return float('inf')
        
        # Vocabulary diversity (higher = more perplexing)
        vocab_ratio = len(set(words)) / len(words)
        
        # Common word ratio (more common = less perplexing)
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                       'could', 'should', 'may', 'might', 'must', 'shall', 'can',
                       'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                       'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
                       'she', 'we', 'they', 'and', 'or', 'but', 'if', 'then'}
        common_ratio = len([w for w in words if w in common_words]) / len(words)
        
        # Estimate perplexity (lower common_ratio and higher vocab_ratio = higher PPL)
        ppl = 50 * (1 + vocab_ratio) * (1 - common_ratio * 0.5)
        
        return max(1.0, min(1000.0, ppl))


# Global model instances (lazy loaded)
_embedding_model: Optional[EmbeddingModel] = None
_perplexity_model: Optional[PerplexityModel] = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create global embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


def get_perplexity_model() -> PerplexityModel:
    """Get or create global perplexity model."""
    global _perplexity_model
    if _perplexity_model is None:
        _perplexity_model = PerplexityModel()
    return _perplexity_model


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def embedding_similarity(original: str, reconstructed: str, 
                         model: Optional[Any] = None,
                         use_transformers: bool = True) -> float:
    """
    Compute embedding cosine similarity.
    
    Args:
        original: Original text
        reconstructed: Reconstructed text
        model: Optional pre-loaded embedding model
        use_transformers: Whether to try loading sentence-transformers
        
    Returns:
        Cosine similarity score [0, 1]
    """
    if model is not None:
        emb1 = model.encode(original)
        emb2 = model.encode(reconstructed)
        return float(cosine_similarity(emb1, emb2))
    
    if use_transformers:
        emb_model = get_embedding_model()
        emb1 = emb_model.encode(original)
        emb2 = emb_model.encode(reconstructed)
        return float(cosine_similarity(emb1, emb2))
    
    # Pure fallback
    return _bag_of_words_similarity(original, reconstructed)


def _bag_of_words_similarity(text1: str, text2: str) -> float:
    """Bag-of-words cosine similarity fallback."""
    words1 = Counter(text1.lower().split())
    words2 = Counter(text2.lower().split())
    
    all_words = set(words1.keys()) | set(words2.keys())
    if not all_words:
        return 0.0
    
    vec1 = np.array([words1.get(w, 0) for w in all_words])
    vec2 = np.array([words2.get(w, 0) for w in all_words])
    
    return float(cosine_similarity(vec1, vec2))


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generate n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference: str, candidate: str, max_n: int = 4,
               smoothing: bool = True) -> float:
    """
    Compute BLEU-n score with optional smoothing.
    
    Args:
        reference: Reference text
        candidate: Candidate text
        max_n: Maximum n-gram size (default 4 for BLEU-4)
        smoothing: Apply smoothing for zero counts
        
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
            if smoothing:
                precisions.append(1e-10)
            else:
                precisions.append(0.0)
        else:
            precision = matches / total
            if smoothing and precision == 0:
                precision = 1e-10
            precisions.append(precision)
    
    # Geometric mean
    if 0.0 in precisions and not smoothing:
        return 0.0
    
    log_precisions = [math.log(max(p, 1e-10)) for p in precisions]
    geo_mean = math.exp(sum(log_precisions) / len(log_precisions))
    
    return bp * geo_mean


def bert_score_approximation(original: str, reconstructed: str,
                             model: Optional[Any] = None) -> Dict[str, float]:
    """
    Approximation of BERTScore using available embeddings.
    
    Returns precision, recall, and F1 scores.
    """
    emb_model = model if model is not None else get_embedding_model()
    
    # Get word-level embeddings (simplified)
    orig_words = original.split()
    recon_words = reconstructed.split()
    
    if not orig_words or not recon_words:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Encode each word
    orig_embs = [emb_model.encode(w) for w in orig_words[:50]]  # Limit for speed
    recon_embs = [emb_model.encode(w) for w in recon_words[:50]]
    
    # Compute max similarity scores
    precision_scores = []
    for r_emb in recon_embs:
        max_sim = max(cosine_similarity(r_emb, o_emb) for o_emb in orig_embs)
        precision_scores.append(max_sim)
    
    recall_scores = []
    for o_emb in orig_embs:
        max_sim = max(cosine_similarity(o_emb, r_emb) for r_emb in recon_embs)
        recall_scores.append(max_sim)
    
    precision = np.mean(precision_scores) if precision_scores else 0.0
    recall = np.mean(recall_scores) if recall_scores else 0.0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def perplexity_ratio(original: str, reconstructed: str,
                     model: Optional[Any] = None,
                     use_transformers: bool = True) -> float:
    """
    Compute perplexity ratio PPL(reconstructed) / PPL(original).
    
    Target: < 1.2 (reconstruction no more than 20% more perplexing)
    """
    if model is not None:
        ppl_orig = model.compute_perplexity(original)
        ppl_recon = model.compute_perplexity(reconstructed)
    elif use_transformers:
        ppl_model = get_perplexity_model()
        ppl_orig = ppl_model.compute_perplexity(original)
        ppl_recon = ppl_model.compute_perplexity(reconstructed)
    else:
        # Pure fallback
        ppl_model = PerplexityModel()
        ppl_orig = ppl_model._fallback_perplexity(original)
        ppl_recon = ppl_model._fallback_perplexity(reconstructed)
    
    if ppl_orig <= 0:
        return float('inf')
    
    return ppl_recon / ppl_orig


def semantic_fidelity(original: str, reconstructed: str,
                     embedding_model: Optional[Any] = None,
                     language_model: Optional[Any] = None,
                     use_transformers: bool = True,
                     weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Compute comprehensive semantic fidelity metrics.
    
    Args:
        original: Original text
        reconstructed: Reconstructed text  
        embedding_model: Optional sentence transformer
        language_model: Optional language model for perplexity
        use_transformers: Whether to try loading transformer models
        weights: Optional metric weights for composite score
        
    Returns:
        Dictionary with all metrics and composite score
    """
    if weights is None:
        weights = {
            'cosine_sim': 0.35,
            'bleu4': 0.25,
            'bert_f1': 0.20,
            'perplexity_ratio': 0.10,
            'length_ratio': 0.10
        }
    
    # Compute individual metrics
    cos_sim = embedding_similarity(original, reconstructed, 
                                   embedding_model, use_transformers)
    bleu4 = bleu_score(original, reconstructed, max_n=4)
    
    bert_scores = bert_score_approximation(original, reconstructed, embedding_model)
    
    ppl_ratio = perplexity_ratio(original, reconstructed, 
                                 language_model, use_transformers)
    
    # Length ratio
    len_orig = len(original)
    len_recon = len(reconstructed)
    length_ratio = min(len_orig, len_recon) / max(len_orig, len_recon) if max(len_orig, len_recon) > 0 else 0
    
    # Normalize perplexity ratio to [0, 1] score
    ppl_score = 1.0 / max(1.0, ppl_ratio)
    
    # Composite score
    composite = (
        weights['cosine_sim'] * cos_sim +
        weights['bleu4'] * bleu4 +
        weights['bert_f1'] * bert_scores['f1'] +
        weights['perplexity_ratio'] * ppl_score +
        weights['length_ratio'] * length_ratio
    )
    
    return {
        'cosine_similarity': cos_sim,
        'bleu4': bleu4,
        'bert_precision': bert_scores['precision'],
        'bert_recall': bert_scores['recall'],
        'bert_f1': bert_scores['f1'],
        'perplexity_ratio': ppl_ratio,
        'perplexity_score': ppl_score,
        'length_ratio': length_ratio,
        'composite_score': composite,
        'meets_cosine_target': cos_sim > 0.90,
        'meets_bleu_target': bleu4 > 0.75,
        'meets_bert_target': bert_scores['f1'] > 0.85,
        'meets_ppl_target': ppl_ratio < 1.2,
        'transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE or TRANSFORMERS_AVAILABLE
    }


def compute_salience(text: str, 
                    corpus_centroid: Optional[np.ndarray] = None,
                    embedding_model: Optional[Any] = None) -> float:
    """
    Compute retrieval priority weight.
    """
    words = text.split()
    if len(words) == 0:
        return 0.0
    
    # Information density
    info_density = len(set(words)) / len(words)
    
    # Distinctiveness
    distinctiveness = 0.5
    if embedding_model is not None and corpus_centroid is not None:
        emb = embedding_model.encode(text)
        distinctiveness = 1 - cosine_similarity(emb, corpus_centroid)
    elif corpus_centroid is not None:
        emb_model = get_embedding_model()
        emb = emb_model.encode(text)
        distinctiveness = 1 - cosine_similarity(emb, corpus_centroid)
    
    # Coherence
    import re
    sentences = len(re.findall(r'[.!?]+', text))
    coherence = min(1.0, sentences / 5) if sentences > 0 else 0.3
    
    # Explicit markers
    markers = ['important', 'remember', 'key', 'critical', 'note', 'essential']
    explicit = 1.0 if any(m in text.lower() for m in markers) else 0.0
    
    return 0.30 * info_density + 0.25 * distinctiveness + 0.25 * coherence + 0.20 * explicit


def compute_polarity(text: str) -> float:
    """Compute contextual orientation [-1, 1]."""
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
    """Protocol for human evaluation of semantic fidelity."""
    
    def __init__(self):
        self.evaluations = []
        
    def add_evaluation(self, sample_id: str, annotator_id: str,
                      original: str, reconstructed: str, score: int):
        """Record a human evaluation (1-5 Likert scale)."""
        if not 1 <= score <= 5:
            raise ValueError("Score must be between 1 and 5")
        
        self.evaluations.append({
            'sample_id': sample_id,
            'annotator_id': annotator_id,
            'original': original,
            'reconstructed': reconstructed,
            'score': score
        })
    
    def compute_agreement(self) -> float:
        """Compute Krippendorff's alpha approximation."""
        from collections import defaultdict
        
        by_sample = defaultdict(list)
        for e in self.evaluations:
            by_sample[e['sample_id']].append(e['score'])
        
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
        alpha = 1 - (mean_within_var / total_var)
        return max(0.0, min(1.0, alpha))
    
    def summary(self) -> Dict[str, float]:
        """Get evaluation summary statistics."""
        scores = [e['score'] for e in self.evaluations]
        
        return {
            'n_samples': len(set(e['sample_id'] for e in self.evaluations)),
            'n_evaluations': len(self.evaluations),
            'mean_score': float(np.mean(scores)) if scores else 0.0,
            'std_score': float(np.std(scores)) if scores else 0.0,
            'agreement_alpha': self.compute_agreement(),
            'meets_agreement_target': self.compute_agreement() > 0.7
        }


def check_dependencies() -> Dict[str, bool]:
    """Check which optional dependencies are available."""
    return {
        'sentence_transformers': SENTENCE_TRANSFORMERS_AVAILABLE,
        'transformers': TRANSFORMERS_AVAILABLE,
        'numpy': True  # Always available (required)
    }
