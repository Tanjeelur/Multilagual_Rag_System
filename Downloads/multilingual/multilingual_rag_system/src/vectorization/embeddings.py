"""
Embedding generation module for multilingual text processing.

This module handles text embedding generation using sentence transformers
optimized for both Bengali and English text processing.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional
import logging
import torch

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generates embeddings for multilingual text using sentence transformers.
    
    This class is optimized for Bengali and English text processing with
    efficient batch processing and caching capabilities.
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name (str): Name of the sentence transformer model
            device (Optional[str]): Device to use ('cpu', 'cuda', or None for auto)
            normalize_embeddings (bool): Whether to normalize embeddings
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model = None
        self.device = device or self._get_optimal_device()
        
        # Load the model
        self._load_model()
        
    def _get_optimal_device(self) -> str:
        """Determine the optimal device for processing."""
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            self.model = SentenceTransformer(self.model_name)
            
            # Move model to specified device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.cuda()
            
            logger.info("Successfully loaded embedding model")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def encode_text(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s) with batch processing.
        
        Args:
            texts (Union[str, List[str]]): Single text or list of texts
            batch_size (int): Batch size for processing
            show_progress (bool): Whether to show progress bar
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        try:
            logger.debug(f"Encoding {len(texts)} texts with batch size {batch_size}")
            
            # Generate embeddings with batch processing
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings
            )
            
            logger.debug(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text (optimized for single queries).
        
        Args:
            text (str): Input text
            
        Returns:
            np.ndarray: Single embedding vector
        """
        try:
            embedding = self.model.encode(
                [text], 
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings
            )[0]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating single embedding: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.
        
        Returns:
            int: Embedding dimension
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        return self.model.get_sentence_embedding_dimension()
    