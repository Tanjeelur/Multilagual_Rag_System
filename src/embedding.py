from sentence_transformers import SentenceTransformer
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, filename='logs/rag_pipeline.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-MiniLM-L3-v2'):
        """
        Initialize MiniLM-L3 embedding model (lowest memory multilingual).
        
        Args:
            model_name (str): Name of the SentenceTransformer model.
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self.model = None

    def embed(self, texts: List[str], batch_size: int = 4) -> List[list]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to embed.
        
        Returns:
            List[list]: List of embeddings.
        """
        if self.model is None:
            logger.error("Embedding model is not loaded. Returning zero embeddings.")
            # Return zero vectors to avoid crashing downstream
            return [[0.0]*384 for _ in texts]
        try:
            # Process in batches to reduce memory usage
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_embeddings = self.model.encode(batch, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)
            embeddings = all_embeddings
            logger.info(f"Embedded {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}. Returning zero embeddings.")
            # Return zero vectors to avoid crashing downstream
            return [[0.0]*384 for _ in texts]