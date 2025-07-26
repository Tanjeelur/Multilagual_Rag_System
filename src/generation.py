from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import numpy as np
from typing import List, Tuple
import logging
import gc
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, filename='logs/rag_pipeline.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Generator:
    def __init__(self, model_name: str = 'google/mt5-small'):
        """
        Initialize mT5 model and tokenizer.
        
        Args:
            model_name (str): Name of the mT5 model.
        """
        try:
            # Force CPU usage to reduce memory consumption
            device = torch.device('cpu')
            # Use low_cpu_mem_usage for more efficient loading
            self.model = MT5ForConditionalGeneration.from_pretrained(
                model_name, 
                low_cpu_mem_usage=True,
                device_map=None
            )
            self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
            self.model.to(device)
            # Run garbage collection to free memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info(f"Loaded generation model: {model_name} on CPU")
        except Exception as e:
            logger.error(f"Failed to load generation model: {str(e)}")
            raise
            
    def _clean_output(self, text: str) -> str:
        """
        Clean the generated output to remove any <extra_id_X> tokens.
        
        Args:
            text (str): The text to clean.
            
        Returns:
            str: Cleaned text without <extra_id_X> tokens.
        """
        import re
        # Remove any <extra_id_X> tokens that might remain
        cleaned_text = re.sub(r'<extra_id_\d+>', '', text)
        # Remove any leading/trailing whitespace
        cleaned_text = cleaned_text.strip()
        
        # If the text is empty after cleaning, provide a fallback response
        if not cleaned_text:
            cleaned_text = "I couldn't generate a proper response. Please try rephrasing your question."
            
        return cleaned_text

    def generate(self, query: str, chunks: List[Tuple], language: str, embedder) -> Tuple[str, float]:
        """
        Generate answer using mT5 and calculate confidence.
        
        Args:
            query (str): Input query.
            chunks (List[Tuple]): List of retrieved chunks (id, text, page).
            language (str): Output language ('bn' or 'en').
            embedder: Embedder instance for confidence calculation.
        
        Returns:
            Tuple[str, float]: Generated answer and confidence score.
        """
        try:
            # Join all chunk texts for context
            context = "\n".join([chunk[1] for chunk in chunks if chunk[1].strip()])
            lang_name = "Bengali" if language == "bn" else "English"
            # Prompt engineering: require a full-sentence answer using only the context
            prompt = (
                f"You are a helpful assistant answering questions about Bengali literature. "
                f"Answer the following question in {lang_name} using only the information provided below. "
                f"If the answer is not present, reply 'উত্তর পাওয়া যায়নি' (Answer not found).\n"
                f"Question: {query}\n"
                f"Context: {context}\n"
                f"Answer in a complete sentence."
            )
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            decoder_start_token_id = self.model.config.decoder_start_token_id
            decoder_input_ids = torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=1.2
            )
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = self._clean_output(answer)
            # Confidence calculation
            query_embedding = embedder.embed([query])[0]
            chunk_embeddings = embedder.embed([chunk[1] for chunk in chunks])
            similarities = [
                np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                for emb in chunk_embeddings
            ]
            confidence = np.mean(similarities) if similarities else 0.0
            logger.info(f"Generated answer: {answer} with confidence {confidence}")
            return answer, confidence
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise