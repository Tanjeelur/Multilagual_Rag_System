"""
Response generation module using OpenAI GPT models.

This module handles intelligent response generation with context awareness,
multilingual support, and conversation continuity.
"""

import openai
from typing import Dict, Any, Optional, List
import logging
from ..utils.helpers import detect_language

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Generates contextual responses using OpenAI's language models.
    
    This class provides intelligent response generation with multilingual support,
    context awareness, and conversation continuity for the RAG system.
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.3,
        top_p: float = 1.0
    ):
        """
        Initialize the response generator with OpenAI configuration.
        
        Args:
            model_name (str): OpenAI model name to use
            api_key (Optional[str]): OpenAI API key
            max_tokens (int): Maximum tokens in response
            temperature (float): Creativity/randomness control (0.0-1.0)
            top_p (float): Nucleus sampling parameter
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Set API key if provided
        if api_key:
            openai.api_key = api_key
        
        logger.info(f"Response generator initialized with model: {model_name}")
    
    def generate_response(
        self, 
        query: str, 
        context: str,
        language: Optional[str] = None,
        response_style: str = "helpful",
        include_sources: bool = False
    ) -> str:
        """
        Generate a response based on query and context.
        
        Args:
            query (str): User's query
            context (str): Retrieved context from memory
            language (Optional[str]): Target response language
            response_style (str): Style of response ("helpful", "concise", "detailed")
            include_sources (bool): Whether to include source references
            
        Returns:
            str: Generated response
        """
        if not query.strip():
            return "দয়া করে একটি বৈধ প্রশ্ন করুন। Please provide a valid question."
        
        try:
            # Detect language if not provided
            if not language:
                language = detect_language(query)
            
            # Create system prompt
            system_prompt = self._create_system_prompt(language, response_style)
            
            # Create user prompt
            user_prompt = self._create_user_prompt(query, context, include_sources)
            
            # Generate response using OpenAI
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            logger.info("Successfully generated response")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._get_fallback_response(language)
    
    def _create_system_prompt(self, language: str, response_style: str) -> str:
        """
        Create system prompt based on target language and style.
        
        Args:
            language (str): Target language
            response_style (str): Response style
            
        Returns:
            str: System prompt
        """
        base_instructions = {
            'bn': {
                'helpful': """আপনি একটি সহায়ক বাংলা সাহিত্য বিশেষজ্ঞ। আপনার কাছে রবীন্দ্রনাথ ঠাকুরের "অপরিচিতা" গল্পের তথ্য রয়েছে। প্রদত্ত প্রসঙ্গের ভিত্তিতে সঠিক এবং সংক্ষিপ্ত উত্তর দিন। সর্বদা বাংলায় উত্তর দিন।""",
                'concise': """আপনি একটি বাংলা সাহিত্য বিশেষজ্ঞ। সংক্ষিপ্ত এবং সরাসরি উত্তর দিন। বাংলায় উত্তর দিন।""",
                'detailed': """আপনি একটি বাংলা সাহিত্য বিশেষজ্ঞ। বিস্তারিত ব্যাখ্যা সহ উত্তর দিন। প্রয়োজনে উদাহরণ দিন। বাংলায় উত্তর দিন।"""
            },
            'en': {
                'helpful': """You are a helpful Bengali literature expert with access to Rabindranath Tagore's story "Aparichita". Provide accurate and concise answers based on the given context. Always respond in English.""",
                'concise': """You are a Bengali literature expert. Provide concise and direct answers. Always respond in English.""",
                'detailed': """You are a Bengali literature expert. Provide detailed explanations with examples when needed. Always respond in English."""
            },
            'mixed': {
                'helpful': """You are a multilingual Bengali literature expert. You have access to information about Rabindranath Tagore's "Aparichita" story. Provide accurate answers based on the given context. Respond in the same language as the user's question, or in Bengali if the context is primarily in Bengali.""",
                'concise': """You are a multilingual literature expert. Provide concise answers in the appropriate language.""",
                'detailed': """You are a multilingual literature expert. Provide detailed answers with context and examples in the appropriate language."""
            }
        }
        
        # Get the appropriate instruction
        lang_key = language if language in ['bn', 'en'] else 'mixed'
        style_key = response_style if response_style in base_instructions[lang_key] else 'helpful'
        
        instruction = base_instructions[lang_key][style_key]
        
        # Add common guidelines
        common_guidelines = """

Guidelines:
- If the answer is not found in the provided context, clearly state that you don't have enough information
- Be precise and factual
- Do not make up information
- Focus on the specific question asked"""
        
        return instruction + common_guidelines
    
    def _create_user_prompt(self, query: str, context: str, include_sources: bool) -> str:
        """
        Create user prompt with query and context.
        
        Args:
            query (str): User query
            context (str): Retrieved context
            include_sources (bool): Whether to include source references
            
        Returns:
            str: User prompt
        """
        prompt_parts = []
        
        # Add context if available
        if context.strip():
            prompt_parts.append("Context:")
            prompt_parts.append(context)
            prompt_parts.append("")
        
        # Add the user's question
        prompt_parts.append(f"Question: {query}")
        prompt_parts.append("")
        
        # Add instruction for source inclusion
        if include_sources and context.strip():
            prompt_parts.append("Please provide a direct answer based on the context above. If you reference specific information, mention the source.")
        else:
            prompt_parts.append("Please provide a direct and accurate answer based on the given context.")
        
        prompt_parts.append("")
        prompt_parts.append("Answer:")
        
        return "\n".join(prompt_parts)
    
    def _get_fallback_response(self, language: str) -> str:
        """
        Get fallback response when generation fails.
        
        Args:
            language (str): Target language
            
        Returns:
            str: Fallback response
        """
        fallback_responses = {
            'bn': "দুঃখিত, আপনার প্রশ্ন প্রক্রিয়া করতে একটি সমস্যা হয়েছে। দয়া করে আবার চেষ্টা করুন।",
            'en': "I apologize, but I encountered an error while processing your question. Please try again.",
            'mixed': "দুঃখিত, আপনার প্রশ্ন প্রক্রিয়া করতে সমস্যা হয়েছে। Sorry, I encountered an error processing your question."
        }
        
        return fallback_responses.get(language, fallback_responses['mixed'])
    
    def generate_follow_up_questions(self, query: str, response: str, context: str) -> List[str]:
        """
        Generate relevant follow-up questions based on the conversation.
        
        Args:
            query (str): Original query
            response (str): Generated response
            context (str): Retrieved context
            
        Returns:
            List[str]: List of follow-up questions
        """
        try:
            language = detect_language(query)
            
            # Create prompt for follow-up questions
            followup_prompt = f"""Based on this conversation:
Question: {query}
Answer: {response}

Generate 3 relevant follow-up questions that a user might ask. Make them specific to the context provided.

Follow-up questions:"""
            
            # Generate follow-up questions
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"Generate follow-up questions in {language} language about Bengali literature."},
                    {"role": "user", "content": followup_prompt}
                ],
                max_tokens=200,
                temperature=0.5
            )
            
            followup_text = response.choices[0].message.content.strip()
            
            # Parse follow-up questions (simple parsing)
            questions = []
            for line in followup_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('১.') or line.startswith('1.')):
                    question = line.lstrip('-123456789১২৩৪৫৬৭৮৯. ').strip()
                    if question:
                        questions.append(question)
            
            return questions[:3]  # Return maximum 3 questions
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {str(e)}")
            return []
    
    def summarize_conversation(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of the conversation history.
        
        Args:
            conversation_history (List[Dict[str, Any]]): List of conversation turns
            
        Returns:
            str: Conversation summary
        """
        if not conversation_history:
            return "No conversation history available."
        
        try:
            # Create conversation text
            conversation_text = []
            for turn in conversation_history:
                conversation_text.append(f"User: {turn['user_query']}")
                conversation_text.append(f"Assistant: {turn['assistant_response']}")
            
            full_conversation = "\n".join(conversation_text)
            
            # Generate summary
            summary_prompt = f"""Summarize this conversation about Bengali literature:

{full_conversation}

Provide a brief summary of the main topics discussed and key information shared."""
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Provide a concise summary of the conversation."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating conversation summary: {str(e)}")
            return "Unable to generate conversation summary."
