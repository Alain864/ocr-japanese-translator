"""Translation module using OpenAI API."""

from typing import List, Optional
from openai import OpenAI

from config import OPENAI_API_KEY, TARGET_LANGUAGE


class Translator:
    """Handles translation of Japanese text using OpenAI API."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the translator with OpenAI client.
        
        Args:
            model: The OpenAI model to use for translation
        """
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
    
    def translate(self, text: str) -> str:
        """
        Translate Japanese manga dialogue to natural English.
        
        Args:
            text: The Japanese text to translate
            
        Returns:
            The translated English text
        """
        if not text.strip():
            return ""
        
        prompt = f"""Translate this Japanese manga dialogue to natural English.
Keep tone emotional and short.
Return only translation.

{text}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Translation error for '{text[:30]}...': {e}")
            return text  # Return original text if translation fails
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate multiple Japanese texts with shared context.
        
        Args:
            texts: List of Japanese texts to translate
            
        Returns:
            List of translated English texts
        """
        if not texts:
            return []
        
        # Combine texts with labels for context
        combined = "\n".join([f"Line {i+1}: {text}" for i, text in enumerate(texts)])
        
        prompt = f"""Translate these Japanese manga dialogues to natural English.
Keep tone emotional and short.
Return only translations, one per line, labeled.

{combined}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse back into list (format "Line 1: English...")
            translations = []
            for line in response_text.split("\n"):
                if line.strip().startswith("Line "):
                    en = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                    translations.append(en)
            
            # Fallback if parsing fails
            if len(translations) != len(texts):
                print(f"Batch translation parsing failed, falling back to individual translations")
                return [self.translate(text) for text in texts]
            
            return translations
            
        except Exception as e:
            print(f"Batch translation error: {e}")
            # Fallback to individual translations
            return [self.translate(text) for text in texts]