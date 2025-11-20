"""
from tts_list_preprocessor import TTSListPreprocessor

# Example usage
preprocessor = TTSListPreprocessor()

list_text = "
1) Red
2. Green
3) Blue 😊
"

tts_input = preprocessor.format_list_for_tts(list_text)
print(tts_input)  # Output: "1, Red. 2, Green. 3, Blue."
"""


import re

class TTSListPreprocessor:
    """
    A class to preprocess lists for TTS by:
    - Removing Markdown syntax (bold, italic, code)
    - Stripping emoticons and emojis
    - Reformatting numbered lists (e.g., "1. Red" or "1) Red" → "1, Red.")
    - Outputs a natural sentence for TTS (e.g., "1, Red. 2, Green.")
    """

    def __init__(self):
        # Pre-compile regex patterns for efficiency
        self._bold_italic_pattern = re.compile(r'(\*\*|__)(.*?)\1')
        self._italic_pattern = re.compile(r'(\*|_)(.*?)\1')
        self._code_pattern = re.compile(r'`(.*?)`')
        self._emoticon_pattern = re.compile(r':[\(\)\[\]\{\}DPd3oO/?\\|*]')
        self._emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # Emoticons
            u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # Transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
            "]+", flags=re.UNICODE
        )

    def remove_markdown(self, text):
        """Remove Markdown syntax (bold, italic, code)."""
        text = self._bold_italic_pattern.sub(r'\2', text)
        text = self._italic_pattern.sub(r'\2', text)
        text = self._code_pattern.sub(r'\1', text)
        return text

    def remove_emoticons(self, text):
        """Remove emoticons and emojis."""
        text = self._emoticon_pattern.sub('', text)
        text = self._emoji_pattern.sub('', text)
        return text

    def format_list_for_tts(self, list_text):
        """
        Preprocess a list for TTS:
        - Remove Markdown and emoticons
        - Reformat numbered lists (e.g., "1. Red" or "1) Red" → "1, Red.")
        - Outputs a natural sentence for TTS (e.g., "1, Red. 2, Green.")
        """
        list_text = self.remove_markdown(list_text)
        list_text = self.remove_emoticons(list_text)

        lines = [line.strip() for line in list_text.split('\n') if line.strip()]
        tts_phrases = []
        for line in lines:
            # Match both "1. Red" and "1) Red"
            match = re.match(r'^\d+[.)]\s*', line)
            if match:
                # Split into number and item
                number = line[:match.end()].strip('.) ')
                item = line[match.end():].strip()
                tts_phrases.append(f"{number}, {item}.")
            else:
                tts_phrases.append(line)

        # Join with spaces and capitalize the first letter
        tts_output = ' '.join(tts_phrases)
        if tts_output:
            tts_output = tts_output[0].upper() + tts_output[1:]  # Capitalize first letter
        return tts_output
