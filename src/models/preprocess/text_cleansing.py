import re

class TextCleansing:
    def __init__(self, text, norm_dict=None):
        self.text = text
        self.norm_dict = norm_dict if norm_dict else {}

    def correct_typos(self, text):
        for typo, correction in self.norm_dict.items():
            clean_text = re.sub(
                rf'\b{typo}\b', correction, text, flags=re.IGNORECASE
            )
        return clean_text
    
    def reduce_extra_characters(self, text):
        """
        Contoh:
        "sukaaaaa" -> "suka"
        """
        return re.sub(r'(.)\1{2,}', r'\1', text)
    
    def split_nya(self, text, exception_words=None):
        if exception_words is None:
            exception_words = ["tanya", "punya", "bertanya", "hanya"]

        if text in exception_words:
            return text

        return re.sub(r'(.*?)nya$', r'\1 nya', text)

    def process_split_nya(self, text):
        words = text.split()
        processed_words = [
            self.split_nya(word.strip()) for word in words
        ]
        return ' '.join(processed_words)

    def clean(self):
        text = self.text.lower()
        text = re.sub(r"[^a-zA-Z\s']", ' ', text)
        text = self.process_split_nya(text)
        text = self.correct_typos(text)
        text = self.reduce_extra_characters(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text