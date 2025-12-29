import stanza
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class StanzaPreprocessor:
    def __init__(self, exclude_stopwords, exclude_lemmatization, add_stopwords={'nya', 'ya', 'nih'}):
        stanza.download("id")

        self.exclude_stopwords = exclude_stopwords
        self.exclude_lemmatization = exclude_lemmatization

        self.nlp = stanza.Pipeline(lang="id", processors='tokenize,mwt,pos,lemma')

        factory = StopWordRemoverFactory()
        sastrawi_stopwords = set(factory.get_stop_words())

        self.stopwords = sastrawi_stopwords.union(add_stopwords)

    def transform(self, text: str) -> str: 
        doc = self.nlp(text)

        processed_tokens = []
        for sentence in doc.sentences:
            for word in sentence.words:
                token = word.text.lower()
                lemma = word.lemma.lower()

                if token in self.exclude_stopwords or token not in self.stopwords:
                    final_token = token if token in self.exclude_lemmatization else lemma
                    processed_tokens.append(final_token)
        return re.sub(r'\s+', ' ', ' '.join(processed_tokens)).strip()