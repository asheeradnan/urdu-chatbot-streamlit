"""
Urdu Text Preprocessor
"""
import re

class UrduPreprocessor:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
    
    def normalize_urdu(self, text):
        """Normalize Urdu text"""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.replace('ٱ', 'ا').replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        text = text.replace('ي', 'ی')
        
        return text
    
    def tokenize(self, text):
        """Tokenize text"""
        return text.split()
    
    def encode_sentence(self, sentence, max_len=50):
        """Encode sentence to indices"""
        tokens = self.tokenize(sentence)
        indices = [self.word2idx.get(token, self.word2idx.get('<UNK>', 0)) for token in tokens]
        
        if len(indices) > max_len:
            indices = indices[:max_len]
        
        while len(indices) < max_len:
            indices.append(self.word2idx.get('<PAD>', 0))
        
        return indices
    
    def decode_sentence(self, indices):
        """Decode indices to sentence"""
        special_tokens = {'<PAD>', '<SOS>', '<EOS>', '<UNK>'}
        words = []
        
        for idx in indices:
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if word not in special_tokens:
                    words.append(word)
        
        return ' '.join(words)