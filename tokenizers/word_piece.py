import re
from collections import defaultdict

class WordPieceTokenizer:
    def __init__(self, corpus, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = self.train(corpus)

    def train(self, corpus):
        # Tokenize the corpus at character level, adding '▁' as a special token at the beginning of each word
        tokens = ["▁" + word for word in re.findall(r'\w+|\W+', corpus)]

        # Count token frequencies
        token_freqs = defaultdict(int)
        for token in tokens:
            token_freqs[token] += 1

        # Initialize vocabulary with unique characters
        vocab = set("".join(tokens))

        # Iteratively merge tokens
        while len(vocab) < self.vocab_size:
            best_pair = self.find_best_pair(token_freqs)
            if not best_pair:
                break

            # Merging the best pair
            new_token = ''.join(best_pair)
            vocab.add(new_token)

            # Update token frequencies with merged tokens
            self.update_token_freqs(token_freqs, best_pair, new_token)

        return vocab

    def find_best_pair(self, token_freqs):
        # Find the most frequent pair of adjacent tokens
        pairs = defaultdict(int)
        for token, freq in token_freqs.items():
            symbols = list(token)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq

        best_pair = None
        max_freq = 0
        for pair, freq in pairs.items():
            if freq > max_freq:
                max_freq = freq
                best_pair = pair

        return best_pair

    def update_token_freqs(self, token_freqs, pair, new_token):
        # Update the token frequencies after merging a pair
        pattern = re.escape(''.join(pair))
        pattern = pattern.replace("\\ ", " ")
        for token in list(token_freqs.keys()):
            if pair[0] in token and pair[1] in token:
                new_token_freq = token_freqs[token]
                new_token = re.sub(pattern, new_token, token)
                token_freqs[new_token] += new_token_freq
                del token_freqs[token]

    def tokenize(self, text):
        # Tokenize a new piece of text using the trained vocabulary
        tokens = ["▁" + word for word in re.findall(r'\w+|\W+', text)]
        wordpiece_tokens = []
        for token in tokens:
            sub_tokens = self.sub_tokenize(token)
            wordpiece_tokens.extend(sub_tokens)
        return wordpiece_tokens

    def sub_tokenize(self, token):
        # Sub-tokenize a token into the largest possible sub-tokens in the vocabulary
        if token in self.vocab:
            return [token]
        for i in range(len(token), 0, -1):
            sub_token = token[:i]
            if sub_token in self.vocab:
                return [sub_token] + self.sub_tokenize(token[i:])
        return [token]

# Example Usage
corpus = "Tokenization in the context of natural language processing (NLP) is the process of converting a sequence of characters (like a sentence or a paragraph) into a sequence of tokens. A token is typically a word, but it can also be a subword, character, or a symbol like punctuation. Tokenization is often one of the first steps in text processing in NLP pipelines. The choice of tokenization method can significantly affect the performance of NLP models, as it determines how the model perceives and processes language data. Advanced NLP models, such as BERT or GPT, use sophisticated tokenization techniques to handle a wide range of language processing tasks more effectively."
tokenizer = WordPieceTokenizer(corpus, vocab_size=50)
print("Vocabulary:", tokenizer.vocab)
print("Tokenized:", tokenizer.tokenize("Tokenize this text with this implementation."))
