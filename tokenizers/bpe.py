import re


class BytePairEncoding:
    def __init__(self, corpus, num_merges):
        self.num_merges = num_merges
        self.vocab = self._build_vocab(corpus)
        self.train()

    def _build_vocab(self, corpus):
        # Initialize vocabulary with character frequency
        vocab = {}
        for word in corpus.strip().split():
            # For each word, count character frequency and add a special token </w> at the end
            chars = list(word)
            chars.append('</w>')
            for char in chars:
                if char in vocab:
                    vocab[char] += 1
                else:
                    vocab[char] = 1
        return vocab

    def _get_pairs(self, vocab):
        # Get pairs of adjacent symbols from the vocabulary
        pairs = {}
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in pairs:
                    pairs[pair] += freq
                else:
                    pairs[pair] = freq
        return pairs

    def _merge_vocab(self, pair, vocab):
        # Merge all occurrences of the most frequent pair
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def train(self):
        # Train BPE algorithm
        for i in range(self.num_merges):
            pairs = self._get_pairs(self.vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.vocab = self._merge_vocab(best_pair, self.vocab)
        return self.vocab

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
bpe = BytePairEncoding(corpus, num_merges=1)
encoded_word = bpe.tokenize("Tokenize this text with this implementation.")
print("Encoded:", encoded_word)
