from typing import List, Dict, Tuple

from collections import defaultdict
from transformers import AutoTokenizer


SPECIAL_TOKENS = ["<|endoftext|>"]


class BytePairEncoding:
    def __init__(self, tokenizer_name: str = "gpt2", target_vocab_size: int = 1000):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.target_vocab_size: int = target_vocab_size
        self.pairs_to_merge: List[str] = []
        self.vocab: List[str] = SPECIAL_TOKENS

    def _get_words_from_tokenizer(self, text: str) -> List[str]:
        """Extract the tokenized words from the input text with tokenizer"""
        pre_tokenize_result = self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(
            text
        )
        words = [word for word, _ in pre_tokenize_result]
        return words

    def _get_word_freqs(self, corpus: List[str]) -> Dict[str, int]:
        """Get the frequency of each word in the corpus"""
        word_freqs = defaultdict(int)
        for text in corpus:
            words = self._get_words_from_tokenizer(text)
            for word in words:
                word_freqs[word] += 1
        return word_freqs

    def _get_alphabets(self, words: List[str]) -> List[str]:
        """Find all characters that appear in the words"""
        alphabets = set()
        for word in words:
            for char in word:
                alphabets.add(char)
        return list(alphabets)

    def _compute_pair_freqs(
        self, word_splits: Dict[str, List[str]], word_freqs: Dict[str, int]
    ) -> Dict[str, int]:
        """Compute the frequency of each pair of characters with word_freqs"""
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = word_splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_pair(
        self, target_a: str, target_b: str, word_splits: Dict[str, List[str]]
    ) -> Dict:
        """Merge the pair of characters in the word_splits"""
        for word, split in word_splits.items():
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == target_a and split[i + 1] == target_b:
                    split = split[:i] + [target_a + target_b] + split[i + 2 :]
                else:
                    i += 1
            word_splits[word] = split
        return word_splits

    def fit(self, corpus: List[str]):
        word_freqs = self._get_word_freqs(corpus)
        words = list(word_freqs.keys())
        alphabets = self._get_alphabets(words)
        self.vocab = self.vocab + alphabets
        word_splits = {word: list(word) for word in words}

        while len(self.vocab) < self.target_vocab_size:
            pair_freqs = self._compute_pair_freqs(word_splits, word_freqs)
            best_pair, best_freq = "", 0
            for pair, freq in pair_freqs.items():
                if freq > best_freq:
                    best_pair, best_freq = pair, freq
            word_splits = self._merge_pair(best_pair[0], best_pair[1], word_splits)
            merged_chars = best_pair[0] + best_pair[1]  # concat two pairs
            self.pairs_to_merge.append(best_pair)
            self.vocab.append(merged_chars)

    def tokenize(self, text: str) -> List[str]:
        words = self._get_words_from_tokenizer(text)
        word_splits = {word: list(word) for word in words}
        for pair_tuple in self.pairs_to_merge:
            word_splits = self._merge_pair(pair_tuple[0], pair_tuple[1], word_splits)
        splits = list(word_splits.values())
        return sum(splits, [])
