import regex as re
import json
import os

class TurkishBPETokenizer:
    def __init__(self):
        self.merges = {} 
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.special_tokens = {} 
        self.cache = {}
        # GPT-4 stili pre-tokenization pattern (Türkçe uyumlu)
        self.base_pattern = r"""'?[^\s\p{L}\p{N}]+| ?\p{L}+| ?\p{N}+|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.base_pattern)

    def train(self, text, vocab_size=512):
        tokens = re.findall(self.base_pattern, text)
        ids_list = [list(t.encode("utf-8")) for t in tokens]

        num_merges = vocab_size - 256
        
        for i in range(num_merges):
            stats = {}
            for chunk in ids_list:
                for pair in zip(chunk, chunk[1:]):
                    stats[pair] = stats.get(pair, 0) + 1
            
            if not stats:
                break
            
            top_pair = max(stats, key=stats.get)
            new_id = 256 + i
            
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            
            new_ids_list = []
            for chunk in ids_list:
                new_chunk = []
                idx = 0
                while idx < len(chunk):
                    if idx < len(chunk) - 1 and chunk[idx] == top_pair[0] and chunk[idx+1] == top_pair[1]:
                        new_chunk.append(new_id)
                        idx += 2
                    else:
                        new_chunk.append(chunk[idx])
                        idx += 1
                new_ids_list.append(new_chunk)
            ids_list = new_ids_list

    def add_special_tokens(self, tokens_list):
        current_max_id = max(self.vocab.keys()) if self.vocab else 255
        start_id = current_max_id + 1

        for i, token in enumerate(tokens_list):
            new_id = start_id + i
            self.special_tokens[token] = new_id
            self.vocab[new_id] = token.encode("utf-8")
        
        special_pattern = "|".join(re.escape(k) for k in self.special_tokens)
        self.compiled_pattern = re.compile(f"({special_pattern})|({self.base_pattern})")

    def _encode_chunk(self, token_bytes):
        if token_bytes in self.cache:
            return self.cache[token_bytes]

        ids = list(token_bytes)
        while len(ids) >= 2:
            stats = {}
            for pair in zip(ids, ids[1:]):
                stats[pair] = stats.get(pair, 0) + 1
            
            pair_to_merge = None
            min_rank = float("inf")
            
            for pair in stats:
                if pair in self.merges:
                    if self.merges[pair] < min_rank:
                        min_rank = self.merges[pair]
                        pair_to_merge = pair
            
            if not pair_to_merge:
                break
            
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair_to_merge[0] and ids[i+1] == pair_to_merge[1]:
                    new_ids.append(self.merges[pair_to_merge])
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        
        self.cache[token_bytes] = ids
        return ids

    def encode(self, text):
        if self.special_tokens:
            chunks = self.compiled_pattern.findall(text)
            final_ids = []
            for match in chunks:
                special_part = match[0]
                normal_part = match[1] if len(match) > 1 else ""
                
                if special_part:
                    final_ids.append(self.special_tokens[special_part])
                elif normal_part:
                    final_ids.extend(self._encode_chunk(normal_part.encode("utf-8")))
                elif isinstance(match, str) and match in self.special_tokens:
                     final_ids.append(self.special_tokens[match])
                elif isinstance(match, str):
                     final_ids.extend(self._encode_chunk(match.encode("utf-8")))     
            return final_ids
        else:
            tokens = re.findall(self.base_pattern, text)
            final_ids = []
            for token in tokens:
                final_ids.extend(self._encode_chunk(token.encode("utf-8")))
            return final_ids

    def decode(self, ids):
        parts = []
        for idx in ids:
            if idx in self.vocab:
                parts.append(self.vocab[idx])
            else:
                parts.append(b"")
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        merges_export = {f"{p0} {p1}": idx for (p0, p1), idx in self.merges.items()}
        
        with open(os.path.join(directory, "merges.json"), "w", encoding="utf-8") as f:
            json.dump(merges_export, f, ensure_ascii=False, indent=2)

        with open(os.path.join(directory, "special_tokens.json"), "w", encoding="utf-8") as f:
            json.dump(self.special_tokens, f, ensure_ascii=False, indent=2)

    def load(self, directory):
        merges_path = os.path.join(directory, "merges.json")
        special_path = os.path.join(directory, "special_tokens.json")

        if os.path.exists(merges_path):
            with open(merges_path, "r", encoding="utf-8") as f:
                merges_import = json.load(f)
            
            self.merges = {}
            for key, idx in merges_import.items():
                p0, p1 = map(int, key.split())
                self.merges[(p0, p1)] = idx
        
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        if os.path.exists(special_path):
            with open(special_path, "r", encoding="utf-8") as f:
                special_tokens_import = json.load(f)
            
            tokens_to_add = list(special_tokens_import.keys())
            if tokens_to_add:
                self.add_special_tokens(tokens_to_add)
                for token, tid in special_tokens_import.items():
                    self.special_tokens[token] = tid
                    self.vocab[tid] = token.encode("utf-8")