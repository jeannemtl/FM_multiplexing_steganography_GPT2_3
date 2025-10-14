import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pickle

class MinEntropyCouplingSteganography:
    """
    Real iMEC implementation with context window management.
    """
    
    def __init__(self, model_name='gpt2', block_size_bits=16):
        print("Loading GPT-2 model for iMEC...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.block_size_bits = block_size_bits
        self.n_blocks = None
        self.vocab_size = len(self.tokenizer)
        self.MAX_CONTEXT = 1024  # GPT-2's max context length
        
        print(f"iMEC initialized with {block_size_bits}-bit blocks")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Max context: {self.MAX_CONTEXT} tokens")
    
    def mec_subroutine(self, mu_i, covertext_probs):
        """
        Approximate MEC subroutine.
        """
        vocab_size = len(covertext_probs)
        
        # Ensure inputs are valid
        mu_i = np.array(mu_i)
        covertext_probs = np.array(covertext_probs)
        
        # Normalize
        if mu_i.sum() > 0:
            mu_i = mu_i / mu_i.sum()
        if covertext_probs.sum() > 0:
            covertext_probs = covertext_probs / covertext_probs.sum()
        
        # Sort by covertext probability
        sorted_indices = np.argsort(-covertext_probs)
        
        n_cipher = len(mu_i)
        coupling = {}
        
        cipher_remaining = mu_i.copy()
        covertext_remaining = covertext_probs.copy()
        
        for cipher_idx in range(n_cipher):
            if cipher_remaining[cipher_idx] <= 1e-10:
                continue
                
            for idx in sorted_indices:
                token_idx = int(idx)
                
                # Validate token
                if token_idx < 0 or token_idx >= vocab_size:
                    continue
                
                if covertext_remaining[token_idx] <= 1e-10:
                    continue
                
                # Assign mass
                mass = min(cipher_remaining[cipher_idx], 
                          covertext_remaining[token_idx])
                
                if mass > 1e-10:
                    coupling[(int(cipher_idx), token_idx)] = float(mass)
                
                cipher_remaining[cipher_idx] -= mass
                covertext_remaining[token_idx] -= mass
                
                if cipher_remaining[cipher_idx] <= 1e-10:
                    break
        
        return coupling
    
    def sample_from_coupling(self, coupling, cipher_value):
        """
        Sample token from coupling.
        """
        # Get all possible tokens for this cipher value
        possible = [(t, mass) for (c, t), mass in coupling.items() if c == cipher_value]
        
        if not possible:
            return None
        
        tokens, masses = zip(*possible)
        tokens = np.array(tokens, dtype=int)
        masses = np.array(masses, dtype=float)
        
        # Validate all tokens
        valid_mask = (tokens >= 0) & (tokens < self.vocab_size)
        
        if not np.any(valid_mask):
            return None
        
        # Filter to valid only
        tokens = tokens[valid_mask]
        masses = masses[valid_mask]
        
        # Normalize
        probs = masses / masses.sum()
        
        # Sample
        token = np.random.choice(tokens, p=probs)
        
        return int(token)
    
    def update_posterior(self, coupling, cipher_probs, sampled_token):
        """
        Update posterior distribution.
        """
        n_cipher = len(cipher_probs)
        posterior = np.zeros(n_cipher, dtype=float)
        
        # Find all cipher values that could have generated this token
        possible = [(c, mass) for (c, t), mass in coupling.items() if t == sampled_token]
        
        if not possible:
            return np.ones(n_cipher) / n_cipher
        
        # Apply Bayes rule
        for c, mass in possible:
            if c < n_cipher:
                posterior[c] = mass * cipher_probs[c]
        
        # Normalize
        total = posterior.sum()
        if total > 1e-10:
            posterior = posterior / total
        else:
            posterior = np.ones(n_cipher) / n_cipher
        
        return posterior
    
    def encode_imec(self, ciphertext_bits, context, max_tokens=2000, 
                    entropy_threshold=0.1):
        """
        iMEC encoding with context window management.
        """
        print(f"\n{'='*80}")
        print(f"iMEC ENCODING")
        print(f"{'='*80}")
        
        # Split into blocks
        self.n_blocks = len(ciphertext_bits) // self.block_size_bits
        if len(ciphertext_bits) % self.block_size_bits != 0:
            padding = self.block_size_bits - (len(ciphertext_bits) % self.block_size_bits)
            ciphertext_bits += '0' * padding
            self.n_blocks = len(ciphertext_bits) // self.block_size_bits
        
        print(f"Ciphertext: {len(ciphertext_bits)} bits")
        print(f"Blocks: {self.n_blocks} × {self.block_size_bits} bits")
        
        # Parse blocks
        x_blocks = []
        for i in range(self.n_blocks):
            start = i * self.block_size_bits
            end = start + self.block_size_bits
            block_value = int(ciphertext_bits[start:end], 2)
            x_blocks.append(block_value)
        
        # Initialize uniform distributions
        n_values = 2 ** self.block_size_bits
        mu = [np.ones(n_values) / n_values for _ in range(self.n_blocks)]
        
        # Start generation
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        stegotext_tokens = []
        
        eos_token_id = self.tokenizer.eos_token_id
        failed_samples = 0
        
        with torch.no_grad():
            for j in range(max_tokens):
                # Check stopping condition
                entropies = [self._entropy(mu_i) for mu_i in mu]
                max_entropy = max(entropies)
                
                if max_entropy < entropy_threshold:
                    print(f"\n✓ Stopping: max entropy {max_entropy:.4f} < {entropy_threshold}")
                    break
                
                # Select highest entropy block
                i_star = np.argmax(entropies)
                
                # Get covertext distribution
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                covertext_probs = torch.softmax(logits, dim=0).cpu().numpy()
                
                # Suppress EOS
                covertext_probs[eos_token_id] = 0.0
                covertext_probs = covertext_probs / covertext_probs.sum()
                
                # Build coupling
                gamma_j = self.mec_subroutine(mu[i_star], covertext_probs)
                
                if not gamma_j:
                    failed_samples += 1
                    if failed_samples > 10:
                        print("Too many failed samples, stopping")
                        break
                    continue
                
                # Sample token
                cipher_value = x_blocks[i_star]
                S_j = self.sample_from_coupling(gamma_j, cipher_value)
                
                if S_j is None:
                    S_j = int(np.random.choice(self.vocab_size, p=covertext_probs))
                    failed_samples += 1
                
                # Validate
                if not isinstance(S_j, (int, np.integer)):
                    S_j = int(S_j)
                
                if S_j < 0 or S_j >= self.vocab_size:
                    S_j = int(np.random.choice(self.vocab_size, p=covertext_probs))
                    failed_samples += 1
                
                stegotext_tokens.append(S_j)
                
                # Update posterior
                mu[i_star] = self.update_posterior(gamma_j, mu[i_star], S_j)
                
                # Update context WITH TRUNCATION
                try:
                    new_token_tensor = torch.tensor([[S_j]], device=self.device, dtype=torch.long)
                    input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
                    
                    # CRITICAL: Keep context within GPT-2's limit
                    if input_ids.shape[1] > self.MAX_CONTEXT:
                        input_ids = input_ids[:, -self.MAX_CONTEXT:]
                        if (j + 1) % 500 == 0:
                            print(f"  (Context truncated to {self.MAX_CONTEXT} tokens)")
                    
                except Exception as e:
                    print(f"⚠️  Error updating input_ids: {e}")
                    break
                
                if (j + 1) % 100 == 0:
                    avg_entropy = np.mean(entropies)
                    print(f"  Token {j+1}: max_H={max_entropy:.4f}, avg_H={avg_entropy:.4f}, fails={failed_samples}")
        
        print(f"\n✓ Generated {len(stegotext_tokens)} tokens")
        print(f"✓ Failed samples: {failed_samples}")
        print(f"✓ Final entropies: min={min(entropies):.4f}, max={max(entropies):.4f}, avg={np.mean(entropies):.4f}")
        
        return stegotext_tokens
    
    def decode_imec(self, stegotext_tokens, context, n_blocks, block_size_bits):
        """
        iMEC decoding with context management.
        """
        print(f"\n{'='*80}")
        print(f"iMEC DECODING")
        print(f"{'='*80}")
        
        # Initialize
        n_values = 2 ** block_size_bits
        mu = [np.ones(n_values) / n_values for _ in range(n_blocks)]
        
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        eos_token_id = self.tokenizer.eos_token_id
        
        with torch.no_grad():
            for j, s_j in enumerate(stegotext_tokens):
                entropies = [self._entropy(mu_i) for mu_i in mu]
                i_star = np.argmax(entropies)
                
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                covertext_probs = torch.softmax(logits, dim=0).cpu().numpy()
                
                covertext_probs[eos_token_id] = 0.0
                covertext_probs = covertext_probs / covertext_probs.sum()
                
                gamma_j = self.mec_subroutine(mu[i_star], covertext_probs)
                mu[i_star] = self.update_posterior(gamma_j, mu[i_star], s_j)
                
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[s_j]], device=self.device, dtype=torch.long)
                ], dim=1)
                
                # CRITICAL: Keep context within limit
                if input_ids.shape[1] > self.MAX_CONTEXT:
                    input_ids = input_ids[:, -self.MAX_CONTEXT:]
                
                if (j + 1) % 100 == 0:
                    print(f"  Processed {j+1} tokens")
        
        # Extract MAP estimates
        ciphertext_bits = ""
        for i in range(n_blocks):
            x_i_hat = np.argmax(mu[i])
            bits = format(x_i_hat, f'0{block_size_bits}b')
            ciphertext_bits += bits
        
        print(f"\n✓ Decoded {len(ciphertext_bits)} bits")
        
        return ciphertext_bits
    
    def _entropy(self, probs):
        """Shannon entropy"""
        probs = np.array(probs)
        probs = probs[probs > 1e-10]
        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    imec = MinEntropyCouplingSteganography(block_size_bits=16)
    
    message = "Hi!"
    message_bytes = message.encode('utf-8')
    ciphertext_bits = ''.join(format(byte, '08b') for byte in message_bytes)
    
    print(f"\nTest message: {message}")
    print(f"Bits: {ciphertext_bits} ({len(ciphertext_bits)} bits)")
    
    context = "The future of artificial intelligence"
    tokens = imec.encode_imec(ciphertext_bits, context, max_tokens=500)
    
    print(f"\nGenerated {len(tokens)} tokens")
    text = imec.tokenizer.decode(tokens)
    print(f"Text: {text[:150]}...")
    
    n_blocks = len(ciphertext_bits) // 16 + (1 if len(ciphertext_bits) % 16 else 0)
    recovered = imec.decode_imec(tokens, context, n_blocks, 16)
    
    recovered_bytes = bytes(int(recovered[i:i+8], 2) 
                           for i in range(0, len(message_bytes)*8, 8))
    recovered_msg = recovered_bytes.decode('utf-8', errors='ignore')
    
    print(f"\nRecovered: {recovered_msg}")
    print(f"Match: {message == recovered_msg}")
