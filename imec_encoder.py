import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import defaultdict
import pickle

class MinEntropyCouplingSteganography:
    """
    Real iMEC implementation based on Algorithm 1 & 2 from the paper.
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
        
        print(f"iMEC initialized with {block_size_bits}-bit blocks")
    
    def mec_subroutine(self, mu_i, covertext_probs):
        """
        Approximate MEC subroutine from Kocaoglu et al. 2017.
        """
        vocab_size = len(covertext_probs)
        sorted_indices = np.argsort(-covertext_probs)
        
        n_cipher = len(mu_i)
        coupling = {}
        
        cipher_remaining = mu_i.copy()
        covertext_remaining = covertext_probs.copy()
        
        for cipher_idx in range(n_cipher):
            if cipher_remaining[cipher_idx] <= 0:
                continue
                
            for token_idx in sorted_indices:
                # Convert to Python int and bounds check
                token_idx = int(token_idx)
                if token_idx < 0 or token_idx >= vocab_size:
                    continue
                
                if covertext_remaining[token_idx] <= 0:
                    continue
                
                mass = min(cipher_remaining[cipher_idx], 
                          covertext_remaining[token_idx])
                
                # Store as Python ints and floats
                coupling[(int(cipher_idx), token_idx)] = float(mass)
                
                cipher_remaining[cipher_idx] -= mass
                covertext_remaining[token_idx] -= mass
                
                if cipher_remaining[cipher_idx] <= 1e-10:
                    break
        
        return coupling
    
    def sample_from_coupling(self, coupling, cipher_value):
        """
        Sample a covertext token given ciphertext value and coupling.
        """
        relevant_pairs = [(c, t, mass) for (c, t), mass in coupling.items() 
                         if c == cipher_value]
        
        if not relevant_pairs:
            raise ValueError(f"No coupling found for cipher value {cipher_value}")
        
        tokens = [t for _, t, _ in relevant_pairs]
        masses = np.array([mass for _, _, mass in relevant_pairs])
        probs = masses / masses.sum()
        
        # CRITICAL FIX: Validate tokens are in vocabulary
        vocab_size = self.tokenizer.vocab_size  # ~50257 for GPT-2
        valid_tokens = [t for t in tokens if 0 <= t < vocab_size]
        
        if not valid_tokens:
            # Fallback: return a safe token
            print(f"⚠️  Warning: No valid tokens in coupling, using fallback")
            return np.random.randint(0, min(1000, vocab_size))
        
        # Re-normalize probabilities for valid tokens only
        valid_indices = [i for i, t in enumerate(tokens) if 0 <= t < vocab_size]
        valid_probs = probs[valid_indices]
        valid_probs = valid_probs / valid_probs.sum()
        
        token = np.random.choice(valid_tokens, p=valid_probs)
        
        return int(token)
    
    def update_posterior(self, coupling, cipher_probs, sampled_token):
        """
        Update ciphertext posterior after observing sampled token.
        """
        n_cipher = len(cipher_probs)
        posterior = np.zeros(n_cipher)
        
        relevant_pairs = [(c, t, mass) for (c, t), mass in coupling.items() 
                         if t == sampled_token]
        
        if not relevant_pairs:
            return np.ones(n_cipher) / n_cipher
        
        for c, _, mass in relevant_pairs:
            posterior[c] = mass * cipher_probs[c]
        
        if posterior.sum() > 0:
            posterior = posterior / posterior.sum()
        else:
            posterior = np.ones(n_cipher) / n_cipher
        
        return posterior
    
    def encode_imec(self, ciphertext_bits, context, max_tokens=1000, 
                    entropy_threshold=0.1):
        """
        iMEC encoding: Algorithm 1 from the paper.
        """
        print(f"\n{'='*80}")
        print(f"iMEC ENCODING")
        print(f"{'='*80}")
        
        # Split ciphertext into blocks
        self.n_blocks = len(ciphertext_bits) // self.block_size_bits
        if len(ciphertext_bits) % self.block_size_bits != 0:
            padding = self.block_size_bits - (len(ciphertext_bits) % self.block_size_bits)
            ciphertext_bits += '0' * padding
            self.n_blocks = len(ciphertext_bits) // self.block_size_bits
        
        print(f"Ciphertext: {len(ciphertext_bits)} bits")
        print(f"Blocks: {self.n_blocks} × {self.block_size_bits} bits")
        
        # Parse ciphertext into blocks
        x_blocks = []
        for i in range(self.n_blocks):
            start = i * self.block_size_bits
            end = start + self.block_size_bits
            block_bits = ciphertext_bits[start:end]
            block_value = int(block_bits, 2)
            x_blocks.append(block_value)
        
        # Initialize uniform distributions
        mu = []
        for i in range(self.n_blocks):
            n_values = 2 ** self.block_size_bits
            mu_i = np.ones(n_values) / n_values
            mu.append(mu_i)
        
        # Start generation
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        stegotext_tokens = []
        
        eos_token_id = self.tokenizer.eos_token_id
        
        with torch.no_grad():
            for j in range(max_tokens):
                # Select block with maximum entropy
                entropies = [self._entropy(mu_i) for mu_i in mu]
                i_star = np.argmax(entropies)
                
                # Check stopping condition
                if entropies[i_star] < entropy_threshold:
                    print(f"\n✓ Stopping: All blocks have H < {entropy_threshold}")
                    break
                
                # Get GPT-2 probabilities
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                covertext_probs = torch.softmax(logits, dim=0).cpu().numpy()
                
                # Suppress EOS
                covertext_probs[eos_token_id] = 0.0
                covertext_probs = covertext_probs / covertext_probs.sum()
                
                # Call MEC subroutine
                gamma_j = self.mec_subroutine(mu[i_star], covertext_probs)
                
                # Sample token
                cipher_value = x_blocks[i_star]
                
                try:
                    S_j = self.sample_from_coupling(gamma_j, cipher_value)
                except Exception as e:
                    print(f"⚠️  Error sampling token: {e}")
                    # Fallback: sample from covertext distribution
                    S_j = np.random.choice(len(covertext_probs), p=covertext_probs)
                
                stegotext_tokens.append(S_j)
                
                # Update posterior
                mu[i_star] = self.update_posterior(gamma_j, mu[i_star], S_j)
                
                # Update input
                input_ids = torch.cat([
                    input_ids, 
                    torch.tensor([[S_j]], device=self.device)
                ], dim=1)
                
                if (j + 1) % 100 == 0:
                    print(f"  Generated {j+1} tokens, max entropy: {entropies[i_star]:.4f}")
        
        print(f"\n✓ Generated {len(stegotext_tokens)} tokens")
        print(f"✓ Final entropies: {[f'{self._entropy(mu_i):.4f}' for mu_i in mu[:5]]}")
        
        return stegotext_tokens
    
    def decode_imec(self, stegotext_tokens, context, n_blocks, block_size_bits):
        """
        iMEC decoding: Algorithm 2 from the paper.
        """
        print(f"\n{'='*80}")
        print(f"iMEC DECODING")
        print(f"{'='*80}")
        
        # Initialize uniform distributions
        mu = []
        for i in range(n_blocks):
            n_values = 2 ** block_size_bits
            mu_i = np.ones(n_values) / n_values
            mu.append(mu_i)
        
        # Process stegotext
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        
        eos_token_id = self.tokenizer.eos_token_id
        
        with torch.no_grad():
            for j, s_j in enumerate(stegotext_tokens):
                # Select block with maximum entropy
                entropies = [self._entropy(mu_i) for mu_i in mu]
                i_star = np.argmax(entropies)
                
                # Get GPT-2 probabilities
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                covertext_probs = torch.softmax(logits, dim=0).cpu().numpy()
                
                # Suppress EOS
                covertext_probs[eos_token_id] = 0.0
                covertext_probs = covertext_probs / covertext_probs.sum()
                
                # Call MEC subroutine
                gamma_j = self.mec_subroutine(mu[i_star], covertext_probs)
                
                # Update posterior
                mu[i_star] = self.update_posterior(gamma_j, mu[i_star], s_j)
                
                # Update input
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[s_j]], device=self.device)
                ], dim=1)
                
                if (j + 1) % 100 == 0:
                    print(f"  Processed {j+1} tokens")
        
        # Extract most likely ciphertext values (MAP estimate)
        ciphertext_bits = ""
        for i in range(n_blocks):
            x_i_hat = np.argmax(mu[i])
            bits = format(x_i_hat, f'0{block_size_bits}b')
            ciphertext_bits += bits
        
        print(f"\n✓ Decoded {len(ciphertext_bits)} bits")
        
        return ciphertext_bits
    
    def _entropy(self, probs):
        """Compute Shannon entropy."""
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))


# ============================================================================
# TEST SCRIPT
# ============================================================================

if __name__ == "__main__":
    imec = MinEntropyCouplingSteganography(block_size_bits=16)
    
    message = "Hello iMEC!"
    message_bytes = message.encode('utf-8')
    ciphertext_bits = ''.join(format(byte, '08b') for byte in message_bytes)
    
    print(f"\nOriginal message: {message}")
    print(f"Ciphertext: {ciphertext_bits} ({len(ciphertext_bits)} bits)")
    
    context = "The future of artificial intelligence"
    stegotext_tokens = imec.encode_imec(ciphertext_bits, context, max_tokens=1000)
    
    stegotext = imec.tokenizer.decode(stegotext_tokens)
    print(f"\nStegotext: {stegotext[:200]}...")
    
    n_blocks = len(ciphertext_bits) // 16 + (1 if len(ciphertext_bits) % 16 else 0)
    recovered_bits = imec.decode_imec(stegotext_tokens, context, n_blocks, 16)
    
    recovered_bytes = bytes(int(recovered_bits[i:i+8], 2) 
                           for i in range(0, len(message_bytes)*8, 8))
    recovered_message = recovered_bytes.decode('utf-8', errors='ignore')
    
    print(f"\nRecovered message: {recovered_message}")
    print(f"Match: {message == recovered_message}")
