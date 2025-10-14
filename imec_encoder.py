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
        self.n_blocks = None  # Will be set based on message length
        
        print(f"iMEC initialized with {block_size_bits}-bit blocks")
    
    def mec_subroutine(self, mu_i, covertext_probs):
        """
        Approximate MEC subroutine from Kocaoglu et al. 2017.
        
        This is the core MEC algorithm that couples:
        - mu_i: Uniform distribution over ciphertext block X_i
        - covertext_probs: GPT-2 probability distribution over next token
        
        Returns a coupling gamma_j.
        """
        # Get sorted indices by covertext probability (descending)
        vocab_size = len(covertext_probs)
        sorted_indices = np.argsort(-covertext_probs)  # Descending order
        
        # Number of ciphertext values
        n_cipher = len(mu_i)
        
        # Initialize coupling
        coupling = {}
        
        # Greedy assignment (Algorithm 1 from Kocaoglu et al.)
        cipher_remaining = mu_i.copy()
        covertext_remaining = covertext_probs.copy()
        
        for cipher_idx in range(n_cipher):
            if cipher_remaining[cipher_idx] <= 0:
                continue
                
            # Find token with highest remaining probability
            for token_idx in sorted_indices:
                if covertext_remaining[token_idx] <= 0:
                    continue
                
                # Assign mass
                mass = min(cipher_remaining[cipher_idx], 
                          covertext_remaining[token_idx])
                
                coupling[(cipher_idx, token_idx)] = mass
                
                cipher_remaining[cipher_idx] -= mass
                covertext_remaining[token_idx] -= mass
                
                if cipher_remaining[cipher_idx] <= 1e-10:
                    break
        
        return coupling
    
    def sample_from_coupling(self, coupling, cipher_value):
        """
        Sample a covertext token given ciphertext value and coupling.
        
        P(C_j | X_i = cipher_value) from the coupling gamma_j
        """
        # Get all (cipher, token) pairs where cipher matches
        relevant_pairs = [(c, t, mass) for (c, t), mass in coupling.items() 
                         if c == cipher_value]
        
        if not relevant_pairs:
            raise ValueError(f"No coupling found for cipher value {cipher_value}")
        
        # Extract tokens and their conditional probabilities
        tokens = [t for _, t, _ in relevant_pairs]
        masses = np.array([mass for _, _, mass in relevant_pairs])
        
        # Normalize to get conditional distribution
        probs = masses / masses.sum()
        
        # Sample token
        token = np.random.choice(tokens, p=probs)
        
        return token
    
    def update_posterior(self, coupling, cipher_probs, sampled_token):
        """
        Update ciphertext posterior after observing sampled token.
        
        Returns: gamma_j(X_i* | C_j = sampled_token)
        """
        n_cipher = len(cipher_probs)
        posterior = np.zeros(n_cipher)
        
        # Get all cipher values that could have generated this token
        relevant_pairs = [(c, t, mass) for (c, t), mass in coupling.items() 
                         if t == sampled_token]
        
        if not relevant_pairs:
            # Token not in coupling - return uniform
            return np.ones(n_cipher) / n_cipher
        
        # Compute posterior using Bayes rule
        for c, _, mass in relevant_pairs:
            posterior[c] = mass * cipher_probs[c]
        
        # Normalize
        if posterior.sum() > 0:
            posterior = posterior / posterior.sum()
        else:
            posterior = np.ones(n_cipher) / n_cipher
        
        return posterior
    
    def encode_imec(self, ciphertext_bits, context, max_tokens=1000, 
                    entropy_threshold=0.1):
        """
        iMEC encoding: Algorithm 1 from the paper.
        
        Args:
            ciphertext_bits: Binary string (e.g., "0110101...")
            context: Initial context string
            max_tokens: Maximum tokens to generate
            entropy_threshold: Stop when all blocks have H(mu_i) < threshold
        
        Returns:
            stegotext_tokens: List of token IDs
        """
        print(f"\n{'='*80}")
        print(f"iMEC ENCODING")
        print(f"{'='*80}")
        
        # Split ciphertext into blocks
        self.n_blocks = len(ciphertext_bits) // self.block_size_bits
        if len(ciphertext_bits) % self.block_size_bits != 0:
            # Pad with zeros
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
        
        # Initialize uniform distributions over each block
        mu = []
        for i in range(self.n_blocks):
            n_values = 2 ** self.block_size_bits
            mu_i = np.ones(n_values) / n_values  # Uniform
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
                
                # Sample token: S_j ~ gamma_j(C_j | X_i* = x_i*)
                cipher_value = x_blocks[i_star]
                S_j = self.sample_from_coupling(gamma_j, cipher_value)
                
                stegotext_tokens.append(S_j)
                
                # Update posterior: mu_i* <- gamma_j(X_i* | C_j = S_j)
                mu[i_star] = self.update_posterior(gamma_j, mu[i_star], S_j)
                
                # Update input for next iteration
                input_ids = torch.cat([
                    input_ids, 
                    torch.tensor([[S_j]], device=self.device)
                ], dim=1)
                
                if (j + 1) % 100 == 0:
                    print(f"  Generated {j+1} tokens, max entropy: {entropies[i_star]:.4f}")
        
        print(f"\n✓ Generated {len(stegotext_tokens)} tokens")
        print(f"✓ Final entropies: {[f'{self._entropy(mu_i):.4f}' for mu_i in mu]}")
        
        return stegotext_tokens
    
    def decode_imec(self, stegotext_tokens, context, n_blocks, block_size_bits):
        """
        iMEC decoding: Algorithm 2 from the paper.
        
        Args:
            stegotext_tokens: List of token IDs
            context: Initial context string
            n_blocks: Number of ciphertext blocks
            block_size_bits: Bits per block
        
        Returns:
            ciphertext_bits: Recovered binary string
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
        probs = probs[probs > 0]  # Remove zeros
        return -np.sum(probs * np.log2(probs))


# ============================================================================
# TEST SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Initialize iMEC
    imec = MinEntropyCouplingSteganography(block_size_bits=16)
    
    # Prepare message
    message = "Hello iMEC!"
    message_bytes = message.encode('utf-8')
    ciphertext_bits = ''.join(format(byte, '08b') for byte in message_bytes)
    
    print(f"\nOriginal message: {message}")
    print(f"Ciphertext: {ciphertext_bits} ({len(ciphertext_bits)} bits)")
    
    # Encode
    context = "The future of artificial intelligence"
    stegotext_tokens = imec.encode_imec(ciphertext_bits, context, max_tokens=1000)
    
    stegotext = imec.tokenizer.decode(stegotext_tokens)
    print(f"\nStegotext: {stegotext[:200]}...")
    
    # Decode
    n_blocks = len(ciphertext_bits) // 16 + (1 if len(ciphertext_bits) % 16 else 0)
    recovered_bits = imec.decode_imec(stegotext_tokens, context, n_blocks, 16)
    
    # Convert back to message
    recovered_bytes = bytes(int(recovered_bits[i:i+8], 2) 
                           for i in range(0, len(message_bytes)*8, 8))
    recovered_message = recovered_bytes.decode('utf-8', errors='ignore')
    
    print(f"\nRecovered message: {recovered_message}")
    print(f"Match: {message == recovered_message}")
