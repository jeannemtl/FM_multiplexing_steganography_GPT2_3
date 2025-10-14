import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pickle
from imec_encoder import MinEntropyCouplingSteganography

class FrequencyiMECHybridEncoder:
    """
    Complete hybrid: Frequency modulation + iMEC obfuscation
    
    Pipeline:
    1. Encode 3 messages via frequency modulation
    2. Convert frequency-modulated tokens to binary
    3. Apply iMEC to hide frequency patterns
    4. Decode: iMEC → recover tokens → FFT analysis → extract messages
    """
    
    def __init__(self, model_name='gpt2'):
        print("Loading GPT-2 for hybrid encoding...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Agent configuration
        self.agents = {
            'ALICE': {'freq': 0.02, 'bits': None},
            'BOB': {'freq': 0.04, 'bits': None},
            'CHARLIE': {'freq': 0.06, 'bits': None}
        }
        
        # iMEC encoder
        self.imec = MinEntropyCouplingSteganography(block_size_bits=16)
        
        print(f"Hybrid encoder ready on {self.device}")
    
    def encode_ask_smooth(self, bits, carrier_freq, sequence_length, transition_tokens=5):
        """ASK modulation with smooth transitions."""
        tokens_per_bit = sequence_length // len(bits)
        bias_signal = np.zeros(sequence_length)
        
        for i, bit in enumerate(bits):
            start = i * tokens_per_bit
            end = min((i + 1) * tokens_per_bit, sequence_length)
            
            amplitude_target = 0.8 if bit == 1 else 0.2
            window_length = end - start
            amplitude = np.ones(window_length) * amplitude_target
            
            if i > 0:
                prev_amp = 0.8 if bits[i-1] == 1 else 0.2
                for j in range(min(transition_tokens, window_length)):
                    t_norm = j / transition_tokens
                    amplitude[j] = prev_amp * (1 - t_norm) + amplitude_target * t_norm
            
            t = np.arange(window_length)
            carrier = np.sin(2 * np.pi * carrier_freq * (start + t))
            bias_signal[start:end] = amplitude * carrier
        
        return bias_signal
    
    def generate_frequency_modulated(self, context, messages, sequence_length=100, 
                                     bias_strength=0.5):
        """
        STAGE 1: Generate frequency-modulated stegotext.
        """
        print("\n" + "="*80)
        print("STAGE 1: FREQUENCY MODULATION")
        print("="*80)
        
        for agent, bits in messages.items():
            self.agents[agent]['bits'] = np.array(bits)
            print(f"{agent:8s}: {bits}")
        
        # Generate bias signals
        bias_signals = {}
        for agent_name, agent_info in self.agents.items():
            bits = agent_info['bits']
            freq = agent_info['freq']
            bias_signals[agent_name] = self.encode_ask_smooth(
                bits, freq, sequence_length
            )
        
        combined_bias = sum(bias_signals.values())
        
        # Generate text
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        freq_tokens = []
        eos_token_id = self.tokenizer.eos_token_id
        
        with torch.no_grad():
            for i in range(sequence_length):
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                base_probs = torch.softmax(logits, dim=0)
                
                # Suppress EOS
                base_probs[eos_token_id] = 0.0
                base_probs = base_probs / base_probs.sum()
                
                # Apply frequency bias
                biased_probs = base_probs * (1 + bias_strength * combined_bias[i])
                biased_probs = biased_probs / biased_probs.sum()
                
                next_token = torch.multinomial(biased_probs, num_samples=1)
                freq_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                if (i + 1) % 200 == 0:
                    print(f"  Generated {i+1} tokens...")
        
        freq_text = self.tokenizer.decode(freq_tokens)
        
        print(f"\n✓ Frequency modulation complete: {len(freq_tokens)} tokens")
        
        return freq_text, freq_tokens, bias_signals
    
    def apply_imec_obfuscation(self, freq_tokens, context):
        """
        STAGE 2: Apply iMEC to hide frequency patterns.
        """
        print("\n" + "="*80)
        print("STAGE 2: iMEC OBFUSCATION")
        print("="*80)
        
        # Convert frequency tokens to binary ciphertext
        print(f"\nConverting {len(freq_tokens)} tokens to binary...")
        ciphertext_bits = ''.join(format(token, '016b') for token in freq_tokens)
        print(f"Ciphertext: {len(ciphertext_bits)} bits")
        
        # Apply iMEC encoding
        obfuscated_tokens = self.imec.encode_imec(
            ciphertext_bits, 
            context, 
            max_tokens=2000,
            entropy_threshold=0.1
        )
        
        obfuscated_text = self.tokenizer.decode(obfuscated_tokens)
        
        print(f"\n✓ iMEC obfuscation complete")
        print(f"✓ Obfuscated to {len(obfuscated_tokens)} tokens")
        
        # Store metadata for decoding
        metadata = {
            'n_freq_tokens': len(freq_tokens),
            'block_size_bits': self.imec.block_size_bits,
            'n_blocks': self.imec.n_blocks
        }
        
        return obfuscated_text, obfuscated_tokens, metadata
    
    def imec_decode_tokens(self, obfuscated_tokens, context, metadata):
        """
        STAGE 3: Decode iMEC to recover frequency-modulated tokens.
        """
        print("\n" + "="*80)
        print("STAGE 3: iMEC DECODING")
        print("="*80)
        
        # Decode with iMEC
        recovered_bits = self.imec.decode_imec(
            obfuscated_tokens,
            context,
            metadata['n_blocks'],
            metadata['block_size_bits']
        )
        
        # Convert binary back to tokens
        n_tokens = metadata['n_freq_tokens']
        recovered_tokens = []
        
        for i in range(n_tokens):
            start = i * 16
            end = start + 16
            if end <= len(recovered_bits):
                token_bits = recovered_bits[start:end]
                token_id = int(token_bits, 2)
                recovered_tokens.append(token_id)
        
        print(f"\n✓ Recovered {len(recovered_tokens)} tokens")
        
        return recovered_tokens
    
    def encode_hybrid(self, context, messages, sequence_length=100):
        """
        Complete hybrid encoding pipeline.
        """
        print("\n" + "="*80)
        print("HYBRID ENCODER: FREQUENCY + iMEC")
        print("="*80)
        
        # Stage 1: Frequency modulation
        freq_text, freq_tokens, bias_signals = self.generate_frequency_modulated(
            context, messages, sequence_length
        )
        
        # Stage 2: iMEC obfuscation
        obf_text, obf_tokens, metadata = self.apply_imec_obfuscation(
            freq_tokens, context
        )
        
        # Save everything
        output_data = {
            'context': context,
            'messages': messages,
            'freq_text': freq_text,
            'freq_tokens': freq_tokens,
            'bias_signals': bias_signals,
            'obfuscated_text': obf_text,
            'obfuscated_tokens': obf_tokens,
            'metadata': metadata
        }
        
        with open('hybrid_freq_imec_data.pkl', 'wb') as f:
            pickle.dump(output_data, f)
        
        print("\n" + "="*80)
        print("ENCODING COMPLETE")
        print("="*80)
        print(f"✓ Frequency-modulated: {len(freq_tokens)} tokens")
        print(f"✓ iMEC obfuscated: {len(obf_tokens)} tokens")
        print(f"✓ Saved to: hybrid_freq_imec_data.pkl")
        
        return output_data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    encoder = FrequencyiMECHybridEncoder()
    
    # Define messages
    messages = {
        'ALICE':   [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        'BOB':     [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        'CHARLIE': [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0]
    }
    
    context = "The future of artificial intelligence"
    
    # Encode with hybrid system
    output_data = encoder.encode_hybrid(
        context=context,
        messages=messages,
        sequence_length=100
    )
    
    print(f"\nFrequency text preview:")
    print(f"  {output_data['freq_text'][:150]}...")
    print(f"\nObfuscated text preview:")
    print(f"  {output_data['obfuscated_text'][:150]}...")
