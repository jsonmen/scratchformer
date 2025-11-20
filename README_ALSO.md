# Transformer Reimplementation: A Hands-On Journey to BLEU 10

Welcome to the *real* README—where I dive deep into my Transformer reimplementation, sharing not just the code but the gritty problems I hit on the road to achieving a BLEU score of 10 (and how I clawed my way through them). This isn't a dry spec sheet; it's a story of trial, error, and "aha!" moments. Whether you're a fellow coder debugging your own model or just curious about the magic behind modern AI, stick around. I'll break down each component, highlight key math, and flag the pitfalls that nearly derailed me—like nan gradients in attention or OOM.

## 📚 Core Code Components
<img src="https://i.imgur.com/6JJJ2Ct.png" alt="Multi-Head Attention Visualization" style="width:500px;"/><br>
This section unpacks the building blocks of my Transformer. I stuck close to the original "Attention Is All You Need" paper (Vaswani et al., 2017), but with practical tweaks for clarity and debugging

### 🧠 Multi-Head Attention: `multi_head_attention.py`

At the heart of the Transformer lies **Scaled Dot-Product Attention** (Section 3.2.1 of the paper), which lets the model focus on relevant parts of the input sequence. But to supercharge it, we use multi-head attention: it runs attention in parallel "heads" (I set `num_heads=4` by default), capturing diverse relationships like syntax in one head and semantics in another.

<img src="https://i.imgur.com/6uUSHqW.png" alt="Multi-Head Attention Visualization" style="width:600px;"/>

The core equations:
$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$
$$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W_O 
\\ \text{where}\; head_i \; = Attention(QW^{Q}_i, KW^{K}_i, VW^{V}_i)$$

**My Twis**: Instead of computing Q, K, V per head (inefficient!), I project the full input once, then use `_split_heads` to reshape into [batch, heads, seq_len, d_k]. This batches everything for GPU speed. At the end, `_combine_heads` concatenates and applies $W_O$ for the final output

### ⚡ Feed-Forward Network: `ffn.py`

The **Position-wise Feed-Forward Network** (FFN) is essentially a two-layer MLP applied identically to each position—fancy talk for "non-linear booster shot" after attention. It's where the model injects creativity: in NMT (like my en-de setup), it refines contextual embeddings; in LLMs, it's rumored to stash factual knowledge.

The math is straightforward:
$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

<img src="https://i.imgur.com/2WToZFk.png" alt="FFN Visualization" style="width:400px;"/>


### 🌊 Positional Encoding: `positional_encoding.py`

Transformers are permutation-invariant without this—words could shuffle like a bad playlist! **Positional Encoding** injects sequence order using fixed sine/cosine waves (Section 3.5), added directly to embeddings.

Equations:
$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

**Why It Works**: Low frequencies for long-range deps, high for short—scalable to any length.

### 🏗️ Transformer Encoder: `encoder.py`

The encoder stack (6 layers default) "understands" the source sequence, producing rich representations for the decoder. Each layer: multi-head self-attention + FFN + layer norm + residuals.

<img src="https://i.imgur.com/BD8xY63.png" alt="Encoder Architecture" style="width:300px;"/>


**Key Flow**: Input $\rightarrow$ Embed + PosEnc $\rightarrow$ Self-Attn $\rightarrow$ Add&Norm $\rightarrow$ LayerNorm $\rightarrow$ FFN $\rightarrow$ Add&Norm $\rightarrow$ Output.

### 🎭 Transformer Decoder: `decoder.py`

Here's the "magic": The decoder generates output autoregressively, using encoder keys/values for cross-attention. Stacked like the encoder, but with masked self-attention (no future peeks) + cross attention.

<img src="https://i.imgur.com/XzWHwf9.png" alt="Decoder Architecture" style="width:300px;"/>

**Key Flow**: Input $\rightarrow$ Embed + PosEnc $\rightarrow$ Masked Self-Attn $\rightarrow$ Add&Norm $\rightarrow$ Cross Attn $\rightarrow$ Add&Norm $\rightarrow$ FFN $\rightarrow$ Add&Norm.

**LLM Shoutout**: Pure decoder stacks power GPTs—proves attention's generative chops.

### 🔤 Tokenizer: `tokenizer.py`

Tokenization turns text into numbers—essentially lossless compression. I implemented two flavors:

- **`tokenizer.py`**: SentencePiece (subword) for efficiency, trained on WMT corpus. Handles OOV words via BPE.
- **`tokenizer_from_scratch.py`**: My DIY BPE from scratch. Fun educational detour, but sloooow.

### 📊 Dataset: `dataset.py`

I mirrored the paper: WMT 2014 English-German (en-de direction, ~4.5M sentence pairs, but I used only subset of 1.5M sentence pairs). 

**Data Flow**: torch.utils.data.Dataset with collate_fn for padding.


### 🚀 Training: `train.py`

Training a Transformer from scratch is where theory meets the grind—optimizing hyperparameters, wrestling with gradients, and celebrating those sweet perplexity drops. My setup mirrors the original paper's recipe (Section 5): AdamW with Noam scheduling, label smoothing, and gradient accumulation for memory efficiency. I targeted WMT 2014 de-en, aiming for BLEU 10 on a single RTX 3090. Key innovations? FP16 mixed precision for speed (2x throughput but not used in actual traning) and per-token perplexity for fairer loss tracking (ignores pads).

**Core Workflow**:
1. **Data Loading**: Subset WMT via `WMTDataset`, batched with dynamic padding in `collate_fn`.
2. **Model Init**: Load or fresh `Transformer`; resume from checkpoints if flagged.
3. **Optimization Loop**: Forward pass on shifted targets (teacher forcing), compute smoothed CE loss, backprop with clipping & scaling.
4. **Scheduling**: Noam LR—warmup to peak (~1e-3), then decay as \( d_{\text{model}}^{-0.5} \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5}) \).
5. **Logging & Eval**: TensorBoard for scalars (loss, PPL), checkpoints every 10k steps, val BLEU via `val_score` (sacrebleu under the hood).
6. **Inference Peek**: Sample translation per epoch to eyeball quality early.

### ⛔ Pitfalls & Fixes

Building a Transformer from scratch has its tough spots—like bugs that pop up late at night. Here are the three main issues I hit on my way to BLEU 10, and how I fixed them. These changes helped make training smoother and faster.

1. **💥 Out-of-Memory (OOM) Issues**  
   My 24GB GPU still ran out of memory with batches bigger than 16. The problem was a large model (about 65M parameters) using too much VRAM, especially in attention layers.  
   **🔧 The Fix**: I made the model smaller (down to ~30M params by cutting d_model from 512 to 384, d_ff from 2048 to 1536, and layers from 6 to 4). I also added dynamic batching in `collate_fn`—it sorts sequences by length, packs short ones together, and adds less padding. Now batches go up to 32 easily, saving about 40% memory.

2. **🐌 Training Not Starting Right**  
   The model loaded okay, but loss stayed flat and didn't drop much. I messed up the Noam scheduler: `LambdaLR` multiplies the base learning rate from the optimizer, it doesn't replace it. I had set the base to 5e-4, which was too low and made training super slow. After days of debugging (and lots of AI help), we found it. Then, after warmup, loss went to NaN from FP16's lower precision causing errors.  
   **🔧 The Fix**: Set base LR to 1.0 in AdamW so the scheduler scales it correctly during warmup (4000 steps). For the NaN, I used FP32 and this got stable training 

3. **📉 Hitting a Plateau**  
   Things were going well—loss dropping, BLEU near 8—but then progress slowed way down (just ~2 PPL drop per epoch). I tried scaling LR up a bit (1.2x), which helped short-term but not enough. I asked AI for ideas to break out.  
   **🔧 The Fix**: Added gradient accumulation—spread grads over 4 small batches to act like one big batch. This cut noise from small batches and sped up learning without more VRAM. After 20 more epochs, val PPL went to ~14.0 and BLEU hit 10.