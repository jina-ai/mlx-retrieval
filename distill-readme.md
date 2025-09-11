# Embedding Distillation from Qwen3-0.6b to Gemma3-270m

> [!IMPORTANT]
> **Heads up: gemma-3-270m doesn't have a friendly license, so I only did this for educational purposes. Don't use it in production!** Also, gemma-3-270m is ***not*** the same as Google's EmbeddingGemma from last week, though both models share similar backbones. The hidden size and output dimensions are different. I did this before EmbeddingGemma was released, which is why you'll see I tried to distill from Qwen3-0.6b to Gemma3-270m.

* By distillation, I mean **directly copying the first 640 dimensions from qwen3-0.6b** (since gemma3-270m outputs 640d). Since qwen3-0.6b uses MRL, learning just the first 640d actually makes sense.
* I didn't touch the MLP because I wanted to keep gemma-3-270m's architecture unchanged.
* As a baseline, qwen3-0.6b's first 640d gets 0.60 ndcg@5 on NanoMSMacro (one of the hardest datasets in NanoBEIR). For reference, our v4 scores around 0.55 and v3 around 0.64 (yep, v4 < v3 on NanoMSMarco, shame on us), which again shows that copying the first 640d isn't nonsense.
* So here's the goal: a perfect distillation model should hit 0.60 ndcg@5 on NanoMSMacro. If not, skill issue.
* **Best performance I achieved: 0.54 ndcg@5 after training on 100M tokens, so comparable to the v4 model (I think?)**

## Variants I tested:
* **Single special token as q/d prefix**: `<bos><unknown_0>...<eos>`, `<bos><unknown_1>...<eos>`, where `<unknown_0/1>` represents v3/v4's multi-token "Query: ", "Document: " prefixes. Tried different orderings with no significant performance difference:
   * `<bos><unknown_0>...<eos>`: leading position, v3/v4 style
   * `<bos><unknown_0>...<unknown_0><eos>`: wrapping, to reinforce the final eos
   * `<bos>...<unknown_0><eos>`: trailing position, to share more info between q & d, only differentiating q/d roles at the end
   * Also tried full natural language prompts from qwen3-0.6b: `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`, `Document:` — no significant performance difference.
* **eos > mean pooling**: eos performs better and gives a cleaner training recipe
* **post-training > LoRA**: post-training performs better with fewer hyperparameters and a cleaner recipe
* **causal+eos > bidirectional+eos** performance-wise. Plus, causal is 10% faster to train. Didn't test bidirectional+mean.
   * For causal vs. non-causal: watch out for gemma3's local/global attention mechanism. Local means a sliding window of 512 (lookahead up to 512 tokens on the right). Global means looking at everything to the right. Every 8 layers you get global attention. When removing the causal mask to go bidirectional, be careful with those global attention layers at layers 8 & 16. Also, global and local attention use different RoPE frequencies.
* **cosine loss vs. cosine+L1 loss**: no performance difference
* **gemma-3-270m-it slightly outperforms gemma-3-270m**: The `-it` version is instruction-tuned, and post-training on it gave slightly better results than post-training on the base model.

## Training data:
* Some calibration data I'd been using for GGUF imatrix building—just reused it because I didn't want to collect more. No time. Total: 100K lines.
* Generated another 50K lines using [python-faker](https://faker.readthedocs.io/en/master/).
* All data is multilingual, ranging from 1 to 1000 tokens. Fed them through qwen3-0.6b to get embeddings, then took `[:640]`
* Performance peaked at 100k steps; beyond that, it started overfitting. So 100M tokens trained in **2 hours on an M3 Ultra chip at 10K tok/s.**
* **Perfect overfitting is achievable: cosine+10xL1 loss can drop below 1e-4!** If your distillation can't even overfit, you've got a bug.

## Other interesting things to test:
* **More fine-grained copying**: Instead of only measuring the difference on the last output, use the last k tokens' outputs. Layer-by-layer copying seems unlikely since qwen3-0.6b and gemma3-270m have different layer counts, but might still be doable.
* **Changing RoPE frequency**: Looking at our actual input data token lengths, 512 tokens isn't a small window for local attention. Consider that in our NanoBEIR eval set, the mean token length for both queries and documents is under 512 on nearly every dataset. So local and global attention effectively see the same right-hand info to lookahead to — the only difference is their RoPE frequency.