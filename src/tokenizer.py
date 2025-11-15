import os
from config import TOKENIZER_PATH, TOKENIZER_OPTIONS, DATASET_TRAIN_FILE
import sentencepiece as sp


def load_tokenizer(
    tokenizer_path: str,
    train_options: dict | None = None,
    train_dataset_path: str | None = None,
) -> sp.SentencePieceProcessor:
    """
    Loads or trains and loads a SentencePiece tokenizer.

    Parameters
    ----------
    tokenizer_path : str
        Path to .model file (e.g., "model/poet_tokenizer.model").
    train_options : dict | None, default=None
        SentencePiece training options (e.g., {'vocab_size': 37000}). Triggers training if model missing.
    train_dataset_path : str | None, default=None
        Path to text file/DataFrame for training if options provided.

    Returns
    -------
    tokenizer : sp.SentencePieceProcessor
        Loaded/trained tokenizer instance.
    """
    if not os.path.exists(tokenizer_path) and train_options is not None:
        train_options["input"] = train_options.get("input", train_dataset_path)
        sp.SentencePieceTrainer.Train(**train_options)
    tokenizer = sp.SentencePieceProcessor()
    tokenizer.LoadFromFile(tokenizer_path)
    return tokenizer


def main():
    text = """Transformers acts as the model-definition framework for state-of-the-art machine learning models in text, computer vision, audio, video, and multimodal model, for both inference and training.
It centralizes the model definition so that this definition is agreed upon across the ecosystem. transformers is the pivot across frameworks: if a model definition is supported, it will be compatible with the majority of training frameworks (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, …), inference engines (vLLM, SGLang, TGI, …), and adjacent modeling libraries (llama.cpp, mlx, …) which leverage the model definition from transformers.
We pledge to help support new state-of-the-art models and democratize their usage by having their model definition be simple, customizable, and efficient.
There are over 1M+ Transformers model checkpoints on the Hugging Face Hub you can use.
Explore the Hub today to find a model and use Transformers to help you get started right away.
Explore the Models Timeline to discover the latest text, vision, audio and multimodal model architectures in Transformers."""
    tokenizer = load_tokenizer(TOKENIZER_PATH, TOKENIZER_OPTIONS, DATASET_TRAIN_FILE)
    print("Tokenizer Trained!")
    tokenized_text = tokenizer.encode(text)
    print("Length of original text:", len(text), end="\n\n---\n\n")
    print("Length of tokenized text:", len(tokenized_text), end="\n\n---\n\n")
    print(f"Compress Ratio: {len(text) / len(tokenized_text):.2f}X", end="\n\n---\n\n")


if __name__ == "__main__":
    main()
