import torch
from torch.utils.data import DataLoader
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def get_data_tensor(dataset, vocab, tokenizer):
    """Tokenizes and numericalizes the dataset into a single tensor."""
    data = []
    for item in dataset:
        if item.strip() == '':
            continue
        data.extend(vocab(tokenizer(item)))
    return torch.tensor(data, dtype=torch.long)

def batchify(data: torch.Tensor, bsz: int, device: str):
    """
    Arranges the data into bsz separate sequences, removing extra elements.
    Args:
        data: Tensor, shape [N]
        bsz: int, batch size
    Returns:
        Tensor of shape [seq_len, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

def get_processed_data(batch_size: int, device: str, vocab_size: int = 10000):
    """
    Downloads, processes, and batches the PennTreebank dataset.
    """
    print("Fetching Penn Treebank dataset...")
    try:
        train_iter, val_iter, test_iter = PennTreebank()
    except Exception as e:
        print(f"Failed to download Penn Treebank: {e}")
        # Fallback for environments without internet.
        dummy_text = ["hello world this is a test"] * 1000
        train_iter, val_iter, test_iter = (dummy_text, dummy_text[:100], dummy_text[:100])

    tokenizer = get_tokenizer('basic_english')

    print("Building vocabulary...")
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'], max_tokens=vocab_size)
    vocab.set_default_index(vocab['<unk>'])

    train_data = get_data_tensor(train_iter, vocab, tokenizer)
    val_data = get_data_tensor(val_iter, vocab, tokenizer)
    test_data = get_data_tensor(test_iter, vocab, tokenizer)

    train_data = batchify(train_data, batch_size, device)
    val_data = batchify(val_data, batch_size, device)
    test_data = batchify(test_data, batch_size, device)

    return train_data, val_data, test_data, vocab

def get_batch(source: torch.Tensor, i: int, bptt: int):
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int, index
    Returns:
        tuple (data, target)
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

if __name__ == '__main__':
    device = "cpu"
    bptt = 35
    train_data, _, _, vocab = get_processed_data(batch_size=20, device=device, vocab_size=1000)
    print(f"Vocab size: {len(vocab)}")
    print(f"Train data shape: {train_data.shape}")

    data, targets = get_batch(train_data, 0, bptt)
    print(f"First batch data shape: {data.shape}")
    print(f"First batch target shape: {targets.shape}")
