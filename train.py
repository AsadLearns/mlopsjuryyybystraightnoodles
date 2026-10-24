import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------
# 1. Dataset: Real-world Sentiment Sentences
# -------------------------------------------------------------
RAW_DATA = [
    # Positive reviews (Label: 1)
    ("this movie was fantastic and brilliant", 1),
    ("an absolute masterpiece of cinema", 1),
    ("i loved every single minute of this film", 1),
    ("superb acting and breathtaking visual effects", 1),
    ("a delightful experience full of warmth and joy", 1),
    ("one of the best films i have ever seen", 1),
    ("wonderfully written and expertly directed", 1),
    ("truly remarkable storytelling and deep emotion", 1),
    ("highly recommended for anyone who loves great cinema", 1),
    ("the soundtrack and performances were magnificent", 1),
    ("an uplifting and thoroughly entertaining adventure", 1),
    ("captivating from start to finish with great depth", 1),
    ("a genuinely heartwarming and memorable story", 1),
    ("stellar performances from the entire cast", 1),
    ("refreshing innovative and exceptionally well crafted", 1),
    ("exceeded all my expectations in every way", 1),
    ("beautifully shot with a touching message", 1),
    ("a thrilling ride that keeps you engaged throughout", 1),
    ("pure genius and brilliantly executed", 1),
    ("charming witty and utterly enjoyable", 1),
    ("impressive direction and powerful emotional delivery", 1),
    ("a triumphant achievement in modern filmmaking", 1),
    ("smart hilarious and emotionally satisfying", 1),
    ("brilliant cinematography with standout performances", 1),
    ("the story was inspiring and beautifully told", 1),

    # Negative reviews (Label: 0)
    ("this movie was completely terrible and boring", 0),
    ("an utter waste of time and money", 0),
    ("i hated every single minute of this garbage", 0),
    ("horrible acting and dreadful special effects", 0),
    ("a painful experience full of cliches and bad jokes", 0),
    ("one of the worst films i have ever suffered through", 0),
    ("poorly written and lazily directed disaster", 0),
    ("truly unwatchable plot with zero emotional depth", 0),
    ("do not watch this movie under any circumstances", 0),
    ("the soundtrack was annoying and acting was flat", 0),
    ("a depressing and thoroughly frustrating experience", 0),
    ("boring from start to finish with lifeless characters", 0),
    ("a cheap sloppy and completely forgettable mess", 0),
    ("abysmal performances across the board", 0),
    ("unoriginal uninspired and excruciatingly dull", 0),
    ("failed to meet even the lowest expectations", 0),
    ("badly shot with a confusing and pointless narrative", 0),
    ("a slow painful drag that never goes anywhere", 0),
    ("utter nonsense without any redeeming qualities", 0),
    ("offensive loud and completely unfunny", 0),
    ("dismal direction and hollow performances", 0),
    ("a catastrophic failure that should never have been made", 0),
    ("predictable tedious and utterly disappointing", 0),
    ("laughably bad dialogue with terrible pacing", 0),
    ("the story was shallow and entirely nonsensical", 0),
]

MAX_SEQ_LENGTH = 16
EMBED_DIM = 64
HIDDEN_DIM = 32
NUM_CLASSES = 2
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.01

# -------------------------------------------------------------
# 2. Tokenizer & Vocabulary Builder
# -------------------------------------------------------------
def build_vocab(texts):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in texts:
        tokens = text.lower().split()
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def text_to_tensor(text, vocab, max_len=MAX_SEQ_LENGTH):
    tokens = text.lower().split()
    indices = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    if len(indices) < max_len:
        indices += [vocab["<PAD>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return torch.tensor(indices, dtype=torch.long)

class SentimentDataset(Dataset):
    def __init__(self, data, vocab):
        self.samples = []
        for text, label in data:
            tensor = text_to_tensor(text, vocab)
            self.samples.append((tensor, torch.tensor(label, dtype=torch.long)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# -------------------------------------------------------------
# 3. Model Architecture (FastText-style Average Pooling)
# -------------------------------------------------------------
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)               # (batch_size, seq_len, embed_dim)
        pooled = torch.mean(embedded, dim=1)       # (batch_size, embed_dim)
        hidden = self.relu(self.fc1(pooled))       # (batch_size, hidden_dim)
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)                  # (batch_size, num_classes)
        return logits

# -------------------------------------------------------------
# 4. Training & Evaluation Pipeline
# -------------------------------------------------------------
def main():
    print("=" * 60)
    print("       STAGE B: MODEL ARCHITECTURE & TRAINING PIPELINE       ")
    print("=" * 60)

    # 80% Train, 20% Test Split
    train_data = RAW_DATA[:20] + RAW_DATA[25:45]   # 40 samples (20 Pos, 20 Neg)
    test_data = RAW_DATA[20:25] + RAW_DATA[45:50]   # 10 samples (5 Pos, 5 Neg)

    all_texts = [text for text, _ in RAW_DATA]
    vocab = build_vocab(all_texts)
    print(f"[*] Vocabulary constructed: {len(vocab)} unique tokens (including <PAD> and <UNK>)")

    # Save vocabulary for inference and edge deployment
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    print("[*] Saved vocabulary to: artifacts/vocab.json")

    train_dataset = SentimentDataset(train_data, vocab)
    test_dataset = SentimentDataset(test_data, vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_data), shuffle=False)

    model = SentimentClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n[*] Starting Neural Network Training...")
    print(f"    - Architecture: Embedding({len(vocab)}, {EMBED_DIM}) -> MeanPool -> Linear({EMBED_DIM}, {HIDDEN_DIM}) -> Linear({HIDDEN_DIM}, {NUM_CLASSES})")
    print(f"    - Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"    - Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        epoch_loss = total_loss / total
        epoch_acc = (correct / total) * 100.0

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch [{epoch:02d}/{EPOCHS:02d}] - Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.1f}%")

    # Evaluation on held-out test data
    model.eval()
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_outputs = model(test_x)
            test_loss = criterion(test_outputs, test_y).item()
            test_preds = torch.argmax(test_outputs, dim=1)
            test_acc = ((test_preds == test_y).sum().item() / test_y.size(0)) * 100.0

    print("\n" + "-" * 60)
    print(f"[*] Evaluation on Held-Out Test Set:")
    print(f"    - Test Loss:     {test_loss:.4f}")
    print(f"    - Test Accuracy: {test_acc:.1f}% ({int(test_acc/10)}/10 correctly classified)")
    print("-" * 60)

    # Save PyTorch model state dictionary
    torch.save(model.state_dict(), "artifacts/model_weights.pt")
    print("[*] Saved PyTorch weights to: artifacts/model_weights.pt")
    print("\n[SUCCESS] Stage B Complete! Ready for Model Serialization.")

if __name__ == "__main__":
    main()
