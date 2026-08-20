# Build a Multimodal Model

> *"Two modalities enter the embedding space. Only the correctly aligned pair gets to leave with dignity."*

## The Concept

A multimodal model learns from more than one type of input. In this guide, we build a compact image-text model that maps images and text into the same embedding space. Matching image-text pairs are pulled together; mismatched pairs are pushed apart. This is the core idea behind contrastive vision-language systems such as CLIP.[1]

The notebook uses CIFAR-10 images and converts each class label into short text prompts. That pairing is intentionally pedagogical. It gives us a real image dataset and a deterministic text modality without downloading a massive caption corpus. It is **not equivalent to natural image captions** and should not be interpreted as a full-scale vision-language pretraining recipe.

The objective is to make every part visible: data pairing, tokenization, image encoding, text encoding, projection into a shared space, contrastive loss, zero-shot classification, image-to-text retrieval, and embedding diagnostics.

---

## Core Philosophy

| Principle | What it means |
|-----------|---------------|
| **Separate the encoders** | Let each modality use a suitable representation pipeline |
| **Fuse in a shared space** | Compare projected image and text embeddings rather than raw features |
| **Make the pairing explicit** | Every positive pair and negative pair must be defined by the dataset |
| **Use the batch as supervision** | Other examples in the batch become contrastive negatives |
| **Evaluate both directions** | Test image-to-text and text-to-image retrieval |
| **Keep the baseline visible** | A unimodal classifier reveals whether multimodal training adds anything |
| **Inspect the data** | Bad captions, duplicate images, and weak pairings can dominate the result |
| **Treat zero-shot claims carefully** | Prompt sensitivity and label leakage can make results look more magical than they are |

---

## Architecture

The model has four major components:

| Component | Job |
|-----------|-----|
| Image encoder | Converts an image into a visual feature vector |
| Text encoder | Converts token IDs into a text feature vector |
| Projection heads | Map both modalities into the same embedding dimension |
| Temperature-scaled contrastive loss | Rewards correct pair alignment and penalizes incorrect pairings |

For a batch of `N` image-text pairs, let `v_i` be the normalized image embedding and `t_j` the normalized text embedding. The similarity matrix is:

\[
S_{ij} = \frac{v_i^\top t_j}{\tau},
\]

where \(\tau\) is a learned or fixed temperature. The diagonal contains the intended matches. We optimize both image-to-text and text-to-image cross-entropy losses:

\[
\mathcal{L} = \frac{1}{2}
\left[
\operatorname{CE}(S, I) + \operatorname{CE}(S^\top, I)
\right].
\]

This formulation follows the paired-batch contrastive pattern used in CLIP-style training.[1] Because this notebook intentionally reuses class-derived prompts, its loss treats all same-class image-text combinations as positives rather than incorrectly treating repeated class prompts as negatives. PyTorch's multimodal tooling also emphasizes composable encoders, transforms, projection/fusion layers, losses, datasets, and evaluation utilities.[2]

---

## What You Will Train

| Setting | Default value | Purpose |
|---------|---------------|---------|
| Dataset | CIFAR-10 | Real images with ten classes |
| Text modality | Class-derived prompts | Simple, reproducible text pairing |
| Image encoder | Small CNN | Notebook-friendly visual representation |
| Text encoder | Embedding plus masked mean pooling | Compact text representation |
| Shared dimension | `128` | Common image/text embedding width |
| Loss | Symmetric contrastive loss | Aligns matching pairs in both directions |
| Evaluation | Retrieval and zero-shot classification | Tests whether the embedding space is useful |
| Runtime | Colab or Kaggle | CPU works; GPU is strongly preferred |

---

## Notebook Runbook

Open a notebook in [Google Colab](https://colab.research.google.com/) or [Kaggle Notebooks](https://www.kaggle.com/code), enable a GPU, and run the cells in order. The code downloads CIFAR-10 through `torchvision`, builds prompt text from the official class labels, trains the dual encoder, evaluates retrieval, and performs zero-shot classification.

The code uses label-derived text because it is compact and self-contained. For a stronger model, replace it with genuinely paired images and captions, such as a properly licensed caption dataset, and keep train/validation identities separated.

---

## Cell 1 — Install and import dependencies

```python
# We are installing the minimum multimodal plumbing. No giant foundation model,
# no mystical checkpoint, just enough machinery to make images and text argue
# inside the same vector space.
!pip -q install -U torch torchvision matplotlib numpy
```

```python
import math
import random
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
```

## Cell 2 — Configuration and image transforms

```python
class Config:
    data_dir = "./data"
    batch_size = 256
    image_size = 32
    embed_dim = 128
    text_width = 128
    text_max_len = 12
    epochs = 8
    learning_rate = 3e-4
    weight_decay = 1e-4
    temperature = 0.07
    num_workers = 2

cfg = Config()

# CIFAR-10 normalization values are standard dataset statistics. The training
# transform adds variation; the validation transform stays deterministic.
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(cfg.image_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

## Cell 3 — Build the paired image-text dataset

```python
base_train = datasets.CIFAR10(
    root=cfg.data_dir,
    train=True,
    download=True,
    transform=train_transform,
)
base_val = datasets.CIFAR10(
    root=cfg.data_dir,
    train=False,
    download=True,
    transform=val_transform,
)

class_names = base_train.classes
print(class_names)

# These templates create multiple surface forms for the same class. This is a
# teaching convenience, not a replacement for real human-written captions.
TEMPLATES = [
    "a photo of a {}",
    "an image showing a {}",
    "a small {}",
    "a picture of the {} class",
]

all_texts = [template.format(name) for name in class_names for template in TEMPLATES]

# Tiny whitespace tokenizer. Real multimodal systems use a serious tokenizer;
# this one is deliberately transparent enough to inspect with one eyeball.
PAD = "<pad>"
UNK = "<unk>"
tokens = {PAD, UNK}
for text in all_texts:
    tokens.update(text.lower().split())

itos = sorted(tokens)
stoi = {token: index for index, token in enumerate(itos)}
PAD_ID = stoi[PAD]
UNK_ID = stoi[UNK]


def tokenize(text):
    ids = [stoi.get(token, UNK_ID) for token in text.lower().split()]
    ids = ids[: cfg.text_max_len]
    ids += [PAD_ID] * (cfg.text_max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class PairedCIFAR(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        template = TEMPLATES[index % len(TEMPLATES)]
        text = template.format(class_names[label])
        return image, tokenize(text), label, text


train_dataset = PairedCIFAR(base_train)
val_dataset = PairedCIFAR(base_val)

loader_kwargs = {
    "batch_size": cfg.batch_size,
    "num_workers": cfg.num_workers,
    "pin_memory": device.type == "cuda",
}
train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

images, text_ids, labels, text_strings = next(iter(train_loader))
print("images:", tuple(images.shape))
print("text IDs:", tuple(text_ids.shape))
print("example text:", text_strings[0])
```

## Cell 4 — Define the image and text encoders

```python
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, images):
        return self.projection(self.features(images))


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, width, embed_dim, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=pad_id)
        self.position = nn.Parameter(torch.randn(1, cfg.text_max_len, width) * 0.02)
        self.transform = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.projection = nn.Linear(width, embed_dim)

    def forward(self, token_ids):
        mask = (token_ids != self.pad_id).float().unsqueeze(-1)
        x = self.embedding(token_ids) + self.position[:, : token_ids.size(1)]
        x = self.transform(x)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)


class MultimodalModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, text_width, pad_id):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, text_width, embed_dim, pad_id)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / cfg.temperature)))

    def forward(self, images, token_ids):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(token_ids)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Clamp prevents the learned temperature from growing into a numerical
        # fireball while still allowing the model to tune contrast sharpness.
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * image_features @ text_features.T
        return logits, image_features, text_features


model = MultimodalModel(
    vocab_size=len(stoi),
    embed_dim=cfg.embed_dim,
    text_width=cfg.text_width,
    pad_id=PAD_ID,
).to(device)

print("parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
```

## Cell 5 — Define the symmetric contrastive loss

```python
def supervised_multimodal_loss(logits, labels):
    """Contrastive loss with all same-class pairs treated as positives."""
    same_class = labels[:, None].eq(labels[None, :]).float()
    positive_count = same_class.sum(dim=1).clamp_min(1.0)

    # Repeated class-derived prompts are not false negatives. This is the key
    # correction that keeps our tiny teaching dataset mathematically honest.
    image_to_text_log_probs = logits.log_softmax(dim=1)
    text_to_image_log_probs = logits.T.log_softmax(dim=1)

    image_to_text = -(
        same_class * image_to_text_log_probs
    ).sum(dim=1) / positive_count
    text_to_image = -(
        same_class.T * text_to_image_log_probs
    ).sum(dim=1) / positive_count
    return 0.5 * (image_to_text.mean() + text_to_image.mean())


with torch.no_grad():
    test_logits, _, _ = model(images.to(device), text_ids.to(device))
print("logits shape:", tuple(test_logits.shape))
print(
    "initial contrastive loss:",
    float(supervised_multimodal_loss(test_logits, labels.to(device))),
)
```

## Cell 6 — Train the two encoders together

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
)

history = []
for epoch in range(1, cfg.epochs + 1):
    model.train()
    running_loss = 0.0
    total = 0
    start_time = time.time()

    for batch_images, batch_text, batch_labels, _ in train_loader:
        batch_images = batch_images.to(device, non_blocking=True)
        batch_text = batch_text.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(batch_images, batch_text)
        loss = supervised_multimodal_loss(logits, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * batch_images.size(0)
        total += batch_images.size(0)

    epoch_loss = running_loss / total
    elapsed = time.time() - start_time
    history.append({"epoch": epoch, "loss": epoch_loss})
    print(
        f"epoch {epoch:02d}/{cfg.epochs} | loss {epoch_loss:.4f} | "
        f"images/s {total / max(elapsed, 1e-6):.1f} | "
        f"temperature {model.logit_scale.exp().item():.3f}"
    )
```

## Cell 7 — Evaluate image-text retrieval

```python
@torch.no_grad()
def collect_embeddings(model, loader):
    model.eval()
    image_features = []
    text_features = []
    labels = []

    for batch_images, batch_text, batch_labels, _ in loader:
        batch_images = batch_images.to(device)
        batch_text = batch_text.to(device)
        _, image_emb, text_emb = model(batch_images, batch_text)
        image_features.append(image_emb.cpu())
        text_features.append(text_emb.cpu())
        labels.append(batch_labels)

    return (
        torch.cat(image_features),
        torch.cat(text_features),
        torch.cat(labels),
    )


image_emb, text_emb, val_labels = collect_embeddings(model, val_loader)
similarity = image_emb @ text_emb.T

# Recall@K is class-aware here: any same-class candidate is a valid positive.
def class_recall_at_k(similarity_matrix, query_labels, candidate_labels, k):
    topk = similarity_matrix.topk(k, dim=1).indices
    retrieved_labels = candidate_labels[topk]
    matches = retrieved_labels.eq(query_labels[:, None])
    return matches.any(dim=1).float().mean().item()

subset = slice(0, 1000)
query_labels = val_labels[subset]
candidate_labels = val_labels[subset]
subset_similarity = similarity[subset, subset]
print(
    "image-to-text class Recall@1:",
    class_recall_at_k(subset_similarity, query_labels, candidate_labels, 1),
)
print(
    "image-to-text class Recall@5:",
    class_recall_at_k(subset_similarity, query_labels, candidate_labels, 5),
)
print(
    "text-to-image class Recall@1:",
    class_recall_at_k(subset_similarity.T, candidate_labels, query_labels, 1),
)
print(
    "text-to-image class Recall@5:",
    class_recall_at_k(subset_similarity.T, candidate_labels, query_labels, 5),
)
```

## Cell 8 — Perform zero-shot classification with text prompts

```python
@torch.no_grad()
def zero_shot_classify(model, images, class_names):
    model.eval()
    prompts = [tokenize(f"a photo of a {name}") for name in class_names]
    prompts = torch.stack(prompts).to(device)

    image_features = F.normalize(model.image_encoder(images.to(device)), dim=-1)
    text_features = F.normalize(model.text_encoder(prompts), dim=-1)
    scores = image_features @ text_features.T
    return scores.argmax(dim=-1).cpu(), scores.cpu()


batch_images, _, batch_labels, _ = next(iter(val_loader))
predictions, scores = zero_shot_classify(model, batch_images, class_names)
print("zero-shot batch accuracy:", (predictions == batch_labels).float().mean().item())

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for axis, image, target, prediction in zip(
    axes.flat,
    batch_images[:10],
    batch_labels[:10],
    predictions[:10],
):
    image = image * torch.tensor(std).view(3, 1, 1)
    image = image + torch.tensor(mean).view(3, 1, 1)
    axis.imshow(image.clamp(0, 1).permute(1, 2, 0))
    color = "green" if int(target) == int(prediction) else "red"
    axis.set_title(
        f"true: {class_names[int(target)]}\npred: {class_names[int(prediction)]}",
        color=color,
    )
    axis.axis("off")
plt.tight_layout()
plt.show()
```

## Cell 9 — Save the model and tokenizer

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "stoi": stoi,
    "itos": itos,
    "config": {
        key: getattr(cfg, key)
        for key in ["embed_dim", "text_width", "text_max_len"]
    },
}
torch.save(checkpoint, "multimodal_model.pt")
print("saved multimodal_model.pt")
```

---

## How to Upgrade This Model

| Upgrade | Why it matters |
|---------|----------------|
| **Natural image captions** | Replaces label-derived text with richer semantic supervision |
| **Subword tokenizer** | Represents real captions more efficiently |
| **Transformer text encoder** | Models word order and longer descriptions better |
| **Vision Transformer encoder** | Provides a stronger visual backbone |
| **Hard-negative mining** | Selects confusing mismatched pairs instead of relying only on random batch negatives |
| **Cross-attention fusion** | Lets image and text features interact token by token |
| **Caption decoder** | Enables image-to-text generation rather than retrieval only |
| **Fine-grained evaluation** | Tests OCR, counting, attributes, and compositional generalization |
| **Distributed training** | Makes larger paired datasets and encoders practical |
| **Data filtering and deduplication** | Reduces noisy, duplicated, or unsafe image-text pairs |

The order matters. Improve the data pairing before increasing the model size. Add hard negatives before declaring that the embedding geometry is broken. Add cross-attention when the task requires token-level interaction, not merely because the layer exists.

---

## Common Failure Modes

### The model learns the labels but not the modality relationship

Label-derived prompts can make the text encoder memorize class words without learning rich semantics. Use genuine captions and evaluate retrieval on unseen descriptions if the goal is natural language grounding.

### Retrieval metrics look too good

Duplicate images, repeated captions, or class-derived text can make the task artificially easy. Deduplicate data and create splits by source identity when possible.

### The temperature explodes

Clamp the learned logit scale, lower the learning rate, and inspect embedding norms. Temperature controls the sharpness of the contrastive distribution; it is not a free permission slip for logits to become astronomical.

### One modality collapses

Check the norm and variance of both embedding streams. A modality may be producing nearly identical vectors for every example. Compare a frozen-encoder experiment with a trainable-encoder experiment and inspect gradients.

### Zero-shot classification fails while training loss falls

The prompt wording may be mismatched with training templates, the text vocabulary may be too small, or the model may be memorizing pair positions. Evaluate with multiple prompt templates and compare retrieval independently.

---

## What This Is and What It Is Not

### This Is

This is a complete dual-encoder multimodal training example with real images, an explicit text modality, paired-batch contrastive learning, retrieval evaluation, zero-shot classification, visualization, and checkpointing.

### This Is Not

This is not a production vision-language model, a captioning system, a safety-filtered web-scale dataset, or evidence that class-derived prompts capture the meaning of natural language. It is a compact laboratory for understanding the mechanics.

> **The honest headline:** multimodal learning is not “put image and text into one model.” It is designing a trustworthy relationship between modalities and proving that the relationship generalizes.

---

## References

[1]: https://arxiv.org/abs/2103.00020 "Learning Transferable Visual Models From Natural Language Supervision"

[2]: https://pytorch.org/blog/introducing-torchmultimodal/ "Introducing TorchMultimodal — PyTorch"

[3]: https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html "Torchvision CIFAR-10 dataset documentation"

---

**Project status:** Notebook-ready educational guide  
**Recommended workflow:** Start with label-derived prompts to validate the pipeline, then replace them with a properly licensed, naturally paired image-text dataset.
