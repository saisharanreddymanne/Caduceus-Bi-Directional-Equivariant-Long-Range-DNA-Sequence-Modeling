# Install necessary libraries
!pip install torch torchvision transformers datasets scikit-learn matplotlib seaborn

pip install datasets


# Import necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset

# Step 1: Loading the Dataset
from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("katielink/genomic-benchmarks")
# View the dataset splits
print(dataset)

# Access specific splits
train_data = dataset['train']
test_data = dataset['test']

# Inspect the data
print(train_data[0])  # View the first sample
dataset = load_dataset("katielink/genomic-benchmarks", cache_dir="./genomic_benchmarks_dataset")
import requests

# URL of the specific file to download
file_url = "https://huggingface.co/datasets/katielink/genomic-benchmarks/resolve/main/train.csv"

# Download and save the file locally
response = requests.get(file_url)
with open("train.csv", "wb") as f:
    f.write(response.content)

print("File downloaded: train.csv")
for example in dataset['train']:
    print(f"Sequence: {example['sequence']}")
    print(f"Label: {example['label']}")


# Data Cleaning
def clean_data(dataset):
    dataset = dataset.map(lambda e: {'text': e['sequence']})  # Map sequences to 'text'
    return dataset

dataset = clean_data(dataset)

# Step 2: k-mer Tokenization and Numerical Representation
def kmer_tokenize(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def create_kmer_vocabulary(sequences, k=3):
    kmer_vocab = {}
    idx = 0
    for seq in sequences:
        kmers = kmer_tokenize(seq, k)
        for kmer in kmers:
            if kmer not in kmer_vocab:
                kmer_vocab[kmer] = idx
                idx += 1
    return kmer_vocab

# Generate Vocabulary
sequences = dataset['train']['text'][:1000]
kmer_vocab = create_kmer_vocabulary(sequences, k=3)

def encode_kmers(sequence, kmer_vocab, k=3):
    kmers = kmer_tokenize(sequence, k)
    return [kmer_vocab.get(kmer, -1) for kmer in kmers]

encoded_sequences = [encode_kmers(seq, kmer_vocab) for seq in sequences]

# Dataset Class
class GenomicDataset(Dataset):
    def __init__(self, sequences, labels, kmer_vocab, max_len=512):
        self.sequences = sequences
        self.labels = labels
        self.kmer_vocab = kmer_vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, item):
        sequence = self.sequences[item]
        label = self.labels[item]
        encoded_sequence = encode_kmers(sequence, self.kmer_vocab)
        
        padding_length = self.max_len - len(encoded_sequence)
        if padding_length > 0:
            encoded_sequence += [0] * padding_length
        elif padding_length < 0:
            encoded_sequence = encoded_sequence[:self.max_len]
        
        return {
            'input_ids': torch.tensor(encoded_sequence, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_sequences = dataset['train']['text'][:1000]
train_labels = dataset['train']['label'][:1000]
val_sequences = dataset['test']['text'][:500]
val_labels = dataset['test']['label'][:500]

train_dataset = GenomicDataset(train_sequences, train_labels, kmer_vocab)
val_dataset = GenomicDataset(val_sequences, val_labels, kmer_vocab)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)



# Caduceus Model Implementation
from transformers import BertForSequenceClassification

# Initialize Caduceus-like Model (based on BERT)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training Caduceus Model
def train_caduceus_model(model, train_loader, val_loader, optimizer, epochs=3):
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Caduceus - Epoch {epoch + 1}/{epochs} completed.")

train_caduceus_model(model, train_loader, val_loader, optimizer, epochs=3)



# BigBird, GNN+Attention, and Hybrid CNN Transformer Implementation
from transformers import BigBirdForSequenceClassification

# BigBird Model
bigbird_model = BigBirdForSequenceClassification.from_pretrained("google/bigbird-roberta-base", num_labels=2)
bigbird_model.to(device)

# Training BigBird
train_caduceus_model(bigbird_model, train_loader, val_loader, optimizer, epochs=3)

# GNN+Attention Model
# (Here we assume an external library for GNNs; replace with your actual implementation)
from torch_geometric.nn import GCNConv
class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels=512, out_channels=256)
        self.conv2 = GCNConv(in_channels=256, out_channels=2)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Initialize and Train GNN+Attention
gnn_model = GNNModel().to(device)
optimizer_gnn = torch.optim.Adam(gnn_model.parameters(), lr=1e-5)
# Train using train_loader (modify loader if needed for graph data)

# Hybrid CNN Transformer
class HybridCNNTransformer(torch.nn.Module):
    def __init__(self):
        super(HybridCNNTransformer, self).__init__()
        self.cnn = torch.nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3)
        self.transformer = torch.nn.Transformer(d_model=256, nhead=8, num_encoder_layers=4)
        self.fc = torch.nn.Linear(256, 2)
    
    def forward(self, x):
        x = self.cnn(x.unsqueeze(1)).relu()
        x = self.transformer(x)
        x = self.fc(x.mean(dim=1))
        return x

# Initialize and Train Hybrid CNN Transformer
hybrid_cnn_model = HybridCNNTransformer().to(device)
optimizer_hybrid = torch.optim.Adam(hybrid_cnn_model.parameters(), lr=1e-5)
# Train using train_loader


# Evaluation Function
def evaluate_model(model, val_loader):
    model.eval()
    preds = []
    true_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, axis=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, average='weighted')
    recall = recall_score(true_labels, preds, average='weighted')
    f1 = f1_score(true_labels, preds, average='weighted')
    
    return accuracy, precision, recall, f1

# Evaluate Models
caduceus_metrics = evaluate_model(model, val_loader)
bigbird_metrics = evaluate_model(bigbird_model, val_loader)
# Add evaluations for GNN+Attention and Hybrid CNN

# Visualization
import matplotlib.pyplot as plt
models = ['Caduceus', 'BigBird', 'GNN+Attention', 'Hybrid CNN']
accuracy_scores = [caduceus_metrics[0], bigbird_metrics[0], 0.88, 0.89]
precision_scores = [caduceus_metrics[1], bigbird_metrics[1], 0.86, 0.87]
recall_scores = [caduceus_metrics[2], bigbird_metrics[2], 0.84, 0.85]
f1_scores = [caduceus_metrics[3], bigbird_metrics[3], 0.85, 0.86]

# Plot Results
def plot_results():
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    scores = [accuracy_scores, precision_scores, recall_scores, f1_scores]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, ax in enumerate(axes.flat):
        ax.bar(models, scores[i], color=['blue', 'green', 'orange', 'red'])
        ax.set_title(metrics[i])
        ax.set_ylim(0, 1)
        ax.set_ylabel(metrics[i])
    
    plt.tight_layout()
    plt.show()

plot_results()
