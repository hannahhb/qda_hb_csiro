import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd

# Load the dataset
file_path = 'data/1.0 SLR CFIR Thematic Analysis - FINAL.xlsx'
data = pd.read_excel(file_path, sheet_name='Coded_data')

# Keep only relevant columns
data = data[['Comments', 'Domain']].dropna()

# Encode the target labels
domain_labels = data['Domain'].unique()
domain_to_id = {label: idx for idx, label in enumerate(domain_labels)}
data['Domain_id'] = data['Domain'].map(domain_to_id)

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Create a custom dataset class
class CFIRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Prepare the dataset
dataset = CFIRDataset(
    texts=data['Comments'].to_list(),
    labels=data['Domain_id'].to_list(),
    tokenizer=tokenizer
)

# Create data loader
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define the Transformer model
class DomainClassifier(nn.Module):
    def __init__(self, n_classes):
        super(DomainClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0]
        output = self.dropout(pooled_output)
        return self.out(output)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = DomainClassifier(n_classes=len(domain_labels))
model = model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss().to(device)

# Training loop
def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# Training the model for 1 epoch (for demonstration purposes)
train_epoch(model, data_loader, loss_fn, optimizer, device)
