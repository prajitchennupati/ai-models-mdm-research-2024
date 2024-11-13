import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Prepare Dataset
class FormatDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=64):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        input_text = f"Format: {row['raw_input']}"
        target_text = row['formatted_output']
        
        # Tokenize inputs and outputs
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer.encode_plus(
            target_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten()
        }

# Load data
def load_data(input_csv):
    df = pd.read_csv(input_csv)
    return df

# Training loop
def train_model(train_loader, model, epochs=3, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Generate formatted output
def generate_formatted_output(text, model, tokenizer, max_len=64):
    input_text = f"Format: {text}"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_len)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main function to train and apply model
def main(input_csv, output_csv):
    # Load and prepare data
    df = load_data(input_csv)
    formatted_dataset = FormatDataset(df, tokenizer)
    train_loader = DataLoader(formatted_dataset, batch_size=8)

    # Train model
    train_model(train_loader, model)

    # Apply model on new data
    df['Formatted_Output'] = df['raw_input'].apply(lambda x: generate_formatted_output(x, model, tokenizer))

    # Save output
    df.to_csv(output_csv, index=False)
    print("Formatted data saved to:", output_csv)

# Run the script with training data
main("sampleTrainingDataset.csv", "formattedOutput.csv")
