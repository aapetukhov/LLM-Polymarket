from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

import torch.nn as nn

class ClassifierModel:
    def __init__(self, model_name="bert-base-uncased", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression"
        ).to(device)

        self.dropout = nn.Dropout(0.1)
        
    def preprocess_input(self, question, context):
        combined_text = f"Question: {question} Context: {context}"
        return self.tokenizer(
            combined_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def predict(self, question, context):
        inputs = self.preprocess_input(question, context)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probability = torch.sigmoid(logits).item()
            
        return {
            "yes_probability": probability,
            "no_probability": 1 - probability
        }
    
    def train(self, train_questions, train_contexts, train_labels, epochs=3, batch_size=8):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(train_questions), batch_size):
                batch_questions = train_questions[i:i + batch_size]
                batch_contexts = train_contexts[i:i + batch_size]
                batch_labels = torch.tensor(train_labels[i:i + batch_size], 
                                         dtype=torch.float).to(self.device)
                
                optimizer.zero_grad()
                
                inputs = self.preprocess_input(batch_questions, batch_contexts)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                loss = criterion(outputs.logits.squeeze(), batch_labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(train_questions) // batch_size)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")