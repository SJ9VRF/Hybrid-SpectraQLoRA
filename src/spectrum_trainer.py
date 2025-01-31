import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AdamW

class SpectrumTrainer:
    def __init__(self, model, target_layers, train_dataloader, eval_dataloader, output_dir):
        self.model = model
        self.target_layers = target_layers
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Apply LoRA adapters to high-SNR layers
        self.lora_config = LoraConfig(
            target_modules=self.target_layers,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05
        )
        self.model = get_peft_model(self.model, self.lora_config)

    def train(self, epochs=3, lr=2e-5):
        optimizer = AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            for batch in self.train_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

        torch.save(self.model.state_dict(), f"{self.output_dir}/qlora_spectrum_finetuned.pth")

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy: {100 * correct / total:.2f}%")

