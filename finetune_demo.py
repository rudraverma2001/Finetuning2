# finetune_demo.py
import json
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import Dataset

MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "./finetuned_model"
DATA_FILE = "training_data.json"


class QADataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=128):
        self.examples = []
        for q, a in pairs:
            text = f"Question: {q}\nAnswer: {a}{tokenizer.eos_token}"
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            self.examples.append(enc)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
        }


def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []


def save_data(pairs):
    with open(DATA_FILE, "w") as f:
        json.dump(pairs, f, indent=2)


def collect_training_data():
    pairs = load_data()
    print(f"Loaded {len(pairs)} existing Q/A pairs.")
    print("Enter training Q/A pairs. Type 'done' as question to stop.\n")

    while True:
        q = input("Question: ").strip()
        if q.lower() == "done":
            break
        a = input("Answer: ").strip()
        pairs.append([q, a])

    save_data(pairs)
    print(f"Saved {len(pairs)} pairs to {DATA_FILE}")
    return pairs


def fine_tune(pairs, epochs=3, batch_size=2, model_path=None):
    if not pairs:
        print("No training data available. Aborting fine-tune.")
        return None, None

    source = model_path if model_path and os.path.exists(model_path) else MODEL_NAME
    print(f"Loading base model: {source}")

    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(source)

    dataset = QADataset(pairs, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        logging_steps=5,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting fine-tuning...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

    return model, tokenizer


def load_finetuned():
    if not os.path.exists(OUTPUT_DIR):
        print("No fine-tuned model found. Please fine-tune first.")
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
    return model, tokenizer


def test_model(model, tokenizer, max_length=80):
    if model is None or tokenizer is None:
        print("No model loaded.")
        return

    print("\nEnter prompts to test the model. Type 'exit' to return to menu.")
    model.eval()

    while True:
        prompt = input("\nYour question: ").strip()
        if prompt.lower() == "exit":
            break

        input_text = f"Question: {prompt}\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the answer portion
        if "Answer:" in generated:
            answer = generated.split("Answer:", 1)[1].strip()
        else:
            answer = generated

        print(f"Model: {answer}")


def main():
    model, tokenizer = None, None

    menu = """
=== Fine-Tuning Demo App ===
1. Add/Edit training data (Q/A pairs)
2. Fine-tune model on collected data
3. Load existing fine-tuned model
4. Test model with prompts
5. Exit
"""

    while True:
        print(menu)
        choice = input("Choose an option: ").strip()

        if choice == "1":
            collect_training_data()

        elif choice == "2":
            pairs = load_data()
            try:
                epochs = int(input("Number of epochs (default 3): ") or 3)
            except ValueError:
                epochs = 3
            try:
                batch_size = int(input("Batch size (default 2): ") or 2)
            except ValueError:
                batch_size = 2
            model, tokenizer = fine_tune(pairs, epochs=epochs, batch_size=batch_size)

        elif choice == "3":
            model, tokenizer = load_finetuned()

        elif choice == "4":
            test_model(model, tokenizer)

        elif choice == "5":
            print("Exiting.")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
