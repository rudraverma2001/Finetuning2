import streamlit as st
import pandas as pd
import tempfile
import os

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

MODEL_NAME = "distilgpt2"

st.set_page_config(page_title="Fine-Tuning Demo", layout="wide")

st.title("LLM Fine-Tuning Demo")

# Session state
if "training_data" not in st.session_state:
    st.session_state.training_data = []

if "model_path" not in st.session_state:
    st.session_state.model_path = None


# --------------------------------------
# Add Training Data
# --------------------------------------
st.header("Add Training Examples")

question = st.text_input("Question")
answer = st.text_area("Answer")

if st.button("Add Q&A Pair"):
    if question and answer:
        st.session_state.training_data.append(
            {
                "question": question,
                "answer": answer
            }
        )
        st.success("Added successfully!")

if st.session_state.training_data:
    df = pd.DataFrame(st.session_state.training_data)
    st.dataframe(df)


# --------------------------------------
# Fine-Tuning Function
# --------------------------------------
def train_model(data):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    texts = []

    for row in data:
        sample = (
            f"Question: {row['question']}\n"
            f"Answer: {row['answer']}"
        )
        texts.append(sample)

    dataset = Dataset.from_dict({"text": texts})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True
    )

    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    output_dir = tempfile.mkdtemp()

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_strategy="no",
        logging_steps=5,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    trainer.train()

    save_dir = os.path.join(output_dir, "fine_tuned_model")

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    return save_dir


# --------------------------------------
# Train Button
# --------------------------------------
st.header("Fine-Tune")

if st.button("Start Fine-Tuning"):

    if len(st.session_state.training_data) == 0:
        st.warning("Please add training examples.")
    else:

        with st.spinner("Training model..."):

            model_dir = train_model(
                st.session_state.training_data
            )

            st.session_state.model_path = model_dir

        st.success("Fine-tuning completed!")


# --------------------------------------
# Inference
# --------------------------------------
st.header("Test Fine-Tuned Model")

prompt = st.text_input("Enter Prompt")

if st.button("Generate Response"):

    if st.session_state.model_path is None:
        st.warning("Please fine-tune the model first.")

    elif not prompt:
        st.warning("Enter a prompt.")

    else:

        tokenizer = AutoTokenizer.from_pretrained(
            st.session_state.model_path
        )

        model = AutoModelForCausalLM.from_pretrained(
            st.session_state.model_path
        )

        input_text = f"Question: {prompt}\nAnswer:"

        inputs = tokenizer(
            input_text,
            return_tensors="pt"
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        response = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        st.subheader("Model Response")
        st.write(response)
