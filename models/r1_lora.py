import evaluate
import numpy as np
import pandas as pd

from torch import bfloat16
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

class LlamaClassifier:
    def __init__(self, lr=None, train_batch=None, eval_batch=None, weight_decay=None, 
                 lora_r=16, lora_alpha=16, lora_dropout=0.1):
        # Set fixed seed for reproducible training
        self._set_seeds()

        self._model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self._label_list = []
        self._id2label = self._label2id = {}

        print("Loading datasets...")
        self._train_df = pd.read_csv("../datasets/propaganda_train.tsv", sep="\t")
        self._test_df = pd.read_csv("../datasets/propaganda_val.tsv", sep="\t")

        print("Loading tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)

        print("Prepping datasets...")
        self._prep_dataset()

        print("Loading base model...")
        self._base_model = AutoModelForSequenceClassification.from_pretrained(
            self._model_id,
            num_labels=len(self._label2id),
            id2label=self._id2label,
            label2id=self._label2id,
            torch_dtype=bfloat16
        )

        # Configuring LoRA
        print("Applying LoRA configuration...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Transformer attention layers, special values bcs fking deepseek MLA
        )
        
        # Create PEFT model
        self._model = get_peft_model(self._base_model, peft_config)
        print(f"LoRA applied. Trainable parameters: {self._get_trainable_param_count()}")

        self._accuracy = evaluate.load("accuracy")
        self._f1 = evaluate.load("f1")

        # Hyperparams
        self._lr = lr if lr else 1e-5
        self._train_batch = train_batch if train_batch else 16
        self._eval_batch = eval_batch if eval_batch else 16
        self._weight_decay = weight_decay if weight_decay else 0.01

    def _prep_dataset(self):
        self._train_df = self._train_df[self._train_df["label"] != "not_propaganda"]
        self._test_df = self._test_df[self._test_df["label"] != "not_propaganda"]

        # Create unified label mapping from all possible labels
        all_labels = sorted(set(self._train_df['label']).union(set(self._test_df['label'])))
        self._label2id = {label: i for i, label in enumerate(all_labels)}
        self._id2label = {i: label for label, i in self._label2id.items()}

        # Map labels to IDs
        self._train_df['label_id'] = self._train_df['label'].map(self._label2id)
        self._test_df['label_id'] = self._test_df['label'].map(self._label2id)
        self._train_df = self._train_df.drop(columns=["label"])
        self._test_df = self._test_df.drop(columns=["label"])

        # Rename and convert to Hugging Face Datasets
        self._train_df = Dataset.from_pandas(self._train_df.rename(columns={"tagged_in_context": "text", "label_id": "label"}))
        self._test_df = Dataset.from_pandas(self._test_df.rename(columns={"tagged_in_context": "text", "label_id": "label"}))
        self._train_df = self._train_df.map(self._tokenize_function, batched=True)
        self._test_df = self._test_df.map(self._tokenize_function, batched=True)

    def _tokenize_function(self, input_stream: dict):
        return self._tokenizer(input_stream["text"], padding="max_length", truncation=True)

    def _get_trainable_param_count(self):
        """Count trainable parameters of the model"""
        trainable_params = 0
        all_params = 0
        for _, param in self._model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        return f"{trainable_params/1e6:.2f}M / {all_params/1e6:.2f}M ({trainable_params/all_params:.2%})"

    def train_model(self, output_dir="./propaganda_deberta_lora"):
        training_args = TrainingArguments(
            output_dir=output_dir,
            save_strategy="epoch",
            logging_dir="./logs",
            learning_rate=self._lr,
            per_device_train_batch_size=self._train_batch,
            per_device_eval_batch_size=self._eval_batch,
            num_train_epochs=3,
            weight_decay=self._weight_decay,
            metric_for_best_model="f1",
            save_total_limit=2,
            #gradient_checkpointing=True, # Reduce VRAM
            gradient_accumulation_steps=8, # 4x train batch size
            bf16=True
        )
        
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=self._train_df,
            eval_dataset=self._test_df,
            tokenizer=self._tokenizer,
            compute_metrics=self._compute_metrics,
        )
        
        print("Starting training with LoRA...")
        trainer.train()
        eval_results = trainer.evaluate()
        print(f"Final evaluation results: {eval_results}")
        
        # Save the adapter separately (much smaller than full model - yippie)
        self._model.save_pretrained(f"{output_dir}/final_adapter")
        
        return eval_results
    
    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        results = {
            "accuracy": self._accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1": self._f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
        }
        return results
    
    def _set_seeds(self):
        seed = 42
        np.random.seed(seed)
        set_seed(seed)  # Hugging Face-specific

    def get_metrics(self):
        pass


if __name__ == "__main__":
    classifier = LlamaClassifier(
        lr=7e-5,  # Higher learning rate often works better with LoRA
        train_batch=1,
        eval_batch=1,
        lora_r=16,  # Rank of LoRA decomposition
        lora_alpha=16,  # Alpha scaling factor, keep at 16
        lora_dropout=0.1
    )
    results = classifier.train_model()
    print(f"Training completed with results: {results}")