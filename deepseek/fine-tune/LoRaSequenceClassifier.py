import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score
import bitsandbytes
import evaluate
import numpy as np


class LoRaSequenceClassifier:
    def __init__(self, 
                 model_name, 
                 num_labels=3, 
                 final_output_dir="./finetuned_classification_model",
                 mapping={"positive": 0, "negative": 1, "neutral": 2}
        ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.final_output_dir = final_output_dir
        self.id2label = mapping
        self.label2id = {value: key for key, value in mapping.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        #quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels, 
            id2label=self.id2label, 
            label2id=self.label2id,
            #load_in_8bit=True,
            #quantization_config=quantization_config,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Apply LoRa 
        self.lora_config = LoraConfig(
            r=32,  
            lora_alpha=8,
            lora_dropout=0.1,
            task_type=TaskType.SEQ_CLS,
        )

        self.model = get_peft_model(self.model, self.lora_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print("Trainable parameters:")
        self.model.print_trainable_parameters()


    def compute_metrics(self, eval_pred):
        clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
        predictions, labels = eval_pred
        predictions = 1/(1 + np.exp(-predictions))
        predictions = (predictions > 0.5).astype(int)
        predictions = predictions.reshape(-1)
        labels = labels.astype(int).reshape(-1)
        return clf_metrics.compute(predictions=predictions, references=labels)

    def fine_tune(self, train_dataset, eval_dataset, num_train_epochs=1, learning_rate=2e-5, batch_size=1):

        training_args = TrainingArguments(
            output_dir=self.final_output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            fp16=True,
            warmup_ratio=0.1,
            logging_steps=500,
            save_steps=1000,
            eval_strategy="steps",
            eval_steps=500,
            learning_rate=learning_rate,
            logging_dir="./logs",
            report_to="none",
            run_name="Sequence_Classification_Fine_Tune",
            load_best_model_at_end=True,
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        trainer.save_model(self.final_output_dir)
        self.tokenizer.save_pretrained(self.final_output_dir)

    def get_inference_model(self):

        peft_model = AutoModelForSequenceClassification.from_pretrained(
            self.final_output_dir, 
            num_labels=self.num_labels, 
            id2label=self.id2class, 
            label2id=self.label2id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        tokenizer = AutoTokenizer.from_pretrained(self.final_output_dir)

        return peft_model, tokenizer