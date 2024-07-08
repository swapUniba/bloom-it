from transformers import AutoModel,TrainingArguments,AutoTokenizer,BloomForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from trl.trainer import SFTTrainer
import os
#PP
#import spacy
#from trl import DataCollatorForCompletionOnlyLM

model_name_or_path = "swap-uniba/bloom-1b7-it"
tokenizer_name_or_path = "swap-uniba/bloom-1b7-it"

dataset_path = "MBZUAI/Bactrian-X"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32
)

model = BloomForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"Di seguito è riportata un'istruzione che descrive un'attività, abbinata a un input che fornisce ulteriore contesto. Scrivi una risposta che completi adeguatamente la richiesta.\n ### Istruzione: {example['instruction'][i]}\n ### Input: {example['input'][i]}\n ### Risposta: {example['output'][i]}\n ### Fine"
        output_texts.append(text)
    return output_texts

dataset = load_dataset(dataset_path, 'it', split="train")
print(dataset)

import torch
torch.cuda.empty_cache()

output_dir = "bloom-1b7-bactrianx-it-out/"

# Epochs
num_train_epochs = 10

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "adamw_bnb_8bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
#max_steps = 25001

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Save checkpoint every X updates steps
save_steps = 10000

# Log every X updates steps
logging_steps = 1000

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = 1024

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

dev = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, add_eos_token=True, truncation=True, max_length=max_seq_length + 1, padding="max_length")
tokenizer.pad_token_id = 0
tokenizer.padding_side = "right"

#Train on completions only
#response_template = " ### Answer:"
#collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    #save_steps=save_steps,
    save_strategy="epoch",
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=True,
    bf16=False,
    log_level='debug',
    max_grad_norm=max_grad_norm,
    #max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    ddp_find_unused_parameters=False,
    report_to="all",
    save_safetensors=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    #eval_dataset=dataset["validation"],
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    dataset_num_proc=4,
    args=training_arguments,
    packing=packing,
    formatting_func=formatting_prompts_func,
    #data_collator=collator
)

# Train model
trainer.train()

new_model = "bloom-1b7-bactrianx-it"

# Saving model
print("Saving last checkpoint of the model...")
os.makedirs(new_model, exist_ok=True)
trainer.model.save_pretrained(new_model)
