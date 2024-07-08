from transformers import AutoModel,TrainingArguments,AutoTokenizer,BloomForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from trl.trainer import SFTTrainer
import os
import spacy
import glob

model_name_or_path = "bigscience/bloom-1b7"
tokenizer_name_or_path = "bigscience/bloom-1b7"

train_path = "swap-uniba/bloom-1b7-it"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = BloomForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

nlp = spacy.load("it_core_news_sm", exclude = ["tagger","ner"])
max_text_token_length=500

def chunk_example(examples):
    chunks = []
    last_chunk = ""
    l = 0
    for text in examples["text"]:
        doc = nlp(text)
        for sent in doc.sents:
            ls = len(sent)
            if l+ls<max_text_token_length:
                last_chunk+=sent.text+" "
                l+=ls
            else:
                chunks.append(last_chunk)
                last_chunk=""
                last_chunk=sent.text+" "
                l=ls
    return {"chunks": chunks}

train_files = glob.glob(train_path)
loaded_dataset = load_dataset("json", data_files={"train":train_files})
print(loaded_dataset)
dataset = loaded_dataset.map(chunk_example, batched=True, remove_columns=loaded_dataset["train"].column_names)
print(dataset)

output_dir = "./bloom_1b_it_v2_out"

# Epochs
num_train_epochs = 5

# Batch size per GPU for training
per_device_train_batch_size = 8

# Batch size per GPU for evaluation
per_device_eval_batch_size = 8

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
save_steps = 50000

# Log every X updates steps
logging_steps = 1000

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = 512

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

dev = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True, add_eos_token=True, padding="max_length")
#tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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
    dataset_text_field="chunks",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    dataset_num_proc=6,
    args=training_arguments,
    packing=packing
)

# Train model
trainer.train()

new_model = "bloom-1b-v2-it"

# Saving model
print("Saving last checkpoint of the model...")
os.makedirs(new_model, exist_ok=True)
trainer.model.save_pretrained(new_model)
