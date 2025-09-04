from datasets import load_dataset
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import TrainerCallback
import re
import torch
from transformers.utils import logging
import math
import ast
from tqdm import tqdm
import sys
from copy import deepcopy
import random

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")
logging.disable_progress_bar()
logging.enable_explicit_format()


# check CUDA
if torch.cuda.is_available():
  print("TRUE: cuda is available.")
else:
  print("FALSE: cuda is not available.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class EarlyStoppingCallback(TrainerCallback):
    """Stop training with patience treshold of 10"""
    def __init__(self, patience=10, min_delta=.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.steps_since_improved = 0

    def on_evaluate(self, args, state, control, **kwargs):
        current_loss = kwargs["metrics"]["eval_loss"]
        if self.best_loss is None or current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.steps_since_improved = 0
        else:
            self.steps_since_improved += 1

        if self.steps_since_improved >= self.patience:
            control.should_training_stop = True
            print(f"Early stopping triggered after {self.patience} steps of no improvement.")


def clean_metadata(data_dict, lan='eng'): #UPDATED 2024-4-15
    """
    Clean special tokens (e.g. _START_PARAGRAPH_) and empty lines
    from the DatasetDict dictionary.
    """
    txt = data_dict['text']
    ## DONT USE FOR ENGLISH ##
    #if lan != 'eng':
    #txt = ast.literal_eval(txt) # ADDED 2025-5-3
    #txt = txt.decode('utf-8') # ADDED 2025-5-3
    
    clean_1_txt = re.sub(r'\n_START_ARTICLE_\n(.+\n_START_SECTION_\n)?.+\n_START_PARAGRAPH_\n', ' ', txt)
    clean_2_txt = re.sub(r'\n_START_SECTION_\n.+\n_START_PARAGRAPH_\n', ' ', clean_1_txt)

    clean_3_txt = clean_2_txt.replace('\xa0', ' ') # ADDED 2025-5-3

    clean_4_txt = re.sub(r'_NEWLINE_', ' ', clean_3_txt)
    clean_5_txt = clean_4_txt.split()
    data_dict['text'] = ' '.join(clean_5_txt)
    return data_dict


def estimate_tokens(dataset, tokenizer, clean_func, token_bound, ctx_size):
    total_tokens = 0
    row_count = 0
    
    for i in tqdm(range(len(dataset)), desc="Estimating rows", unit="rows"):
        row = deepcopy(dataset[i])
        clean_row = clean_func(row)

        tokens = tokenizer(clean_row["text"], truncation=True, padding=False, max_length=ctx_size)["input_ids"]
        total_tokens += len(tokens)

        if total_tokens >= token_bound:
            row_count = i  # Include this row
            break

    return row_count, total_tokens 


# change data path
saved_data = "/proj/uppmax2024-2-24/magi5470/datasets" #UPPMAX
cache_data = "/proj/uppmax2024-2-24/magi5470/cache"

# 1. Load 'train' datasets
data = load_dataset('parquet', data_dir=saved_data) # DON'T USE SPLIT TO COUNT (177000 = 90M TOTAL (train+test+valid)

# 2. Check stuff
print('-'*30)
print('--Partitioned Dataset Size (Total Rows)')
print(f"Training dataset size: {len(data['train'])}")
print(f"Validation dataset size: {len(data['validation'])}")
print(f"Test dataset size: {len(data['test'])}")
print('-'*30)

# 3. Change context length: default 512
ctx = int(sys.argv[1])
rows = {'512': 283500, '256': 449019, '128': 788313}
rows_needed = rows.get(str(ctx))

print('--Using Configuration with:')
print(f'Context Window: {ctx}; Rows Needed: {rows_needed}')
print('-'*20)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
config = AutoConfig.from_pretrained("bert-base-cased", 
                                    vocab_size=len(tokenizer.vocab), 
                                    max_position_embeddings=ctx) # n_ctx= context length
model = BertForMaskedLM(config)
model.to(device)

#print('--PAD token:')
#print(tokenizer.pad_token_id)

# Enable gradient checkpointing to save memory
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled.")

# Disable cache to reduce memory usage during training
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False
    print("Model cache disabled for training efficiency.")

# print GPT parameter: n_ctx= context length
model_size = sum(t.numel() for t in model.parameters())
#print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
#print(config)


def tokenize(raw, ctx=ctx): 
    outputs = tokenizer(
        raw["text"],
        truncation=True,
        max_length=ctx,
        padding=True, # USE FALSE TO EXCLUDE PAD TOKENS
        return_overflowing_tokens=False,
    )
    return outputs


random.seed(42)
plus = int(rows_needed * 0.11) # +10% = 89M across models
print(f'--Adding Rows:')
print(f'Numer: {plus}')

scaled_data = data['train'].select(range(rows_needed+plus))
print(scaled_data)

valid_train = scaled_data.shuffle(seed=42).train_test_split(test_size=0.1, seed=42)
valid_test_split = valid_train['test'].shuffle(seed=42).train_test_split(test_size=0.5, seed=42)  # 5% each for val and test
train_dataset = valid_train['train']
val_dataset = valid_test_split['train']
test_dataset = valid_test_split['test']
splitted_datasets = DatasetDict({"train": train_dataset, "test": test_dataset, "validation": val_dataset})
print(splitted_datasets)

print('-'*30)
print('--Scaled dataset size (rows)')
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print('-'*30)
print('--Raw:')
print(splitted_datasets['train'][5]['text'])
raw_datasets = splitted_datasets.map(clean_metadata) 
print('-'*30)
print('--Cleaned:')
print(raw_datasets['train'][5]['text'])
print('-'*30)

print('Tokenizing dataset partitions...')
print('-'*30)
tokenized_datasets = raw_datasets.map(tokenize, 
                                          batched=True, 
                                          fn_kwargs={'ctx': ctx},
                                          remove_columns=['wikidata_id', 'text', 'version_id']) #  wiki40b dataset
dc = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
early_stop = EarlyStoppingCallback(patience=10, min_delta=.0)

print('--Debug Tokenized Length:')
print(len(tokenized_datasets['train']['input_ids'][0]))
print(tokenizer.convert_ids_to_tokens(tokenized_datasets['train']['input_ids'][5]))
    
print(f'--Counting Actual Tokens (Context Size of {ctx})...')
print('-'*30)
pad_token = tokenizer.pad_token_id
train_tokens = sum(sum(1 for token in text if token != pad_token) for text in tokenized_datasets['train']['input_ids'])
test_tokens = sum(sum(1 for token in text if token != pad_token) for text in tokenized_datasets['test']['input_ids'])
val_tokens = sum(sum(1 for token in text if token != pad_token) for text in tokenized_datasets['validation']['input_ids'])
total = train_tokens + test_tokens + val_tokens

print('--Cleaned and Tokenized Training Datasets (tokens):')
print(f"Training tokens: {train_tokens}")
print(f"Validation tokens: {val_tokens}")
print(f"Test tokens: {test_tokens}")
print(f"Total tokens: {total}")
print('-'*30)

args = TrainingArguments(
    output_dir=f"bert-models-new/model-{ctx}",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps", 
    save_strategy = "steps", 
    eval_steps=500,
    logging_steps=10, #default 500
    logging_strategy='steps',
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    adam_beta1=.9,
    adam_beta2=.999, 
    adam_epsilon=1e-8, 
    weight_decay=0.01,
    max_grad_norm=1.0,
    warmup_steps=10000,
    lr_scheduler_type="cosine",
    learning_rate=1e-4, 
    save_total_limit=5, # numbers of checkpoint
    report_to='tensorboard', 
    logging_dir=f'bert-models-new/logs-{ctx}',
    fp16=True, # for GPU only
    load_best_model_at_end=True, 
    gradient_checkpointing=True,
    save_only_model=True, # deprecated
    disable_tqdm=True,
    greater_is_better=False,
    metric_for_best_model='eval_loss'
)

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    data_collator=dc,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    callbacks=[early_stop],
    )

torch.cuda.empty_cache()
trainer.train()
trainer.save_metrics('train', trainer.state.log_history[-1])
trainer.log_metrics('train', trainer.state.log_history[-1])
trainer.save_model(f'bert-models-new/model-{ctx}/best_model')

eval_results = trainer.evaluate(tokenized_datasets['test'])
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
