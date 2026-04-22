# 1. Imports
#######################################################################################
# Import the necessary modules.  Where possible, use Auto- modules for portability
#######################################################################################
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate
from datetime import datetime
from transformers import set_seed
set_seed(42)  # (reproducibility)

########################################################################################
# 2. Set the hyper-parameters
########################################################################################
# Hyper-parameters
########################################################################################

OUTPUT_DIR="./results"
NUMBER_EPOCHS=7 
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=64
LEARNING_RATE=4.975e-6
WEIGHT_DECAY=0.01495
SCHEDULER_TYPE="linear"
WARMUP_STEPS=547  
EVAL_STRATEGY="epoch"
SAVE_STRATEGY="epoch"
LOAD_BEST_MODEL_AT_END=True
METRIC_FOR_BEST_MODEL="accuracy"
LOGGING_STEPS=100
FP16=True

########################################################################################
# 3. Load Dataset (IMDB sentiment - positive vs negative)
########################################################################################
# Load the dataset
########################################################################################

dataset = load_dataset("imdb")

########################################################################################
# 4. Load Tokenizer and Model
########################################################################################
# Load the model, tokenizer, using AutoTokenizer, AutoModelForSequenceClassification
########################################################################################

model_name = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    classifier_dropout=0.2   
)

########################################################################################
# 5. Tokenization
########################################################################################
# Tokenize the dataset
########################################################################################

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

########################################################################################
# 6. Data Collator & Metrics
########################################################################################
# Collate the dataset for padding and format
########################################################################################

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

########################################################################################
# 7. Set Training Arguments
########################################################################################
# Training Arguments
########################################################################################

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUMBER_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type=SCHEDULER_TYPE,
    warmup_steps=WARMUP_STEPS,
    eval_strategy=EVAL_STRATEGY,
    save_strategy=SAVE_STRATEGY,
    load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
    metric_for_best_model=METRIC_FOR_BEST_MODEL,
    greater_is_better=True,
    save_total_limit=2,
    logging_steps=LOGGING_STEPS,
    report_to="none",
    bf16=True,
    seed=42,
    data_seed=42,   
)

########################################################################################
# 8. Set the Trainer
########################################################################################
# Setup the training (finetuning) session
########################################################################################

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

########################################################################################
# 9. Train (finetune)
########################################################################################
# Start the trainer
########################################################################################

trainer.train()

########################################################################################
# 10. Plot with SINGLE clear Validation Loss line by epoch
########################################################################################
# Track your progress and plot graphs for visual inspection
########################################################################################

def plot_training_metrics(trainer):
    history = pd.DataFrame(trainer.state.log_history)
    
    ########################################################################################
    # Training loss (light blue, by step)
    ########################################################################################
    train_logs = history[history['loss'].notna()].copy()
    
    # Evaluation logs - ONLY these points for validation loss (by epoch)
    eval_logs = history[history['eval_loss'].notna()].copy()
    
    plt.figure(figsize=(12, 5))
    ########################################################################################
    # Left plot: Loss
    ########################################################################################
    plt.subplot(1, 2, 1)
    plt.plot(train_logs['step'], train_logs['loss'], 
             label='Training Loss', color='lightblue', alpha=0.7, linewidth=1)
    ########################################################################################
    # SINGLE prominent line for Validation Loss by epoch
    ########################################################################################
    plt.plot(eval_logs['step'], eval_logs['eval_loss'], 
             label='Validation Loss (by epoch)', 
             color='red', linewidth=3, marker='o', markersize=8)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    ########################################################################################
    # Include learning rate in the chart title
    ########################################################################################
    plt.title("roBERTa Learning Rate: " +str(LEARNING_RATE) + ' Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ########################################################################################
    # Right plot: Validation Accuracy
    ########################################################################################
    plt.subplot(1, 2, 2)
    if 'eval_accuracy' in eval_logs.columns:
        plt.plot(eval_logs['step'], eval_logs['eval_accuracy'], 
                 label='Validation Accuracy', color='green', linewidth=3, marker='s')
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy (Weight Decay = '+str(WEIGHT_DECAY)+')')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.text(2, 20, 'Weight_Decay = '+str(WEIGHT_DECAY), fontsize=10)
    plt.tight_layout()
    ########################################################################################
    # Save the chart_graphic with a unique name to allow comparing different tuning sessions
    ########################################################################################
    current_datetime = datetime.now()
    ########################################################################################
    # Format as "YYYY-MM-DD HH:MM:SS"
    ########################################################################################
    timestamp_str = current_datetime.strftime("___%H:%M:%S___%m-%d-%Y")
    filename = "roBERTa_Learning_Rate_" + str(LEARNING_RATE) + "__time__"+timestamp_str+".png" 
    plt.savefig(filename)
    plt.show()

plot_training_metrics(trainer)

########################################################################################
# 11. Final Evaluation & Save
########################################################################################
# Save the results of the finetuning session
########################################################################################

results = trainer.evaluate()
print("Final Evaluation Results:", results)

trainer.save_model("./fine_tuned_roberta_imdb")
tokenizer.save_pretrained("./fine_tuned_roberta_imdb")

########################################################################################
# 12. Fixed Inference (model moved to CPU)
########################################################################################
# Test inference with quick samples
########################################################################################

model = trainer.model.cpu()

def predict_sentiment(text):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return "Positive" if prediction == 1 else "Negative"



print("\nTest Predictions:")
print("Positive example:", predict_sentiment("This movie was absolutely fantastic and I loved every minute!"))
print("Negative example:", predict_sentiment("This was the worst film I have ever seen. Total waste of time."))


########################################################################################
# 12. Extended Test Inferences (~20 examples)
########################################################################################
print("\n=== Extended Test Inferences (20 examples) ===")

test_reviews = [
    "This movie was absolutely fantastic and I loved every minute of it!",
    "The worst film I have ever seen. Total waste of time and money.",
    "Great acting but the plot was boring and predictable.",
    "One of the best movies of the year! Highly recommended.",
    "Terrible script, bad acting, and zero entertainment value.",
    "I laughed, I cried, I couldn't stop watching. Masterpiece!",
    "Don't waste your time. This is the most disappointing movie ever.",
    "The visuals were stunning but the story fell flat.",
    "A brilliant performance by the lead actor. 10/10!",
    "Horrible dialogue and the ending made no sense at all.",
    "This film exceeded all my expectations. What a gem!",
    "Extremely slow-paced and not worth watching.",
    "The best superhero movie I've seen in years.",
    "Awful directing and terrible CGI. Avoid at all costs.",
    "Heartwarming story with excellent cast and soundtrack.",
    "Completely overrated. Nothing special here.",
    "Incredible cinematography and emotional depth.",
    "The acting was so bad it was almost funny.",
    "A perfect blend of humor, action, and drama. Loved it!",
    "One of the most boring experiences I've had at the cinema."
]

def predict_sentiment(text):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    confidence = torch.softmax(outputs.logits, dim=-1).max().item()
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment, confidence

# Run the tests
for i, review in enumerate(test_reviews, 1):
    sentiment, confidence = predict_sentiment(review)
    print(f"{i:2d}. {sentiment:8s} ({confidence:.4f}) | {review}")
