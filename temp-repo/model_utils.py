from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, pipeline
import torch

# === Main fine-tuned model setup ===
model_path = "./fine_tuned_model"
config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
model.eval()

# Label mapping for tweet_eval-style fine-tuned model
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

def classify_text(text):
    if not text.strip():
        return "Invalid Input", 0.0

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, label_id = torch.max(probs, dim=1)

    label = label_map.get(label_id.item(), "Unknown")
    confidence_percent = round(confidence.item() * 100, 2)  # e.g., 79.87%

    return label, confidence_percent

# === Backup model: Zero-Shot Classifier ===
# Loads once and reuses (slow only at startup)
# zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
zero_shot = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")


def backup_classify(text, candidate_labels=["Positive", "Neutral", "Negative"]):
    if not text.strip():
        return "Invalid Input", 0.0

    result = zero_shot(text, candidate_labels)
    label = result["labels"][0]
    confidence = round(result["scores"][0] * 100, 2)

    return label, confidence
