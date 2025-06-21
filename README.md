# 🧠 Self-Healing Classification DAG with Fine-Tuned Model


This project implements a robust sentiment classification CLI application using a fine-tuned Transformer model and a self-healing mechanism powered by LangGraph.

It performs inference using a primary model, evaluates confidence, and triggers fallback (via a user or a backup model) when necessary — prioritizing correctness over blind automation.

---

## 📁 Project Structure

```
.
├── cli.py                     # Main CLI interface
├── model_utils.py            # Inference and backup model loading
├── nodes.py                  # LangGraph nodes: ConfidenceCheck, Fallback
├── fine_tuned_model/         # Pre-trained model directory (local or from HF)
├── logs.txt                  # Logged predictions and fallbacks
├── confidence_curve.png      # Auto-generated plot (optional)
├── requirements.txt          # Python dependencies
└── README.md                 # You're reading this!
```

---

## 🔧 Features

- ✅ Fine-tuned DistilBERT for sentiment analysis (`tweet_eval` dataset)
- ✅ LangGraph DAG: Inference → Confidence Check → Fallback
- ✅ Confidence-based fallback with user clarification
- ✅ Backup zero-shot classifier using `facebook/bart-large-mnli`
- ✅ CLI logging (`logs.txt`)
- ✅ Confidence curve plot
- ✅ Fallback frequency summary

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/annie-2314/self-healing-classification-dag.git
cd langgraph-sentiment-cli
```

### 2. Create Virtual Environment (optional but recommended)

```bash
python -m venv env
source env/bin/activate     # On Windows: env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Use Fine-Tuned Model

This project uses a locally fine-tuned `DistilBERT` model on the `tweet_eval` sentiment classification dataset.

If `fine_tuned_model/` is not present:

1. Run the fine-tuning script to train your own model:
   ```bash
   python train_model.py


---

## 🧪 How to Run the Classifier

```bash
python cli.py
```

### Example

```bash
Input: I liked the visuals but the story was dull.
[InferenceNode] Predicted label: Neutral | Confidence: 52.13%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[BackupModel] Trying zero-shot fallback model...
[BackupModel] Predicted label: Negative | Confidence: 82.43%
Final Label: Negative (From backup model)
```

---

## 📊 Bonus Features

### 📈 Confidence Curve

After several inputs, a plot is generated:

```
confidence_curve.png
```

It shows how confident the model was across inputs.

### 📉 Fallback Frequency Summary

After exiting the CLI (`exit` command), you will see:

```
Fallback Stats:
- Total Inputs: 10
- Fallback Triggered: 4
- Backup Model Used: 3
- User Clarifications: 1
```

---

## 📁 Deliverables Summary

| Deliverable              | Status       |
|--------------------------|--------------|
| ✅ Fine-tuned model       | Included / Download link |
| ✅ Source Code            | ✔️ All scripts provided |
| ✅ README.md              | ✔️ You're reading it |
| ✅ logs.txt               | Auto-generated |
| ✅ Demo video (optional)  | Add separately |

---

## 🧠 Model Notes

- Fine-tuned using HuggingFace `Trainer` on `tweet_eval` (sentiment)
- LoRA-based training is possible with PEFT (optional)
- Backup model uses `facebook/bart-large-mnli`

---

## 📦 Requirements

```
transformers
torch
langgraph
matplotlib
huggingface_hub
```

---

## 📽️ Demo Video

📺 *[Add YouTube or Drive link here]*

---

## 🧑‍💻 Author

**Your Name** – `your.email@example.com`  
Project for: *LangGraph AI Challenge*

---
