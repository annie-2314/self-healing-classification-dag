# 🧠 Self-Healing Classification DAG with Fine-Tuned Model


This project implements a robust sentiment classification CLI application using a fine-tuned Transformer model and a self-healing mechanism powered by LangGraph.

It performs inference using a primary model, evaluates confidence, and triggers fallback (via a user or a backup model) when necessary — prioritizing correctness over blind automation.
## 🎥 Video Demo

Watch the full demo of the project in action:

🔗 [Click here to view the demo video](https://www.loom.com/share/d9b1389cb52045109932663fd5418841?sid=ff227e8f-e28f-46b8-878c-54c55feb7f36)

---

## 🚀 Live Demo

You can explore the optional Streamlit UI for this project at the link below:

🔗 **[Launch the Live App](https://self-healing-classification-dag-cjmz7mvi5mytysocaztzsw.streamlit.app/)**


> 💡 Note: The primary interface for this project is via the command-line (`cli.py`). The web UI is provided as an additional enhancement for demonstration and multi-input testing.


---
## 📸 Project Screenshots

---

### 1. Training Process Screenshot
This screenshot captures the model training phase, showing loss reduction and metrics improvement.

![Training Process](assets/Training%20Process%20Screenshot.png)

---

### 2. CLI Output - Running `cli.py`
These screenshots show the command-line execution of the `cli.py` script, including predicted labels, fallback logic, and user clarification for low-confidence predictions.

![CLI Output 1](assets/CLI%20Output%20-%20Running%201.png)  
![CLI Output 2](assets/CLI%20Output%20-%20Running%202.png)

---

### 3. Single or Multiple Input Options
This screen shows that the UI allows both single and multiple text inputs for classification.

![Single or Multiple Inputs](assets/Single%20or%20Multiple%20Input%20Options.png)

---

### 4. Single Input Classification Output
After providing a single input, the app returns the predicted label with high accuracy.

![Single Input Output](assets/Single%20Input%20Classification%20Output.png)

---

### 4A. Single Input Output Display
This screenshot displays how the single input prediction is rendered clearly in the UI.

![Single Input Output Display](assets/single%20input%20output.png)

---

### 5. Multiple Inputs Classification Output
Here, multiple sentences were input, and the app successfully predicted labels for each.

![Multiple Inputs Output](assets/Multiple%20Inputs%20Classification%20Output.png)

---

### 6. Graph Generated from All Inputs
The output visualization is presented in a graph format to show label distribution or confidence scores.

![Graph Output](assets/Graph%20Generated%20from%20All%20Inputs.png)






## 📁 Project Structure

```
.
├── cli.py                    # Main CLI interface
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

 If you face version conflicts, run the following command to explicitly install compatible versions:

 pip install transformers==4.41.1 sentence-transformers==4.1.0

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
| ✅ Demo video              | 🔗 [Click here to view the demo video]([https://your-video-link.com](https://www.loom.com/share/c8e37178b1cd412c97bcac04c7ddc477?sid=9a163371-3f8a-4e00-8b2f-7f9ace34bab4)) |

---

## 🧠 Model Notes

- Fine-tuned using HuggingFace `Trainer` on `tweet_eval` (sentiment)
- LoRA-based training is possible with PEFT (optional)
- Backup model uses `facebook/bart-large-mnli`

---

## 📦 Requirements

```
streamlit>=1.38.0
transformers>=4.45.0
tokenizers>=0.19.1
torch>=2.0.0
langgraph==0.2.28
langgraph-checkpoint==1.0.12
huggingface_hub>=0.23.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
sentencepiece==0.2.0
accelerate>=0.28.0
peft>=0.10.0
python-dotenv>=1.0.1
```

---

