# =========================
# 1. Setup
# =========================
import torch
import json
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2. Load model
# =========================
model_name = "unitary/toxic-bert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

# ✅ Get correct label mapping
id2label = model.config.id2label
label2id = model.config.label2id

print("Label mapping:", id2label)

# Dynamically find toxic / non-toxic indices
toxic_index = None
non_toxic_index = None

for idx, label in id2label.items():
    if "toxic" in label.lower():
        toxic_index = idx
    else:
        non_toxic_index = idx

# =========================
# 3. Load dataset
# =========================
data = load_dataset("google/civil_comments")["test"].select(range(20))

# =========================
# 4. Scoring function
# =========================
def score_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]

    # ✅ Convert to probabilities
    probs = F.softmax(logits, dim=0)

    toxic_prob = probs[toxic_index].item()
    non_toxic_prob = probs[non_toxic_index].item()

    pred = "toxic" if toxic_prob > non_toxic_prob else "non-toxic"

    return toxic_prob, pred

# =========================
# 5. Integrated Gradients
# =========================
def integrated_gradients(text, steps=20):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]

    embed_layer = model.bert.embeddings.word_embeddings

    embeddings = embed_layer(input_ids).detach()
    baseline = torch.zeros_like(embeddings)

    grads = []

    for i in range(steps + 1):
        alpha = float(i) / steps

        scaled = baseline + alpha * (embeddings - baseline)
        scaled.requires_grad_(True)

        outputs = model(inputs_embeds=scaled)
        logits = outputs.logits[:, toxic_index]

        score = logits.sum()

        grad = torch.autograd.grad(score, scaled)[0]
        grads.append(grad.detach())

    avg_grads = torch.mean(torch.stack(grads), dim=0)

    attributions = (embeddings - baseline) * avg_grads
    attributions = attributions.sum(dim=-1).squeeze()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return tokens, attributions.cpu().numpy()

# =========================
# 6. Run
# =========================
results = []

for item in data:
    text = item["text"]

    toxic_prob, pred = score_text(text)
    tokens, attributions = integrated_gradients(text)

    results.append({
        "text": text,
        "prediction": pred,
        "toxic_probability": toxic_prob,
        "tokens": tokens,
        "importance": attributions.tolist()
    })

# =========================
# 7. Save
# =========================
with open("bert_ig_output.json", "w") as f:
    json.dump(results, f, indent=4)

print("Done! Saved to bert_ig_output.json")
