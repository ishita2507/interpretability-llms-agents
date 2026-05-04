# =========================
# 1. Setup
# =========================
import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2. Load BERT model
# =========================
model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
model.eval()

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

    # Assume index 1 = toxic
    score = logits[1] - logits[0]
    pred = "toxic" if score > 0 else "non-toxic"

    return score.item(), pred

# =========================
# 5. Integrated Gradients
# =========================
def integrated_gradients(text, steps=20):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]

    # :fire: BERT embedding layer
    embed_layer = model.bert.embeddings.word_embeddings

    embeddings = embed_layer(input_ids).detach()
    baseline = torch.zeros_like(embeddings)

    grads = []

    for i in range(steps + 1):
        alpha = float(i) / steps

        scaled = baseline + alpha * (embeddings - baseline)
        scaled.requires_grad_(True)

        outputs = model(inputs_embeds=scaled)
        logits = outputs.logits[:, 1]  # toxic class

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

    score, pred = score_text(text)
    tokens, attributions = integrated_gradients(text)

    results.append({
        "text": text,
        "prediction": pred,
        "score": score,
        "tokens": tokens,
        "importance": attributions.tolist()
    })

# =========================
# 7. Save
# =========================
with open("bert_ig_output.json", "w") as f:
    json.dump(results, f, indent=4)

print("Done! Saved to bert_ig_output.json")
