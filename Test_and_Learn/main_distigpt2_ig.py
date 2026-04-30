# =========================
# 1. Setup
# =========================
import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# :white_check_mark: FIX 1: define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2. Load Model
# =========================
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # :white_check_mark: FIX 2
model.eval()

# =========================
# 3. Load Dataset
# =========================
data = load_dataset("google/civil_comments")["test"].select(range(20))

# =========================
# 4. Prompt
# =========================
def create_prompt(text):
    return f"Is this text toxic? Answer: {text}"

# =========================
# 5. Zero-shot scoring
# =========================
def score_text(text):
    prompt = create_prompt(text)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    last_token_logits = logits[0, -1]

    # :white_check_mark: FIX 3: safer tokens
    toxic_token = tokenizer.encode(" toxic", add_special_tokens=False)[0]
    safe_token = tokenizer.encode(" safe", add_special_tokens=False)[0]

    score = last_token_logits[toxic_token] - last_token_logits[safe_token]
    pred = "toxic" if score > 0 else "non-toxic"

    return score.item(), pred

# =========================
# 6. Integrated Gradients
# =========================
def integrated_gradients(text, steps=20):
    prompt = create_prompt(text)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # :white_check_mark: FIX 1: detach embeddings (no graph reuse)
    embeddings = model.transformer.wte(input_ids).detach()

    baseline = torch.zeros_like(embeddings)

    grads = []

    for i in range(steps + 1):
        alpha = float(i) / steps

        scaled = baseline + alpha * (embeddings - baseline)
        scaled.requires_grad_(True)

        outputs = model(inputs_embeds=scaled)
        logits = outputs.logits[:, -1, :]

        toxic_token = tokenizer.encode(" toxic", add_special_tokens=False)[0]
        score = logits[:, toxic_token].sum()

        # :white_check_mark: FIX 2: use autograd.grad instead of backward
        grad = torch.autograd.grad(score, scaled)[0]

        grads.append(grad.detach())

    avg_grads = torch.mean(torch.stack(grads), dim=0)

    attributions = (embeddings - baseline) * avg_grads
    attributions = attributions.sum(dim=-1).squeeze()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return tokens, attributions.cpu().numpy()

# =========================
# 7. Run
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
# 8. Save
# =========================
with open("output_with_ig.json", "w") as f:
    json.dump(results, f, indent=4)

print("Done! Saved to output_with_ig.json")