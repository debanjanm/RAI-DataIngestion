import pandas as pd
import lmstudio as lms
from tqdm import tqdm

# Load CSV
df = pd.read_csv("arxiv_data/all_chunks.csv")

df = df[:5]

# Initialize LLM
model = lms.llm("qwen2.5-vl-7b-instruct")

# Function to clean text
def clean_text(text):
    prompt = f"Clean and normalize this text (remove noise, fix formatting, keep meaning intact):\n\n{text}"
    try:
        result = model.respond(prompt)
        return result.strip()
    except Exception as e:
        print(f"Error: {e}")
        return text  # fallback

# Iterate one by one with tqdm
cleaned_texts = []
for text in tqdm(df["chunk_text"], desc="Cleaning texts"):
    cleaned_texts.append(clean_text(text))

# Save results
df["cleaned_text"] = cleaned_texts
df.to_csv("cleaned_output.csv", index=False)
