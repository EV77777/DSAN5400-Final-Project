from transformers import pipeline
import pandas as pd
from tqdm.auto import tqdm

# Load the FinBERT sentiment analysis pipeline
pipe = pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert", device=0)

# Load the CSV file
df = pd.read_csv("test.csv")

# Extract the second column containing the texts
texts = df.iloc[:, 1]

# Process the texts using the FinBERT model
results = []
for text in tqdm(texts):
    try:
        # Run the text through the FinBERT pipeline
        sentiment = pipe(text, truncation=True)
        results.append({"text": text, "label": sentiment[0]["label"], "score": sentiment[0]["score"]})
    except Exception as e:
        results.append({"text": text, "error": str(e)})

# Save results to a new CSV file
output_df = pd.DataFrame(results)
output_df.to_csv("finbert_test.csv", index=False)

print("Processing complete. Results saved to 'finbert_test.csv'.")
