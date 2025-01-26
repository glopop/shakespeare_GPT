import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)  # Remove punctuation (preserve apostrophes)
    tokens = text.split()  # Split into words
    return tokens

def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def create_bigrams(tokens):
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

# Filepath to Shakespeare's works
filepath = "shakespeare.txt"

# Load and preprocess the text
raw_text = load_text(filepath)
tokens = preprocess_text(raw_text)

# Create bigrams
bigrams = create_bigrams(tokens)

# Display results
print(f"Total number of tokens: {len(tokens)}")
print(f"First 20 tokens: {tokens[:20]}")
print(f"Total number of bigrams: {len(bigrams)}")
print(f"First 10 bigrams: {bigrams[:10]}")
