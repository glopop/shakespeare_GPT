import re

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation using regular expressions
    # This removes everything except letters, numbers, and spaces
    text = re.sub(r"[^\w\s]", "", text)
    
    # Split the text into tokens (words) based on spaces
    tokens = text.split()
    
    return tokens

# Load the text file
def load_text(filepath):
    """Loads text from a file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

# Filepath to Shakespeare's works
filepath = "shakespeare.txt"

# Load and preprocess the text
raw_text = load_text(filepath)
tokens = preprocess_text(raw_text)

# Display results
print(f"Total number of tokens: {len(tokens)}")
print(f"First 20 tokens: {tokens[:20]}")

