import re
from collections import defaultdict

#---------------LOAD & PREPROCESS DATA------------------------
def load_and_preprocess_text(filepath, start_marker="From fairest creatures we desire increase,"):

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Find the starting point in the text
        start_index = text.find(start_marker)
        if start_index != -1:
            text = text[start_index + len(start_marker):]  # Skip to the relevant content
        else:
            print(f"Start '{start_marker}' not found. Processing entire file.")
        
        # Preprocess the text
        text = text.lower()
        text = re.sub(r"[^\w\s']", "", text)  # Remove punctuation except apostrophes
        tokens = text.split()  # Split into tokens
        
        return tokens
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

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

#---------------BIGRAM NEXT TOKEN------------------------

def build_bigram_next_token_counts(tokens):
    from_bigram_to_next_token_counts = defaultdict(lambda: defaultdict(int))
    
    # Iterate through the tokens to build the bigram -> next token counts
    for i in range(len(tokens) - 2):  # Stop at len(tokens) - 2 to include the next token
        bigram = (tokens[i], tokens[i + 1])  # Current bigram
        next_token = tokens[i + 2]  # The token that follows the bigram
        from_bigram_to_next_token_counts[bigram][next_token] += 1  # Increment count
    
    return from_bigram_to_next_token_counts

#---------------MAIN EXECUTION------------------------

# Filepath to Shakespeare's works
filepath = "shakespeare_sonnet.txt"

# Load and preprocess the text (skipping content before the marker)
tokens = load_and_preprocess_text(filepath)

# Display basic token information
print(f"Total number of tokens: {len(tokens)}")
print(f"First 50 tokens of the relevant content: {tokens[:50]}")

# Create bigrams
bigrams = create_bigrams(tokens)

# Display bigram information
print(f"Total number of bigrams: {len(bigrams)}")
print(f"First 10 bigrams: {bigrams[:10]}")

# Build the bigram-to-next-token counts dictionary
from_bigram_to_next_token_counts = build_bigram_next_token_counts(tokens)

# Display bigram-next-token counts
print("\nBigram to next-token counts:")
for bigram, next_tokens in list(from_bigram_to_next_token_counts.items())[:5]:  # Display first 5 bigrams
    print(f"{bigram}: {dict(next_tokens)}")
