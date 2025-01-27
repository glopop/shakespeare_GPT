import re
from collections import defaultdict
import random

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

#---------------BIGRAM NEXT TOKEN PROBABILITY------------------------

def calculate_bigram_next_token_probs(from_bigram_to_next_token_counts):
    from_bigram_to_next_token_probs = defaultdict(dict)
    
    for bigram, next_token_counts in from_bigram_to_next_token_counts.items():
        total_count = sum(next_token_counts.values())  # Total occurrences of the bigram
        # Calculate probability for each next token
        next_token_probs = {token: round(count / total_count, 2) for token, count in next_token_counts.items()}
        from_bigram_to_next_token_probs[bigram] = next_token_probs  # Store probabilities
    
    return from_bigram_to_next_token_probs

#---------------SAMPLING------------------------

def sample_next_token(bigram, from_bigram_to_next_token_probs):
    if bigram not in from_bigram_to_next_token_probs:
        return None  # Return None if the bigram is not found

    next_token_probs = from_bigram_to_next_token_probs[bigram]
    tokens = list(next_token_probs.keys())
    probabilities = list(next_token_probs.values())
    
    # Use random.choices() to sample the next token
    next_token = random.choices(tokens, weights=probabilities, k=1)[0]
    return next_token

#---------------TEXT GENERATION------------------------

def generate_text_from_bigram(initial_bigram, word_count, from_bigram_to_next_token_probs):
    
    # Start with the initial bigram
    current_bigram = initial_bigram
    generated_words = list(current_bigram)
    
    # Generate the specified number of words
    for _ in range(word_count - 2):  # Subtract 2 because the initial bigram already has two words
        next_token = sample_next_token(current_bigram, from_bigram_to_next_token_probs)
        
        # Stop generation if no next token is found
        if not next_token:
            break
        
        # Add the sampled token to the list of generated words
        generated_words.append(next_token)
        
        # Update the current bigram
        current_bigram = (current_bigram[1], next_token)
    
    # Join the generated words into a string and return
    return " ".join(generated_words)


#---------------MAIN------------------------

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
    
# Calculate probabilities from bigram-to-next-token counts
from_bigram_to_next_token_probs = calculate_bigram_next_token_probs(from_bigram_to_next_token_counts)

# Display bigram-next-token probabilities
print("\nBigram to next-token probabilities:")
for bigram, next_tokens in list(from_bigram_to_next_token_probs.items())[:5]:  # Display first 5 bigrams
    print(f"{bigram}: {next_tokens}")

# Test the sample_next_token function
test_bigram = ('to', 'be')

# Sample the next token based on the probabilities
next_token = sample_next_token(test_bigram, from_bigram_to_next_token_probs)

# Display the sampled token
print(f"Next token after {test_bigram}: {next_token}")  

# Generate text starting from an initial bigram
initial_bigram = ('to', 'be')
word_count = 50

generated_text = generate_text_from_bigram(initial_bigram, word_count, from_bigram_to_next_token_probs)

# Display the generated text
print(f"Generated text:\n{generated_text}")
