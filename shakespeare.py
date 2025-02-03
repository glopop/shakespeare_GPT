import re
from collections import defaultdict
import random

#---------------LOAD & PREPROCESS DATA------------------------
# function to load and preprocess the text file
def load_and_preprocess_text(filepath, start_marker="From fairest creatures we desire increase,"):
    try:
        # open the file in read mode
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # find the starting point of shakespeare 
        start_index = text.find(start_marker)
        if start_index != -1:
            # text to start from the point that we need
            text = text[start_index + len(start_marker):]
        else:
            # if we cant find the marker, print out this message
            print(f"start '{start_marker}' not found so processing entire file.")
        
        # lowercase
        text = text.lower()
        # remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", "", text)
        # split the text into tokens
        tokens = text.split()
        
        # return the list of tokens
        return tokens
    except FileNotFoundError:
        # handle the case when file is not found
        print(f"error: the file '{filepath}' was not found.")
        return []
    except Exception as e:
        # anyother exceptions
        print(f"an error occurred: {e}")
        return []

# function to preprocess plain text
def preprocess_text(text):
    # lowercase
    text = text.lower()
    # remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", "", text)
    # split the text into tokens
    tokens = text.split()
    # return the tokens
    return tokens

# function to load text from a file
def load_text(filepath):
    # open the file and read its contents
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

# function to create bigrams from tokens
def create_bigrams(tokens):
    # generate bigrams using consecutive tokens
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

#---------------BIGRAM NEXT TOKEN------------------------

# function to build bigram next token counts
def build_bigram_next_token_counts(tokens):
    # create a nested dictionary to store counts
    from_bigram_to_next_token_counts = defaultdict(lambda: defaultdict(int))
    
    # iterate through tokens to count next tokens for each bigram
    for i in range(len(tokens) - 2):
        bigram = (tokens[i], tokens[i + 1])  # current bigram
        next_token = tokens[i + 2]  # token following the bigram
        from_bigram_to_next_token_counts[bigram][next_token] += 1  # increment 
    
    # return the dictionary of bigram counts
    return from_bigram_to_next_token_counts

#---------------BIGRAM NEXT TOKEN PROBABILITY------------------------

# function to calculate bigram next token probabilities
def calculate_bigram_next_token_probs(from_bigram_to_next_token_counts):
    # create a dictionary to store probabilities
    from_bigram_to_next_token_probs = defaultdict(dict)
    
    # iterate through bigram counts to calculate probabilities
    for bigram, next_token_counts in from_bigram_to_next_token_counts.items():
        total_count = sum(next_token_counts.values())  # total occurrences of the bigram
        # calculate probability for each next token
        next_token_probs = {token: round(count / total_count, 2) for token, count in next_token_counts.items()}
        from_bigram_to_next_token_probs[bigram] = next_token_probs  # store probabilities
    
    # return the dictionary of probabilities
    return from_bigram_to_next_token_probs

#---------------SAMPLING------------------------

# function to sample the next token based on probabilities
def sample_next_token(bigram, from_bigram_to_next_token_probs):
    # check if the bigram exists in the dictionary
    if bigram not in from_bigram_to_next_token_probs:
        return None  # not found

    # retrieve tokens and probabilities for the bigram
    next_token_probs = from_bigram_to_next_token_probs[bigram]
    tokens = list(next_token_probs.keys())
    probabilities = list(next_token_probs.values())
    
    # sample the next token using weighted random choice
    next_token = random.choices(tokens, weights=probabilities, k=1)[0]
    return next_token

#---------------TEXT GENERATION------------------------

# function to generate text starting from a bigram
def generate_text_from_bigram(initial_bigram, word_count, from_bigram_to_next_token_probs):
    # initialize the current bigram and generated words
    current_bigram = initial_bigram
    generated_words = list(current_bigram)
    
    # generate the specified number of words
    for _ in range(word_count - 2):  # subtract 2 because the initial bigram has two words
        next_token = sample_next_token(current_bigram, from_bigram_to_next_token_probs)
        # stop generation if no next token is found
        if not next_token:
            break
        # add the sampled token to the list of words
        generated_words.append(next_token)
        # update the current bigram
        current_bigram = (current_bigram[1], next_token)
        
        # join and return the generated text
    return " ".join(generated_words)

#---------------BUILD TRIGRAM------------------------

def build_trigram_next_token_counts(tokens):
    from_trigram_to_next_token_counts = defaultdict(lambda: defaultdict(int))
    
    for i in range(len(tokens) - 3):  # Stop at len(tokens) - 3 for trigrams
        trigram = (tokens[i], tokens[i + 1], tokens[i + 2])  # Current trigram
        next_token = tokens[i + 3]  # The token that follows the trigram
        from_trigram_to_next_token_counts[trigram][next_token] += 1
    
    return from_trigram_to_next_token_counts



#---------------TRIGRAM PROB------------------------

def calculate_trigram_next_token_probs(from_trigram_to_next_token_counts):
    from_trigram_to_next_token_probs = defaultdict(dict)
    
    for trigram, next_token_counts in from_trigram_to_next_token_counts.items():
        total_count = sum(next_token_counts.values())
        next_token_probs = {token: round(count / total_count, 2) for token, count in next_token_counts.items()}
        from_trigram_to_next_token_probs[trigram] = next_token_probs
    
    return from_trigram_to_next_token_probs


#---------------BUILD QUADGRAM------------------------

def build_quadgram_next_token_counts(tokens):
    from_quadgram_to_next_token_counts = defaultdict(lambda: defaultdict(int))
    
    for i in range(len(tokens) - 4):  # Stop at len(tokens) - 4 for quadgrams
        quadgram = (tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3])  # Current quadgram
        next_token = tokens[i + 4]  # The token that follows the quadgram
        from_quadgram_to_next_token_counts[quadgram][next_token] += 1
    
    return from_quadgram_to_next_token_counts

#---------------QUADGRAM PROB------------------------

def calculate_quadgram_next_token_probs(from_quadgram_to_next_token_counts):
    from_quadgram_to_next_token_probs = defaultdict(dict)
    
    for quadgram, next_token_counts in from_quadgram_to_next_token_counts.items():
        total_count = sum(next_token_counts.values())
        next_token_probs = {token: round(count / total_count, 2) for token, count in next_token_counts.items()}
        from_quadgram_to_next_token_probs[quadgram] = next_token_probs
    
    return from_quadgram_to_next_token_probs

#---------------EX------------------------

def generate_text_from_trigram(initial_trigram, word_count, from_trigram_to_next_token_probs):
    current_trigram = initial_trigram
    generated_words = list(current_trigram)
    
    for _ in range(word_count - 3):  # Subtract 3 because the initial trigram already has three words
        next_token = sample_next_token(current_trigram, from_trigram_to_next_token_probs)
        if not next_token:
            break
        
        generated_words.append(next_token)
        current_trigram = (current_trigram[1], current_trigram[2], next_token)
    
    return " ".join(generated_words)


def generate_text_from_quadgram(initial_quadgram, word_count, from_quadgram_to_next_token_probs):
    current_quadgram = initial_quadgram
    generated_words = list(current_quadgram)
    
    for _ in range(word_count - 4):  # Subtract 4 because the initial quadgram already has four words
        next_token = sample_next_token(current_quadgram, from_quadgram_to_next_token_probs)
        if not next_token:
            break
        
        generated_words.append(next_token)
        current_quadgram = (current_quadgram[1], current_quadgram[2], current_quadgram[3], next_token)
    
    return " ".join(generated_words)




#---------------MAIN------------------------

filepath = "shakespeare_sonnet.txt"  # path to the text file

# load and preprocess the text (done while skipping the extra web info before the marker)
tokens = load_and_preprocess_text(filepath)

# token information
print(f"total number of tokens: {len(tokens)}")
print(f"first 50 tokens: {tokens[:50]}")

# create bigrams
bigrams = create_bigrams(tokens)

# bigram information
print(f"total number of bigrams: {len(bigrams)}")
print(f"frist 10 bigrams: {bigrams[:10]}")

# build bigram counts
from_bigram_to_next_token_counts = build_bigram_next_token_counts(tokens)

# show bigram-next-token counts
print("\n bigram to next-token counts:")
for bigram, next_tokens in list(from_bigram_to_next_token_counts.items())[:5]:  
    print(f"{bigram}: {dict(next_tokens)}")
    
    
# calculate bigram probabilities
from_bigram_to_next_token_probs = calculate_bigram_next_token_probs(from_bigram_to_next_token_counts)

#  bigram-next-token probabilities
print("\nbigram to next-token probabilities:")
for bigram, next_tokens in list(from_bigram_to_next_token_probs.items())[:5]:  
    print(f"{bigram}: {next_tokens}")

# test
test_bigram = ('to', 'be')

# sample the next token based on the probabilities
next_token = sample_next_token(test_bigram, from_bigram_to_next_token_probs)

#  sampled token
print(f"next token after {test_bigram}: {next_token}")  

#  text starting from an initial bigram
initial_bigram = ('to', 'be')
word_count = 50

generated_text = generate_text_from_bigram(initial_bigram, word_count, from_bigram_to_next_token_probs)

# trigram and quadgram counts
from_trigram_to_next_token_counts = build_trigram_next_token_counts(tokens)
from_quadgram_to_next_token_counts = build_quadgram_next_token_counts(tokens)

# trigram and quadgram probabilities
from_trigram_to_next_token_probs = calculate_trigram_next_token_probs(from_trigram_to_next_token_counts)
from_quadgram_to_next_token_probs = calculate_quadgram_next_token_probs(from_quadgram_to_next_token_counts)

#text from trigrams
initial_trigram = ('to', 'be', 'or')
trigram_generated_text = generate_text_from_trigram(initial_trigram, 50, from_trigram_to_next_token_probs)
print(f"\text from trigram:\n{trigram_generated_text}")

#text from quadgrams
initial_quadgram = ('to', 'be', 'or', 'not')
quadgram_generated_text = generate_text_from_quadgram(initial_quadgram, 50, from_quadgram_to_next_token_probs)
print(f"\text from quadgram:\n{quadgram_generated_text}")

print(f"text:\n{generated_text}")


