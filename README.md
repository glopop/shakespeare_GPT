# shakespeare_GPT 
Shakespearean Text Generator

An n-gram text generator designed to imitate the style of William Shakespeare. This project processes a corpus of Shakespeare’s works, builds probability distributions over bigrams (and optionally trigrams, quadgrams, etc.), and generates new text using weighted random sampling.

## TABLE OF CONTENTS 
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Tasks](#tasks)
   3.1 [TASK 1: DATA PREPARATION](#task-1-data-preparation)


## PROJECT OVERVIEW
The goal of this project is to make a text generation model that works with 2-grams/3-grams/4-grams to imitate the style of William Shakespeare. 

Key Steps:

Preprocessing: Convert text to lowercase, remove punctuation, split into tokens.

N-Gram Modeling: Build dictionaries mapping n-grams to possible next tokens.

Probability Distributions: Convert raw counts to probabilities.

Sampling: Use weighted random choice to select the next token.

Generation: Iteratively produce text from an initial context.

## INSTALLATION
Python 3.7+ (preferable).

pip for installing dependencies.

A text file containing Shakespeare’s works (shakespeare.txt). Make sure you have the right to use it for this project.

## TASKS
### TASK 1: DATA PREPARATION
The first step in this project is to load a text file containing Shakespeare's works. We must then preprocess the text by converting it to lowercase, removing punctuation, and splitting it into tokens. We also want to do a list of bigrams from the preprocessed text.

Then, we want to write the code that fills the from_bigram_to_next_token_counts dictionary, where each key is a bigram (tuple of two tokens) and the value is a dictionary of counts of tokens that follow the bigram. 

An example of this would be: from_bigram_to_next_token_counts[('to', 'be')] = {'or': 10, 'not': 5}

### TASK 2: PROBABILITY DISTRIBUTION
After completing the first task, we then want to do the code that fills the from_bigram_to_next_token_probs dictionary, where each key is a bigram and the value is a dictionary of probabilities of tokens that follow the bigram. We should use the counts from from_bigram_to_next_token_counts to calculate the probabilities. 

An example of this would be: from_bigram_to_next_token_probs[('to', 'be')] = {'or': 0.67, 'not': 0.33}

### TASK 3: SAMPLING NEXT TOKEN
Up next, we want to implement a sampling function (sample_next_token function) which should return the next token sampled based on the probability distribution from from_bigram_to_next_token_probs. It is important to do this sampling through a weighted random choice.

### TASK 4: GENERATING TEXT (yay!)
We finally reach the step where we can generate our text, through the implementation of the generate_text_from_bigram function, which generates text by starting with an initial bigram and sampling the next token iteratively. The function should return a string of generated text with a specified number of words. 

An example of this would be: generate_text_from_bigram(('to', 'be'), 50) might return "to be or not to be that is the question ..."

### TASK 5: EXPLORATION OF DIFFERENT N-GRAMS
This is where we can explore different variations of our n-grams, and experiment with trigrams and quadgrams. We can 
implement similar functions for these n-grams: from_trigram_to_next_token_counts, from_trigram_to_next_token_probs, from_quadgram_to_next_token_counts, and from_quadgram_to_next_token_probs. We can also analyse the differences in text generation quality between bigrams, trigrams, and quadgrams.

### TASK 6: HUMAN EVALUATION
This is where we decide how our machine is working. We can design a survey to collect feedback from human participants on the quality of the generated text. Then, we can decide how well our model imitates the style of Shakespeare.

