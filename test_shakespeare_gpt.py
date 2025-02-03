import unittest
from shakespeare import ( 
    preprocess_text, load_and_preprocess_text, create_bigrams, 
    build_bigram_next_token_counts, calculate_bigram_next_token_probs, 
    sample_next_token, generate_text_from_bigram 
)


class TestShakespeareGPT(unittest.TestCase):

    def setUp(self):
        """Set up a small sample text for testing."""
        self.sample_text = "to be or not to be that is the question"
        self.tokens = preprocess_text(self.sample_text)
        self.bigrams = create_bigrams(self.tokens)
        self.bigram_counts = build_bigram_next_token_counts(self.tokens)
        self.bigram_probs = calculate_bigram_next_token_probs(self.bigram_counts)

    # 1. tokenization
    def test_preprocess_text(self):
        expected_tokens = ["to", "be", "or", "not", "to", "be", "that", "is", "the", "question"]
        self.assertEqual(self.tokens, expected_tokens)

    # 2. bigram generation
    def test_create_bigrams(self):
        expected_bigrams = [
            ("to", "be"), ("be", "or"), ("or", "not"), ("not", "to"),
            ("to", "be"), ("be", "that"), ("that", "is"), ("is", "the"), ("the", "question")
        ]
        self.assertEqual(self.bigrams, expected_bigrams)

    # 3. bigram count dictionary
    def test_build_bigram_next_token_counts(self):
        expected_counts = {
            ("to", "be"): {"or": 1, "that": 1},
            ("be", "or"): {"not": 1},
            ("or", "not"): {"to": 1},
            ("not", "to"): {"be": 1},
            ("be", "that"): {"is": 1},
            ("that", "is"): {"the": 1},
            ("is", "the"): {"question": 1}
        }
        self.assertEqual(self.bigram_counts, expected_counts)

    # 4. probability calculations
    def test_calculate_bigram_next_token_probs(self):
        expected_probs = {
            ("to", "be"): {"or": 0.5, "that": 0.5},
            ("be", "or"): {"not": 1.0},
            ("or", "not"): {"to": 1.0},
            ("not", "to"): {"be": 1.0},
            ("be", "that"): {"is": 1.0},
            ("that", "is"): {"the": 1.0},
            ("is", "the"): {"question": 1.0}
        }
        self.assertEqual(self.bigram_probs, expected_probs)

    # 5. sampling function
    def test_sample_next_token(self):
        sampled_token = sample_next_token(("to", "be"), self.bigram_probs)
        self.assertIn(sampled_token, ["or", "that"])

    # 6. text generation
    def test_generate_text_from_bigram(self):
        generated_text = generate_text_from_bigram(("to", "be"), 10, self.bigram_probs)
        self.assertTrue(isinstance(generated_text, str))
        self.assertGreater(len(generated_text.split()), 2)

if __name__ == "__main__":
    unittest.main()
