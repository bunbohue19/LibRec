from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# Initialize the Tokenizer with a WordLevel model
tokenizer = Tokenizer(models.WordLevel())

# Train the Tokenizer
trainer = trainers.WordLevelTrainer(
    vocab_size=40000, 
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)
tokenizer.train([
    "/users/anhld/LibRec/test.txt",
    # "./path/to/dataset/2.txt",
    # "./path/to/dataset/3.txt"
], trainer=trainer)

# Save the trained tokenizer
tokenizer.save("word-level.tokenizer.json", pretty=True)

from tokenizers import Tokenizer

# Load the tokenizer
tokenizer = Tokenizer.from_file("word-level.tokenizer.json")

# Encode text using the tokenizer
encoded = tokenizer.encode("I can feel the magic, can you?")
print(encoded.tokens)

