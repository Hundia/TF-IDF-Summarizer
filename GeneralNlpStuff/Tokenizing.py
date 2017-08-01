import nltk

text="Mary had a little lamb. Her fleece was white as snow"

from nltk.tokenize import word_tokenize, sent_tokenize

# Sentence tokenizer, seperates to sentences seperated by dot - '.'
sentenses = sent_tokenize(text)
print(sentenses)

# Words tokenizer, seperates to sentences seperated by dot - '.'
wordTokens = word_tokenize(text)
print(wordTokens)

# Tokenize words in each sentence tokenized into list of words in each sentence.
words = [word_tokenize(sent) for sent in sentenses]
# print(words)


