
# Lets build up a sentense and remove its stop words.

from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
customStopWords = set(stopwords.words('english') + list(punctuation))
text='Eli is a little keves, he also likes eating keves..! So.. Lets remove some stop words..!!!'
textWithoutStopWords = [word for word in word_tokenize(text) if word not in customStopWords]

# Now lets consider Bi-Grams, and detect consecutive words that are usually related / come together
# as Bi-Grams
import nltk
# I will use the Collocation library, which helps find relationship between words, words that are
# collocated next to each other in high propebility
# So we get bi-grams, from a list of words
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(textWithoutStopWords)

# The sorting helps us to sort the bi-grams according to their frequency
for bigram in sorted(finder.ngram_fd.items()):
    print(bigram)

print('-------------------------')
# We can also do tri-grams using nltk.collocations.TrigramAssocMeasures
trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder = TrigramCollocationFinder.from_words(textWithoutStopWords)

# The sorting helps us to sort the bi-grams according to their frequency
for trigram in sorted(finder.ngram_fd.items()):
    print(trigram)