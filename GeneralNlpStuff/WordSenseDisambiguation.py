# Well use wordnet as our lexicon (thesaurus)
# Wordnet comes referenced in nltk corpus
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
# Lets get the synset for the word 'shoot'
for ss in wn.synsets('pool'):
    print(ss, ss.definition())

print('-----------------------------------------------------------------')
# We also get an algorithm for word sense disambiguation called lesk
# Given a sentense, you can give the algorithm a word in the sentense and it will
# disambiguate its meaning
from nltk.wsd import lesk
sense1 = lesk(word_tokenize("Pool also refers to a game where you try to put the colored and numbered balls into the holes around the edges of the table"),'pool')
print(sense1, sense1.definition())

sense1 = lesk(word_tokenize("The pool was a communal combination of funds"),'pool')
print(sense1, sense1.definition())