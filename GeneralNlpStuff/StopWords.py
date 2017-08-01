# Removing stop words
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
customStopWords = set(stopwords.words('english') + list(punctuation))

# English known stop words
# print(customStopWords)

text='Eli is a little keves, he also likes eating keves..! So.. Lets remove some stop words..!!!'
print('Text with stop words:')
print(text)
textWithoutStopWords = [word for word in word_tokenize(text) if word not in customStopWords]
print('Text without stop words: ')
newText=''
for word in textWithoutStopWords:
    newText += word + ' '
print(newText)