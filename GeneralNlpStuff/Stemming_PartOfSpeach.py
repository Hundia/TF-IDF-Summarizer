from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
st=LancasterStemmer()

text='Eli is a little keves, he also likes eating keves..!'

stemmedWords=[st.stem(word) for word in word_tokenize(text)]
print(stemmedWords)

# Now to the part of speech
pos_res = pos_tag(word_tokenize(text))
for e in pos_res:
    print(e)
