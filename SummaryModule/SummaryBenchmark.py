# Url lib will help us get articles from the web
from urllib.request import Request,urlopen
from nltk.text import TextCollection

# from nltk.text import *
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from collections import defaultdict
# bs4 will help us parse articles from theire HTML tempalate
from bs4 import BeautifulSoup
from nltk.probability import FreqDist
import sys
import re

minIdfVal = 110000
maxIdfVal = 0

def reade_from_file(out_path):
    mapped_obj = dict()
    min = 1000
    max = 0


    with open(out_path, mode='r') as infile:
        content = infile.readlines()
        for element in content:
            key = re.split(':-:', element)[0]
            val = re.split(':-:', element)[1]
            # print('key: ' + key + ' val: ' + str(val));
            val = float(val.rstrip())
            mapped_obj[key] = val
            if(val < min):
                min = val
            if(val > max):
                max = val

    # print(mapped_obj)
    print('Min IDF Val: ' + str(min) + ' Max IDF Val: ' + str(max))
    # Update our vals
    minIdfVal = min
    maxIdfVal = max
    return mapped_obj


# Now lets define a normalizing function based on standard normalizing function
# We will use this normlizing function to normalize TF values into IDF values
def normalizeToIdf(numberToNormalize, min, max):
    return (numberToNormalize - min) / (max - min)

emma_words = nltk.corpus.gutenberg.words('austen-emma.txt')
print('Words with duplicates count len: ' + str(len(emma_words)))
setlst = set(emma_words);
print('Words without duplicates count len: ' + str(len(setlst)))
import datetime as dt

import random
idfDict = defaultdict(int)
#  IDF Mapping of all gutenberg for emma
idfOfEmma = reade_from_file('emma_idf.txt')

for w in setlst:
    n1 = dt.datetime.now()
    # Original IDF calculation, takes 3 sec for each word
    # idfWordRankings[w] = gutenberg.idf(w)

    # our pre-built IDF, takes less then a mili sec to build per word
    idfDict[w] = idfOfEmma[w]

    # print(w + ': ' + str(idfWordRankings[w]) + ' it took: ' + str((n2 - n1).seconds) + ' seconds')


gutenberg = TextCollection(nltk.corpus.gutenberg)


def getTfIdfVector(text):
    idfWordRankings = defaultdict()

    words = word_tokenize(text.lower())

    # Get all known stop words in corpus in english from the punctuation
    englishStopWords = set(stopwords.words('english') + list(punctuation))

    # Create our words set without the stop words
    words_set = [word for word in words if word not in englishStopWords]

    words = set(words)
    freq = FreqDist(words_set)
    for w in words_set:
        # idf = gutenberg.idf(w) -- Return original
        tfidf = idfDict[w]*freq[w]
        # print(w + ': ' + str(idf))
        idfWordRankings[w] = tfidf;

    keys = idfWordRankings.keys()
    # for word in keys:
    #     print('key: ' + word + ' val: ' + str(idfWordRankings[word]))

    return idfWordRankings

# -- Uncomment to print out the entire article
# print(artText)
def Summarizer(artText, numOfLines, A, B):
    # Text preprocessing for summarization

    # So we want to tokenize our sentenses
    # We want to remove Stop words since they dont help us grade the importance of a sentence
    # IMPORTANT: Pay attention that every line is expected to end by a '.' dot and a following ' ' space.
    sents = sent_tokenize(artText)



    # print('------------------')
    # print('IDF Vector for original text: ')
    vecIdfOrig = getTfIdfVector(artText.lower())
    sortedVecOrig = sorted(vecIdfOrig.items(), key=lambda x: x[1], reverse=True)
    # print(sortedVecOrig);
    # print(vecIdfOrig)
    # print('------------------')
    # -- Uncomment to print out tokenizes lines
    # i = 0;
    # for l in sents:
    #     i = i+1
    #     str2print = str(i)+ ' ' + l + '\n'
    #     print(str2print)

    # return
    # Lets get the words in the article, removing stop words inorder to get words importance ratings
    words = word_tokenize(artText.lower())
    N = len(words)
    # Get all known stop words in corpus in english from the punctuation
    englishStopWords = set(stopwords.words('english') + list(punctuation))
    # Create our words set without the stop words
    words_set = [word for word in words if word not in englishStopWords]

    # -- Uncomment to print out tokenizes words from the article without stop words
    # print(words_set)

    # Now lets start the summarization process
    # First, we want to know the frequency distribution of each word in the article
    # nltk Freq Dist library helps us figure out just that :-)

    freq = FreqDist(words_set)

    # ----- Calculate the min max of freq dist for normalization to 0 - 1
    from heapq import nlargest,nsmallest
    maxFreqValTmp = nlargest(1,freq, key=freq.get)
    minFreqValTmp = nsmallest(1, freq, key=freq.get)

    # print('MAX freq val: ' + freq[maxFreqValTmp[0]])
    # print('MIN freq val: ' + freq[minFreqValTmp[0]])

    minFreqVal = freq[minFreqValTmp[0]]
    maxFreqVal = freq[maxFreqValTmp[0]]

    # We can use nlargest to get N heighest ranking words, or we can just use most_common method in FreqDist class
    # -- Most common example
    # for f in freq.most_common(10):
    #     print(f[0] + ' ' + str(f[1]))



    # NLargest example, using

    # res = nlargest(10,freq, key=freq.get)
    # print(res)


    # defaultdict in python is different than the regular dictionary since if
    # you look up a key in it, and it doesnt exist, it add's it to the dictionary

    # Now lets create a dictionary of sentences as key's and score's as their values
    sentRankings = defaultdict(int)
    sentRankingsNoIdf = defaultdict(int)
    # Check out how to enumerate sentences
    # for i in enumerate(sents):
    #     print(i)

    # i == enumerated index of sentence, sent == sentence itself


    # Golden standard A and B, A set the relevance we give to TF weight, and B gives the relevance we give to the IDF Weight

    TOP_TO_CONSIDER = 20

    minNorVal = min(minFreqVal,minIdfVal)
    maxNorVal = max(maxFreqVal,maxIdfVal)

    for index,sentence in enumerate(sents):
            # Tokenize each sentence into words
            for word in word_tokenize(sentence.lower()):
                # Ask the word frequencies
                if word in freq:
                    # Calculate our A and B while normalizing the values to ranges from 0 to 1, for TF and for TF-IDF
                    from math import log
                    sentRankings[index] += normalizeToIdf(freq[word], minNorVal, maxNorVal)*A + normalizeToIdf(idfDict[word]*freq[word], minNorVal, maxNorVal)*B # Here we will insert our top secret formula
                    # if(0 == index):
                    #     print("word: " + word + " Freq: " + str(normalizeToIdf(freq[word], minFreqVal, maxFreqVal)) + " idf: " + str(normalizeToIdf(idfDict[word]*10, minIdfVal, maxIdfVal)))
                    sentRankingsNoIdf[index] += freq[word]


    # Print out the scores dicitonary - optional
    # for y in sentRankings:
    #         print (y,':',sentRankings[y])




    bestSentIndexes = nlargest(numOfLines, sentRankings, key=sentRankings.get)

    # print("Best ranking sentences indexes are: " + str(bestSentIndexes))

    bestSentIndexesNoIdf = nlargest(numOfLines, sentRankingsNoIdf, key=sentRankingsNoIdf.get)

    # print("Best ranking sentences indexes are: " + str(bestSentIndexesNoIdf))
    # print("The summary: ")

    summaryText = ''
    for i in bestSentIndexes:
        summaryText += sents[i]
        summaryText += '\n'

    # print(summaryText)
#    print('------------------')
#     print('IDF Vector for Summary text: ')
    sumIdfVec = getTfIdfVector(summaryText.lower())
    sortedVec = sorted(sumIdfVec.items(), key=lambda x:x[1], reverse=True)
    # print(sortedVec);
    # print(sumIdfVec);
    # sortedSum = sorted(sumIdfVec.items(), key=lambda x: x[1])
    #
    # for w in sortedSum:
    #     print(w + ' SumIDF:  ' + str(sumIdfVec[w]) + ' OrigIDF: ' + str(vecIdfOrig[w]))
    # print('------------------')

    # print('Lets count how many words were in the original IDF vector top 10 that arent in the new top 10 summary IDF vector')

    i = 0
    j = 0
    sumOfDiff = 0


    while i < 10:
        idfWord = sortedVecOrig[i][0]
        flagForFindingIdf = False # initialize flag with not found, check it after iterating to see if its true, meaning we found it in top 10
        j = 0
        while j < 10:
            # print('Orig word: ' + str(sortedVec[j][0]) + ' idfWord: ' + str(idfWord) + ' found in position: ' + str(j))

            if(sortedVec[j][0] == idfWord):
                # print('FOUND .. !! -- >Orig word: ' + str(sortedVec[j][0]) + ' idfWord: '  + str(idfWord) + ' found in position: ' + str(j))
                flagForFindingIdf = True

            j = j + 1

        i = i + 1

        if(False == flagForFindingIdf):
            sumOfDiff += 1

    # print('total diff: ' + str(sumOfDiff))


    return sumOfDiff

def getEmmaChapter():

    i = 3
    # line 2 to line 166 is chapter 1
    emma = nltk.corpus.gutenberg.sents('austen-emma.txt')
    # for l in emma:
    chapterText = ''
    while i < 166:
        # print(str(i) + ': ')
        k = 0
        l = emma[i]
        line = ''
        for w in l:
            line += l[k] + ' '
            k = k + 1
        # print(str(i) + ': ' + line + '\n')
        chapterText += line + '\n'
        i = i + 1

    # print (chapterText)
    return chapterText
    # print(emma)


def runGUI():
    global app
    from appJar import gui
    # function called by pressing the buttons
    def press(btn):
        if btn == "Exit":
            app.stop()
        if btn == "Summarize":
            sum = Summarizer(app.getTextArea("text_to_summarize"), int(app.getEntry('lines')))
            app.clearTextArea("summarized")
            app.setTextArea("summarized", sum)

    app = gui("Cool TF-IDF Text summarize", "800x800")
    app.addLabel("title", "Welcome to Eli and Roman summarizer app", 0, 0, 2)  # Row 0,Column 0,Span 2
    app.addLabel("lines", "Number Of lines for summary:", 1, 0)  # Row 1,Column 0
    app.addEntry("lines", 2, 0)  # Row 1,Column 1
    app.addLabel("user", "Text to summarize paste here:", 3, 0)  # Row 1,Column 0
    app.addScrolledTextArea("text_to_summarize", 4, 0)  # Row 2,Column 0
    app.addButtons(["Summarize"], press, 5, 0, 1)  # Row 3,Column 0,Span 2
    app.addScrolledTextArea("summarized", 6, 0)  # Row 2,Column 0
    app.addButtons(["Exit"], press, 7, 0, 1)  # Row 3,Column 0,Span 2
    # app.setEntryFocus("text_to_summarize")
    app.setLabelBg("user", "blue")
    app.setLabelBg("title", "green")
    app.setBg("Brown")
    app.setFont(20)
    # Color up some stuff
    app.setEntry('lines', 4);
    app.setEntryMaxLength('lines', 2)
    app.go()


# runGUI()

print('Running Benchmark for A - B - Number of lines ')
print('We will start with A = 0, B = 1, Number of lines = 5')
print('We define distance between two IDF vectors, by the number of elements that are not in the top X original IDF vector');
txt2parse = getEmmaChapter()
A = 1
B = 9
diffRes = 0
amountOfLinesInSummary = 10

Aa = 0

LinesInSumToTest = 10

labels = ""
Data = ""
while Aa < 10:
    #   Prepare the A and B as completion to 1
    A = 10 - Aa
    B = 10 - A
    LinesInSumToTest = 5

    while LinesInSumToTest < 100:
        diffRes = Summarizer(txt2parse, LinesInSumToTest, A, B)
        labels += '"A: '+ str(A) + ' B: ' + str(B) + ' #Lines: ' + str(LinesInSumToTest) +'" ,'
        Data += str(diffRes) + ', '
        print('A: '+ str(A) + ' B: ' + str(B) + ' LinesInSumToTest: ' + str(LinesInSumToTest) + ' diff: ' + str(diffRes))
        LinesInSumToTest += 1

    Aa = Aa + 1

print(labels)
print(Data)

