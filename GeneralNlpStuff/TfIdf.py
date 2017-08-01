def getEmmaChapter():
    from nltk.text import TextCollection
    # from nltk.text import *
    import nltk
    # from nltk.book import text1, text2, text3
    gutenberg = TextCollection(nltk.corpus.gutenberg)

    # ----- IDF EXAMPLE ----
    # print(gutenberg.idf('Dick'))
    # ----- IDF EXAMPLE ----

    i = 2
    # line 2 to line 166 is chapter 1
    emma = nltk.corpus.gutenberg.sents('austen-emma.txt')
    # for l in emma:
    chapterText = ''
    while i < 167:
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

    print (chapterText)
    return
    # print(emma)



getEmmaChapter()