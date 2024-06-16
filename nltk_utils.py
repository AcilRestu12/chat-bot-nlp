import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
stemmer = PorterStemmer()

def tokenize(setence):
    return nltk.word_tokenize(setence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_setence, all_words):
    pass


# Example of Tokenize
# a = 'Hi there, what can I do for you?'
# print(f'Before \t: {a}')

# a = tokenize(a)
# print(f'After \t: {a}')

# Example of Stemming
words = ['organize', 'organizes', 'organizing']
stemmed_words = [stem(w) for w in words]
print(stemmed_words)

