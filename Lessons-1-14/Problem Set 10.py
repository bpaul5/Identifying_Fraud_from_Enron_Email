### Text Learning 

# bag of words tech
# not effected by order 
# longer phrases do matter 
# can't handle complex phrases 

from sklearn.feature_extraction.text import CountVectorizer 
vectorizer = CountVectorizer()
string1 = 'hi katie the self driving car will be late'
string2 = 'hi sebastian the machine learning class will be great great great'
string3 = 'hi katie the machine learning class will be most excellent'
email_list = [string1, string2, string3]
bag_of_words = vectorizer.fit(email_list)
bag_of_words = vectorizer.transform(email_list)

# (1,5) refers to string 2 word 5 
print bag_of_words

# use to obtain the the corresponding word number 
print vectorizer.vocabulary_.get('great')

# filter out comman words that are not important 
stopwords = ['the', 'in', 'for', 'you', 'will', 'have', 'be']

# natural language tool kit 
# for first time 
import nltk 
nltk.download()

# import stopwords 
from nltk.corpus import stopwords
sw = stopwords.words('english')

# create stemmers (do this before stopwords)
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
stemmer.stem('responsiveness')

#########################################################################################   
# MINI PROJECT 

import os
import pickle
import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### project part 2: comment out the line below
        #words = text_string
        word = text_string.split()
        for i in word:
        	words += stemmer.stem(i) + " "

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        

    return words


#########################################################################################
# vecotrize.py 

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        temp_counter += 1
        #if temp_counter < 151:
        path = os.path.join('..', path[:-1])
        print path
        email = open(path, "r")
        ### use parseOutText to extract the text from the opened email
        email_text = parseOutText(email)

        ### use str.replace() to remove any instances of the words
        ### ["sara", "shackleton", "chris", "germani"]
        replace_words = ['sara', 'shackleton', 'chris', 'germani', 'sshacklensf', 'cgermannsf']
        for i in replace_words:
        	if i in email_text:
        		email_text = email_text.replace(i,'')
        		email_text = ''.join(email_text)

        ### append the text to word_data
        word_data.append(email_text)

        ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
        if name == 'sara':
        	from_data.append(0)
        if name == 'chris':
        	from_data.append(1)

        email.close()

print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )


### in Part 4, do TfIdf vectorization here
vectorizer = TfidfVectorizer(stop_words="english") # add max_df=0.5 as param to remove words that occur 50% 
vectorizer.fit_transform(word_data)
print len(vectorizer.get_feature_names())

vocab_list = vectorizer.get_feature_names()
vocab_list[35787]









    
