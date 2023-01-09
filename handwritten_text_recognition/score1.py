import string
import re
from math import *
import time
import spacy
import en_core_web_lg
from rake_nltk import Rake
import pytextrank
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")
#nlp = spacy.load('en_core_web_sm')

lemmatizer = WordNetLemmatizer()
stopw = set(stopwords.words('english'))
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
sp = en_core_web_lg.load()
r = Rake()
sp.add_pipe("textrank")
model1 = KeyedVectors.load_word2vec_format("/content/drive/MyDrive/GoogleNews-vectors-negative300.bin.gz", binary=True)

contraction = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
    "he'll've": "he he will have", "he's": "he is", "how'd": "how did",
    "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
    "I'll've": "I will have", "I'm": "I am", "I've": "I have",
    "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
    "i'll've": "i will have", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
    "it'll": "it will", "it'll've": "it will have", "it's": "it is",
    "let's": "let us", "ma'am": "madam", "mayn't": "may not",
    "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
    "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
    "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
    "she's": "she is", "should've": "should have", "shouldn't": "should not",
    "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
    "this's": "this is",
    "that'd": "that would", "that'd've": "that would have", "that's": "that is",
    "there'd": "there would", "there'd've": "there would have", "there's": "there is",
    "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
    "they'll've": "they will have", "they're": "they are", "they've": "they have",
    "to've": "to have", "wasn't": "was not", "we'd": "we would",
    "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
    "we're": "we are", "we've": "we have", "weren't": "were not",
    "what'll": "what will", "what'll've": "what will have", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is",
    "when've": "when have", "where'd": "where did", "where's": "where is",
    "where've": "where have", "who'll": "who will", "who'll've": "who will have",
    "who's": "who is", "who've": "who have", "why's": "why is",
    "why've": "why have", "will've": "will have", "won't": "will not",
    "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
    "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
    "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
    "you'll've": "you will have", "you're": "you are", "you've": "you have"}


def clean(text):
    text = text.lower()
    temp = ""
    for i in text.split():
        try:
            temp += contraction[i]+' '
        except:
            temp += i+' '
    text = temp.strip()
    text = text.lower().translate(remove_punctuation_map)
    text = re.sub("[^a-zA-Z#]", " ", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r",", "", text)
    text = re.sub(r"\.", "", text)
    text = re.sub(r"!", "!", text)
    text = re.sub(r"\/", "", text)
    text = re.sub(r"'", "", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", ":", text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def stopwordremoval(text):
    text = word_tokenize(text)
    text = [i for i in text if i not in stopw]
    return " ".join(text)

def pp_set(text, op):
    key_tokenized_sentences = sent_tokenize(text)
    key_tokenized_words = word_tokenize(text)
    if op == "token_sent":
        return key_tokenized_sentences
    elif op == "token_word":
        return key_tokenized_words
    elif op == "clean_sent":
        return [clean(i) for i in key_tokenized_sentences]
    elif op == "clean_word":
        return [clean(i) for i in key_tokenized_words]
    elif op == "lem_sent":
        key_clean_sentences = pp_set(text, "clean_sent")
        return [" ".join([lemmatizer.lemmatize(j) for j in i.split()]) for i in key_clean_sentences]
    elif op == "lem_word":
        key_clean_words = pp_set(text, "clean_word")
        return [lemmatizer.lemmatize(i) for i in key_clean_words]
    elif op == "prep_sent":
        key_clean_sentences = pp_set(text, "clean_sent")
        return [" ".join([i for i in j.split() if i not in stopw]) for j in key_clean_sentences]
    elif op == "prep_word":
        key_preprocessed_sentences = pp_set(text, "prep_sent")
        key_preprocessed_words = []
        for i in key_preprocessed_sentences:
            key_preprocessed_words.extend(word_tokenize(i))
        return key_preprocessed_words
    elif op == "pp_lem_word":
        return [lemmatizer.lemmatize(i) for i in pp_set(text, "prep_word")]

def avg_sentence_vector(words, model, num_features, index2word_set):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])

    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec.reshape(1, -1)

def semantic_sim(key1, key2):
    print("key1:", key1)
    print("key2:", key2)
    try:
        sim = model.wv.n_similarity(key1, key2)
    except:
        vec1 = avg_sentence_vector(
            pp_set(key1, "pp_lem_word"), model1, 300, model.index2word)
        vec2 = avg_sentence_vector(
            pp_set(key2, "pp_lem_word"), model1, 300, model.index2word)
        sim = cosine_similarity(vec1, vec2)[0][0]
    finally:
        return sim

def extract_keywords(text):
    # Effectiveness : tokenized > lemmatized > clean
    r.extract_keywords_from_sentences(pp_set(text, "lem_sent"))
    rake_keywords = r.get_ranked_phrases()
    spdoc = sp(text)
    ner_keywords = []
    for ent in spdoc.ents:
        ner_keywords.append(ent.text)
    spdoc = sp(" ".join(pp_set(text, "clean_word")))
    pytr_keywords = []
    for p in spdoc._.phrases:
        for term in p.chunks:
            if term.text not in pytr_keywords and term.text not in stopw:
                x = term.text
                pytr_keywords.append(x)

    all_keywords = rake_keywords+pytr_keywords+ner_keywords
    all_keywords = list(set(all_keywords))
    sorted_keywords = list(all_keywords)
    sorted_keywords.sort()
    for i in range(len(sorted_keywords)):
        sorted_keywords[i] = re.sub(r' +', ' ', sorted_keywords[i])

    return sorted_keywords

def group(sorted_keywords):
    grouped_keys = []
    for i in sorted_keywords:
        if len(grouped_keys) == 0:
            grouped_keys.append([i])
            continue
        else:
            flag = False
            for j in grouped_keys:
                if i in j:
                    flag = True
                    break
                temp1 = " ".join([lemmatizer.lemmatize(t)
                                  for t in stopwordremoval(i).split()])
                for k in j:
                    temp2 = " ".join([lemmatizer.lemmatize(t)
                                      for t in stopwordremoval(k).split()])
                    short = min(temp1, temp2)
                    long = max(temp1, temp2)
                    if short in long:
                        flag = True
                        j.append(i)
                        break
                if flag == True:
                    break
            if flag == False:
                grouped_keys.append([i])
    temp = []
    for i in grouped_keys:
        k = sorted(i, key=len)
        temp.append(k)
    return temp

def remove_duplicates(grouped_keys):
    for i in range(len(grouped_keys)):
        grouped_keys[i] = list(set(grouped_keys[i]))
        temp = list(grouped_keys[i])
        process_set = [" ".join([lemmatizer.lemmatize(
            l) for l in stopwordremoval(j).split()]) for j in grouped_keys[i]]
        process_set = list(set(process_set))
        for temp_key1 in grouped_keys[i]:
            x = " ".join([lemmatizer.lemmatize(k)
                          for k in stopwordremoval(temp_key1).split()])
            if process_set.count(x) > 1:
                temp.remove(temp_key1)
        grouped_keys[i] = temp
        grouped_keys[i] = sorted(grouped_keys[i])

    for i in range(len(grouped_keys)):
        temp = list(grouped_keys[i])
        for j in range(len(grouped_keys[i])):
            word = grouped_keys[i][j]
            for k in temp:
                if word in k and word != k:
                    temp.remove(word)
                    break
        grouped_keys[i] = sorted(temp, key=len, reverse=True)
    grouped_keys = [i for i in grouped_keys if len(i) > 0]
    return grouped_keys

def finalize(grouped_keys):
    temp_keywords = []
    final_keywords = []
    for i in grouped_keys:
        for j in i:
            temp_keywords.append(j)

    temp_keywords = remove_duplicates(group(temp_keywords))

    for i in temp_keywords:
        for j in i:
            final_keywords.append(j)
    return final_keywords

def dictionarize(final_keywords, text):
    answer_key = dict()
    sentences = pp_set(text, "token_sent")
    for i in sentences:
        answer_key[i] = list()
    temp = list(final_keywords)
    for i in range(len(temp)):
        key = " ".join(pp_set(temp[i], "token_word"))
        for j in answer_key:
            x = j.strip().lower()
            if key in x:
                answer_key[j].append(key)
                final_keywords.remove(temp[i])
                break
    return answer_key 

def vectorize_text(answer_key):
    vector_keys = []
    vector_sent = []
    for i in list(answer_key.keys()):
        vector_sent.append(avg_sentence_vector(
            pp_set(i, "token_word"), model1, 300, model1.index2word))
        temp = []
        for j in list(answer_key[i]):
            temp.append(avg_sentence_vector(
                pp_set(j, "token_word"), model1, 300, model1.index2word))
        vector_keys.append(temp)

    return vector_sent, vector_keys

def score(key, test):
    vec_key_sent, vec_key_keys = vectorize_text(key)
    vec_test_sent, vec_test_keys = vectorize_text(test)
    sum = 0
    sims = dict()
    for i in range(len(vec_test_sent)):
        sims[i] = []
        for j in range(len(vec_key_sent)):
            sim = cosine_similarity(vec_test_sent[i].reshape(
                1, -1), vec_key_sent[j].reshape(1, -1))
            if sim > 0.7:
                sims[i].append(j)

    count = 0
    for keyidx in sims:
        ans_kw = vec_test_keys[keyidx]
        key_kw = []
        checked = []
        for i in sims[keyidx]:
            key_kw.extend(vec_key_keys[i])

        for akw in ans_kw:
            max_sim = -1
            max_kkw = []
            for kkw in key_kw:
                if kkw in checked:
                    continue
                sim = cosine_similarity(kkw, akw)[0][0]
                if sim > max_sim:
                    max_sim = sim
                    max_akw = kkw
            if sim > 0.9:
                sum += 1
#                 count += 1
            else:
                sum += max_sim
#                 print(max_sim)
#                 count += 1
            checked.append(max_kkw)
    return sum, count

    