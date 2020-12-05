'''
a naive bayes approach to the overview of a film
'''
import pandas as pd
import numpy as np
import string
from collections import defaultdict
import nltk
## required with first time use nltk.download('stopwords')
## required with first time usenltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix

def main():
    '''
    '''
    data = pd.read_csv("dbs/data_2010s.csv")
    overviews = pd.read_csv("dbs/movies_from_2010s.csv")[['tmdb_id', 'overview']]
    data = data.merge(overviews, on="tmdb_id")
    data["class"] = data.apply(make_class, axis=1)

    train, test = train_test_split(data, test_size=0.2, random_state=18)
    classes = train["class"].value_counts() # class name and count of samples in that class

    class_conditionals, priors, updated_train, all_words, class_map = train_classifier(train, classes)
    preds, updated_test = test_classifier(test, classes, np.log(class_conditionals), np.log(priors), all_words, class_map)

    # Compute performance metrics
    print(len(test["class"].values),len(preds))
    actual = np.array([class_map[k] for k in test["class"].values])
    print("PERFORMANCE METRICS HAND BUILT")
    count_correct = np.sum(actual == preds) # smart python way
    accuracy = count_correct / len(test)
    print(f"Accuracy: {count_correct}/{len(test)} ({100*accuracy:.2f} %)")
    one_away = one_away_count(preds, actual)
    accuracy = one_away / len(test)
    print(f"One-Away Accuracy: {one_away}/{len(test)} ({100*accuracy:.2f} %)")
    cm = confusion_matrix(actual, preds)
    print("Confusion matrix:")
    print("\n".join(["  ".join([f"{val:2}" for val in row]) for row in cm]))

    type_classifiers = [
        ("Gaussian", GaussianNB(priors=priors)),
        ("Multinomial", MultinomialNB(class_prior=priors)),
        ("Complement", ComplementNB(class_prior=priors)),
        ("Bernoulli", BernoulliNB(class_prior=priors)),
        ]
    # test the different type of classifiers and show the results
    for name, classifier in type_classifiers:
        print(f"\nPERFORMANCE METRICS SKLEARN {name}")
        classifier.fit(updated_train[:, :-1], updated_train[:, -1])
        preds = classifier.predict(updated_test[:, :-1])
        count_correct = np.sum(actual == preds) # smart python way
        accuracy = count_correct / len(test)
        print(f"Accuracy: {count_correct}/{len(test)} ({100*accuracy:.2f} %)")
        # compare to one away accuracy
        one_away = one_away_count(preds, actual)
        accuracy = one_away / len(test)
        print(f"One-Away Accuracy: {one_away}/{len(test)} ({100*accuracy:.2f} %)")
        cm = confusion_matrix(actual, preds)
        print("Confusion matrix:")
        print("\n".join(["  ".join([f"{val:2}" for val in row]) for row in cm]))

def one_away_count(preds, actual):
    ''''''
    return len(preds[np.where(np.abs(actual-preds) <=1)])

def test_classifier(data, classes, log_class_conditionals, log_priors, all_words, class_map):
    num_classes = len(classes)
    print("Counting words in each document...")
    _, _, word_info = get_word_counts(
        num_classes, classes.index, data["overview"].values, data["class"].values)
    updated_features = np.zeros((len(data), len(all_words)+1))
    for movie_idx, word_dict in enumerate(word_info):
        class_idx = class_map[data.iloc[movie_idx]['class']]
        for word in word_dict.keys():
            if word in all_words:
                word_idx = all_words[word]
                updated_features[movie_idx, word_idx] = word_dict[word]
        updated_features[movie_idx, -1] = class_idx

    log_posterior = np.dot(updated_features[:, :-1], log_class_conditionals.T) # <3 matrix multiplication
    log_posterior = log_posterior + log_priors # <3 logarithms

    preds = np.argmax(log_posterior, axis=1) # take arg max for each row
    return preds, updated_features

def train_classifier(data, classes):
    num_classes = len(classes)
    print(data.index)
    words_per_class, all_words, word_info = get_word_counts(
        num_classes, classes.index, data["overview"].values, data["class"].values)
    print(words_per_class)
    print(len(all_words))
    # calculate the likelihood of a class across all training data
    priors = np.divide(classes.values, len(data))
    class_conditionals = np.zeros((num_classes, len(all_words))) # create matrix for classes by words

    # identify mapping from class to index
    class_map = dict()
    for i, class_name in enumerate(classes.index):
        class_map[class_name] = i
    seen = set()
    updated_features = np.zeros((len(data), len(all_words)+1))
    for movie_idx, word_dict in enumerate(word_info):
        class_idx = class_map[data.iloc[movie_idx]['class']]
        if class_idx not in seen:
            print(class_idx)
            seen.add(class_idx)
        for word in word_dict.keys():
            word_idx = all_words[word]
            class_conditionals[class_idx, word_idx] += word_dict[word]
            updated_features[movie_idx, word_idx] = word_dict[word]
        updated_features[movie_idx, -1] = class_idx

    # add dirochlet
    dirochlet_alpha = 1 / len(all_words)
    class_conditionals += dirochlet_alpha
    words_per_class += 1
    # divide a matrix by vector value with same number of rows (avoid loop):
    class_conditionals = np.divide(class_conditionals, words_per_class[:, None])
    return class_conditionals, priors, updated_features, all_words, class_map

def get_word_counts(num_classes, class_names, overviews, movie_classes):
    '''
    return tuple of words per class, and total word count
    '''
    words_per_class = np.zeros(num_classes) # get words per class
    class_counts = dict()
    
    all_words = dict()
    for k in class_names:
        class_counts[k] = set()
    
    # initialize the word information, specifying movie, word, and count_per_document
    word_information = []
    for i, overview in enumerate(overviews):

        word_information.append(defaultdict(int))
        # set_words = dict_class_counts[movie_class[i]]
        # good enough for now:
        stripped_punct = overview.translate(str.maketrans("", "", string.punctuation))
        stop_words = set(stopwords.words('english')) 
        word_tokens = word_tokenize(stripped_punct)
        tokens = [word for word in word_tokens if word not in stop_words]
        for word in tokens:
            class_counts[movie_classes[i]].add(word)
            if word not in all_words:
                all_words[word] = len(all_words)
            word_information[-1][word] += 1
            # break

    for i, class_name in enumerate(class_counts.keys()):
        words_per_class[i] = len(class_counts[class_name])

    return words_per_class, all_words, word_information

def make_class(series_object):
    revenue = series_object['revenue']

    if revenue < 100*10**6:
        return "Low"
    elif revenue < 200*10**6:
        return "Low-Mid"
    elif revenue < 400*10**6:
        return "Mid"
    elif revenue < 800*10**6:
        return "Mid-High"
    else:
        return "High"

if __name__ == "__main__":
    main()