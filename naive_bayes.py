'''
a naive bayes approach to the overview of a film
'''
import pickle
import string
from nltk import classify
import numpy as np
import pandas as pd
from collections import defaultdict
## required with first time use: import nltk
## required with first time use: nltk.download('stopwords')
## required with first time use: nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB

class OverviewClassPrediction():
    '''
    Class to specify the naive bayes class for the overview parameter
    '''
    def __init__(self, best=None, dirochlet_alpha=None, num_classes=5):
        '''
        this describes

        Parameters
        ==========
        `best`:
            path to a pre-built class.

        `dirochlet_alpha`:
            weight to give to words missing

        `num_classes`:
            number of classes to predict
        '''
        self.dirochlet_alpha = dirochlet_alpha
        self.class_thresholds = self.create_class_thresholds(num_classes)
        self.labels = [i for i in range(num_classes)]
        self.class_conditionals = None
        self.priors = None
        self.all_words = None
        self.type_classifiers = None
        self.classifier_metrics = None

        # if best:
        #     self.__gather_best(best)
    
    def fit(self, training_data):
        '''
        fit model to training data DataFrame
        '''
        training_data['class'] = training_data.apply(self.make_class, axis=1)
        class_counts = training_data["class"].value_counts()
        print(class_counts)
        self.class_conditionals, self.priors, updated_train, self.all_words = self.train_classifier(
            training_data, class_counts
        )
        self.type_classifiers = {
            "Self": self,
            "Gaussian": GaussianNB(priors=self.priors).fit(updated_train[:, :-1], updated_train[:, -1]),
            "Multinomial": MultinomialNB(class_prior=self.priors).fit(updated_train[:, :-1], updated_train[:, -1]),
            "Complement": ComplementNB(class_prior=self.priors).fit(updated_train[:, :-1], updated_train[:, -1]),
            "Bernoulli": BernoulliNB(class_prior=self.priors).fit(updated_train[:, :-1], updated_train[:, -1]),
        }
        self.classifier_metrics = {
            "Self": dict(),
            "Gaussian": dict(),
            "Multinomial": dict(),
            "Complement": dict(),
            "Bernoulli": dict(),
        }

    def save_model(self, path):
        '''
        save the model
        '''
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, path):
        '''
        retrieve the saved model
        '''
        with open(path, 'rb') as f:
            classifier = pickle.load(f)
        return classifier

    def make_class(self, series_object):
        '''
        method to add a row to the data
        '''
        revenue = series_object['revenue']
        
        for i, threshold in enumerate(self.class_thresholds):
            if revenue < threshold:
                return i
        return len(self.class_thresholds) - 1

    def train_classifier(self, data, class_counts):
        '''
        method to train classifier by a custom model

        Parameters
        ==========
        `data`:
            DataFrame object with the `overview` and `class` columns

        `class_counts`:
            classes and their respective counts
        '''
        num_classes = len(self.labels)
        print(num_classes)
        words_per_class, all_words, word_info = self.get_word_counts(
            num_classes, self.labels, data["overview"].values, movie_labels=data["class"].values)

        # calculate the likelihood of a class across all training data
        priors = np.divide(class_counts.values, len(data))
        # create matrix for classes by words
        class_conditionals = np.zeros((num_classes, len(all_words)))
        # created updated features for sklearn
        updated_features = np.zeros((len(data), len(all_words)+1))

        for movie_idx, word_dict in enumerate(word_info):
            class_idx = data.iloc[movie_idx]['class']

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
        return class_conditionals, priors, updated_features, all_words

    def update_test_data(self, test_data):
        '''
        udpate the test data so it can be used
        '''
        num_classes = len(self.labels)
        class_idx = -1
        _, _, word_info = self.get_word_counts(
            num_classes, self.labels, test_data["overview"].values, train=False)
        updated_features = np.zeros((len(test_data), len(self.all_words)+1))
        for movie_idx, word_dict in enumerate(word_info):
            if "class" in test_data.columns:
                class_idx = test_data.iloc[movie_idx]['class']
            for word in word_dict.keys():
                if word in self.all_words:
                    word_idx = self.all_words[word]
                    updated_features[movie_idx, word_idx] = word_dict[word]
            updated_features[movie_idx, -1] = class_idx

        return updated_features[:, :-1], updated_features[:, -1]

    def predict(self, test_data, model=None):
        '''
        predict the model on test data
        '''
        if model:
            # test_data['class'] = test_data.apply(self.make_class, axis=1)
            test_data, _ = self.update_test_data(test_data)
            return self.type_classifiers[model].predict(test_data)
        log_class_conditionals = np.log(self.class_conditionals)
        log_priors = np.log(self.priors)
        log_posterior = np.dot(test_data, log_class_conditionals.T)
        log_posterior = log_posterior + log_priors

        preds = np.argmax(log_posterior, axis=1) # take arg max for each row
        return preds

    def test_all_models(self, test_data):
        '''
        test all models

        Return
        =========
        the updated classifier metrics dictionary
        '''
        test_data['class'] = test_data.apply(self.make_class, axis=1)
        test_data, actual = self.update_test_data(test_data)
        # actual = np.array([k for k in test_data["class"].values])
        
        # test the different type of classifiers and show the results
        for classifier in self.type_classifiers:
            preds = self.type_classifiers[classifier].predict(test_data)
            self.classifier_metrics[classifier]["predictions"] = preds
            count_correct = np.sum(actual == preds) # smart python way
            self.classifier_metrics[classifier]["accuracy"] = count_correct / len(preds)
            # compare to one away accuracy
            one_away = one_away_count(preds, actual)
            self.classifier_metrics[classifier]["one_away_accuracy"] = one_away / len(preds)
            self.classifier_metrics[classifier]["confusion_matrix"] = confusion_matrix(actual, preds)

        return self.classifier_metrics

    def evaluate_models(self):

        # test the different type of classifiers and show the results
        for classifier in self.classifier_metrics:
            print(f"\nPERFORMANCE METRICS {classifier}")
            preds = self.classifier_metrics[classifier]["predictions"]
            accuracy = self.classifier_metrics[classifier]["accuracy"]
            print(f"Accuracy: {int(accuracy*130)}/{130} {100*accuracy:.2f} %")
            one_away = self.classifier_metrics[classifier]["one_away_accuracy"]
            print(f"One-Away Accuracy:  {100*one_away:.2f} %")
            print("Confusion matrix:")
            print("\n".join(["  ".join([f"{val:4}" for val in row])
                for row in self.classifier_metrics[classifier]["confusion_matrix"]]))

    def get_best(self, metric="one_away_accuracy"):
        '''
        return the predicitions of the best model based on `metric`
        '''
        if len(self.classifier_metrics["Self"]) == 0:
            raise TypeError("Model has not been tested on all models. Cannot compare metrics. Call test_all_models(test_data) to fix.")
        best_metric = -1
        best_model = None
        for classifier in self.classifier_metrics:
            if self.classifier_metrics[classifier][metric] > best_metric:
                best_metric = self.classifier_metrics[classifier][metric]
                best_model = classifier
        
        return self.classifier_metrics[best_model]["predictions"], best_model
        

    @staticmethod
    def get_word_counts(num_classes, class_labels, overviews, movie_labels=None, train=True):
        '''
        return tuple of words per class, and total word count
        '''
        words_per_class = np.zeros(num_classes) # get words per class

        all_words = dict()
        label_counts = [set() for _ in class_labels]

        # initialize the word information
        # meaning, for each movie get count of each word in overview
        word_information = []
        for i, overview in enumerate(overviews):
            # default dict handles missing keys when incremented
            word_information.append(defaultdict(int))
            # good enough for now:
            stripped_punct = overview.translate(str.maketrans("", "", string.punctuation))
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(stripped_punct)
            tokens = [word for word in word_tokens if word not in stop_words]

            for word in tokens:
                if train:
                    label_counts[movie_labels[i]].add(word)
                    # this line will only matter when training
                    # it adds a word to the all_words dictionary and gives teh word an index
                    if word not in all_words:
                        all_words[word] = len(all_words)
                word_information[-1][word] += 1 # increment the count for the movie
        if train:
            for i, label in enumerate(label_counts):
                words_per_class[i] = len(label)

        return words_per_class, all_words, word_information

    @staticmethod
    def create_class_thresholds(num_classes):
        basis_threshold = 500 / num_classes
        return [basis_threshold * 2**n * 10**6 for n in range(num_classes)]

def one_away_count(preds, actual):
    ''''''
    return len(preds[np.where(np.abs(actual-preds) <=1)])

def main():
    data = pd.read_csv("dbs/data_2010s.csv")
    overviews = pd.read_csv("dbs/movies_from_2010s.csv")[['tmdb_id', 'overview']]
    data = data.merge(overviews, on="tmdb_id")
    train, test = train_test_split(data, test_size=0.2, random_state=18)
    classifier = OverviewClassPrediction()
    classifier.fit(train)
    # print(classifier.class_thresholds)
    classifier.test_all_models(test)
    classifier.evaluate_models()
    classifier.save_model("pickled-naive_bayes.pickle")

def o():
    data = pd.read_csv("dbs/data_2010s.csv")
    overviews = pd.read_csv("dbs/movies_from_2010s.csv")[['tmdb_id', 'overview']]
    data = data.merge(overviews, on="tmdb_id")
    train, test = train_test_split(data, test_size=0.2, random_state=18)
    classifier = OverviewClassPrediction().load_model("pickled-naive_bayes.pickle")
    print(classifier.predict(test, model="Bernoulli"))
    
if __name__ == "__main__":
    main()
    # o()
