import os
from collections import defaultdict
from nltk import RegexpTokenizer
import pickle
import math


class NaiveBayes:
    def __init__(
        self, labels: list[str], path_to_data: str, psuedocount: float = 1
    ) -> None:
        self.vocab = set()
        self.path_to_data = path_to_data
        self.labels = labels
        self.train_dir = os.path.join(path_to_data, "train")
        self.test_dir = os.path.join(path_to_data, "test")

        self.psuedocount = psuedocount

        self.label_total_doc_counts = {}
        for label in self.labels:
            self.label_total_doc_counts[label] = 0.0

        self.label_word_counts = {}
        for label in self.labels:
            self.label_word_counts[label] = defaultdict(float)

        self.label_total_word_counts = {}
        for label in self.labels:
            self.label_total_word_counts[label] = 0.0

    def p_word_given_label_alpha(self, word, label):
        word_count = self.label_word_counts[label][word] + self.psuedocount
        total_word_count = self.label_total_word_counts[label] + self.psuedocount * len(
            self.label_word_counts[label]
        )

        return word_count / total_word_count

    def log_prior(self, label):
        return self.label_total_doc_counts[label] / sum(
            self.label_total_doc_counts.values()
        )

    def log_likelihood(self, bow, label):
        return sum(
            [math.log(self.p_word_given_label_alpha(word, label)) for word in bow]
        )

    def unnormalized_log_posterior(self, bow, label):
        return self.log_prior(label) + self.log_likelihood(bow, label)

    def classify(self, bow):
        predictions = dict()
        for label in self.labels:
            predictions[label] = self.unnormalized_log_posterior(bow, label)

        return max(predictions, key=predictions.get)

    def report_stats(self):
        print("REPORTING CORPUS STATISTICS")
        for label in self.labels:
            print(
                "NUMBER OF DOCUMENTS IN {} CLASS:".format(label),
                self.label_total_doc_counts[label],
            )
            print(
                "NUMBER OF TOKENS IN {} CLASS:".format(label),
                self.label_total_word_counts[label],
            )

        print(
            "VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:",
            len(self.vocab),
        )

    def pickle_data(self, pickle_destination: str):
        training_data = {
            "labels": self.labels,
            "vocab": self.vocab,
            "label_total_doc_counts": self.label_total_doc_counts,
            "label_word_counts": self.label_word_counts,
            "label_total_word_counts": self.label_total_word_counts,
        }

        dbfile = open(pickle_destination, "ab")
        pickle.dump(training_data, dbfile)
        dbfile.close()

    def depickle_data(self, pickle_source: str):
        dbfile = open(pickle_source, "rb")
        db = pickle.load(dbfile)
        self.labels = db["labels"]
        self.vocab = db["vocab"]
        self.label_total_doc_counts = db["label_total_doc_counts"]
        self.label_total_word_counts = db["label_total_word_counts"]
        self.label_word_counts = db["label_word_counts"]
        dbfile.close()

    def tokenize(self, doc: str):
        bow = defaultdict(float)
        tokenizer = RegexpTokenizer(r"[A-Za-z]+")
        tokens = tokenizer.tokenize(doc)
        lowered_tokens = map(lambda t: t.lower(), tokens)
        for token in lowered_tokens:
            bow[token] += 1.0

        return bow

    def tokenize_and_update(self, doc_content: str, label: str):
        bow = self.tokenize(doc_content)

        self.label_total_doc_counts[label] += 1
        self.label_total_word_counts[label] += sum(bow.values())
        self.vocab.update(bow.keys())
        for word, count in bow.items():
            self.label_word_counts[label][word] += count

    def train(self, pickle_location: str):
        paths = [(os.path.join(self.train_dir, label), label) for label in self.labels]
        for path, label in paths:
            for f in os.listdir(path):
                with open(os.path.join(path, f), "r") as doc:
                    self.tokenize_and_update(doc.read(), label)

        self.pickle_data(pickle_location)
        self.report_stats()

    def evaluate_classifier(self):
        correct = 0.0
        total = 0.0

        paths = [(os.path.join(self.test_dir, label), label) for label in self.labels]
        for p, label in paths:
            for f in os.listdir(p):
                with open(os.path.join(p, f), "r") as doc:
                    content = doc.read()
                    bow = self.tokenize(content)
                    if self.classify(bow) == label:
                        correct += 1.0
                    total += 1.0
        return 100 * correct / total


def depickle_data(pickle_source: str, path_to_data: str) -> NaiveBayes:
    dbfile = open(pickle_source, "rb")
    db = pickle.load(dbfile)
    nb = NaiveBayes(db["labels"], path_to_data)
    nb.vocab = db["vocab"]
    nb.label_total_doc_counts = db["label_total_doc_counts"]
    nb.label_total_word_counts = db["label_total_word_counts"]
    nb.label_word_counts = db["label_word_counts"]
    dbfile.close()

    return nb
