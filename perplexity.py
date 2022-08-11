import numpy as np
from ngramModels import *

class newDict(dict):
    def __init__(self):
        self = dict()
    
    def add(self, key, val):
        self[key] = val
    
    def increment(self, key):
        self[key] += 1

    def delete(self, key):
        del self[key]

class perplexity():
    def __init__ (self):
        self.totalWords = 0
        self.tokens = newDict()
        self.biTokens = newDict()
        self.triTokens = newDict()

        self.alpha = 0
        self.lambdaUni = 0
        self.lambdaBi = 0
        self.lambdaTri = 0

    def tokenPreprocessing(self, path, mode):
        # Open file to parse
        train_frame = open(path, mode, encoding="utf8")

        # Add <UNK> token onto unigram tokens dictionary
        self.tokens.add("<UNK>", 0)

        # Parse each line of the file
        for currLine in train_frame:
            currLine = currLine + " <STOP>"     # Append <STOP> onto each line
            prevWord = "<START>"                # Initialize prevWord to <START> for use in bigram and trigram
            prevPrevWord = "<START>"            # Initialize prevPrevWord to <START> for use in trigram

            # Parse each word in each line
            for word in currLine.split():
                # Unigram
                if word not in self.tokens:
                    self.tokens.add(word, 1)
                else:
                    self.tokens.increment(word)

                # Bigram
                biFeature = (prevWord, word)
                if biFeature not in self.biTokens:
                    self.biTokens.add(biFeature, 1)
                else:
                    self.biTokens.increment(biFeature)

                # Trigram
                triFeature = (prevPrevWord, prevWord, word)
                if triFeature not in self.triTokens:
                    self.triTokens.add(triFeature, 1)
                else:
                    self.triTokens.increment(triFeature)

                # Update preceeded words
                prevPrevWord = prevWord
                prevWord = word

                self.totalWords += 1     # Increment word count

        # Remove unigram features with less than 3 occurrences, incrementing <UNK> with each removal
        for j in list(self.tokens.keys()):
            if self.tokens[j] < 3 and j != "<UNK>":
                self.tokens.delete(j)
                self.tokens.increment("<UNK>")

        return
        # return tokens, biTokens, triTokens, totalWords
    
    def preprocess(self):
        self.tokenPreprocessing("./data/1b_benchmark.train.tokens", "r+")

        print("===== Preprocessed Data =====")
        print("Unique Unigrams Tokens:", len(self.tokens))
        print("Unique Bigrams Tokens:", len(self.biTokens))
        print("Unique Trigrams Tokens:", len(self.triTokens))
        print("Total Words:", self.totalWords)
        print("Total Tokens:", len(self.tokens))
        print("\n")

    def interpolate(self, corpus, uniPrb, biPrb, triPrb):
        # Store final probabilities of each set of interpolated probabilities
        # across unigram, bigram and trigram probabilities.
        weighted_avgs = [] 

        # Obtain list of words separated by whitespace in the given body of text.
        tokens = corpus.split() 

        # Initial values, relevant to bigram and trigram keys
        prevPrevWord = "<START>"
        prevWord = "<START>"

        for word in tokens:

            # If we detect current word is part of the next sentence, set initial values.
            if prevWord == "<STOP>":
                prevPrevWord = "<START>"
                prevWord = "<START>"

            # Keys for trigram and bigram
            triFeature = (prevPrevWord, prevWord, word)
            biFeature = (prevWord, word)

            # P(wi | wi2,wi1) = l1(wi) + l2(wi | wi1) + l3(wi | wi2,wi1)
            if triFeature in triPrb and biFeature in biPrb and word in uniPrb:
                xProbability = (self.lambdaUni * uniPrb[word]) + (self.lambdaBi * biPrb[biFeature]) + (self.lambdaTri * triPrb[triFeature])
                weighted_avgs.append(xProbability)

            # Update pointers to next upcoming tokens
            prevPrevWord = prevWord
            prevWord = word

        # Convert all collected probabilities into log-space
        avgs = np.array(weighted_avgs)
        logAvgs = np.log(avgs)

        # Take the sum to compute the total log-probability of the data set
        dataSetProbability = np.sum(logAvgs)

        # Average log-probability per word of the data set
        # Total tokens = total words in the corpus.
        logPrbPerWord = (1 / len(tokens)) * dataSetProbability

        # Return perplexity score, that is, the degree of uncertainty 
        perplexity_score = np.exp(-logPrbPerWord)
        return perplexity_score

    def printPerplexity(self, label, sentence):
        data = ""
        for currLine in sentence:
            currLine += " <STOP> "
            data += currLine

        if label == "train":
            print("===== Train Data =====")
        elif label == "dev":
            print("===== Dev Data =====")
        elif label == "test":
            print("===== Test Data =====")
        elif label == "debug":
            print("===== Debug Data =====")
        else:
            print("Invalid label")
            return

        unigramPerplexity, uniProbabilities = unigramModel(data, self.tokens, self.totalWords, self.alpha)
        print("Unigram Perplexity:", np.around(unigramPerplexity, 1))
        print("Exact:", unigramPerplexity, "\n")

        bigramPerplexity, biProbabilities = bigramModel(data, self.tokens, self.biTokens, self.alpha)
        print("Bigram Perplexity:", np.around(bigramPerplexity, 1))
        print("Exact:", bigramPerplexity, "\n")

        trigramPerplexity, triProbabilities = trigramModel(data, self.tokens, self.biTokens, self.triTokens, self.alpha)
        print("Trigram Perplexity:", np.around(trigramPerplexity, 1))
        print("Exact:", trigramPerplexity, "\n")
        print()
        return data, uniProbabilities, biProbabilities, triProbabilities

    def printInterpolate(self, label, data, uniPrbs, biPrbs, triPrbs):
        perplexity = self.interpolate(data, uniPrbs, biPrbs, triPrbs)

        if label == "train":
            print("===== Train Data =====")
        elif label == "dev":
            print("===== Dev Data =====")
        elif label == "test":
            print("===== Test Data =====")
        elif label == "debug":
            print("===== Debug Data =====")
        else:
            print("Invalid label")
            return

        print("Perplexity:", np.around(perplexity, 1))
        print("Exact:", perplexity)
        print()
        return
