import numpy as np

# Unigram Model
def unigramModel(sentences, tokens, totalWords, alpha):
    probability = 0
    uniProbabilities = {}
    sentences = sentences.split()
    for word in sentences:
        feature = 0.0
        if word in tokens:                                                              # If token exists in dictionary, get the probability with (occurrences of word / total words in corpus)
            feature = (tokens[word] + alpha) / (totalWords + (len(tokens) * alpha))
            uniProbabilities[word] = feature
        else:                                                                           # Else, use <UNK> as it is a trivial feature
            feature = (tokens["<UNK>"] + alpha) / (totalWords + (len(tokens) * alpha))
        probability += np.log(feature)
    
    return np.exp(-(1 / (len(sentences))) * probability), uniProbabilities

def bigramModel(sentences, tokens, biTokens, alpha):
    probability = 0
    biProbabilities = {}
    feature = 0.0
    sentences = sentences.split()
    prevWord = "<START>"
    for word in sentences:
        if prevWord == "<STOP>":
            prevWord = "<START>"
        biFeature = (prevWord, word)
        if biFeature in biTokens:                   # If token exists in dictionary, get the probability
            if prevWord == "<START>":               # If prevWord is <START>, use <STOP> as the given feature as its occurrence rate is theoretically the same as <START>
                givenFeature = tokens["<STOP>"]
                currFeature = biTokens[biFeature]
            else:                                   # Else, use prevWord as normal
                if prevWord not in tokens:
                    prevWord = word
                    continue
                givenFeature = tokens[prevWord]
                currFeature = biTokens[biFeature]
            feature = (currFeature + alpha) / (givenFeature + (len(tokens) * alpha))    # Calculate probability of bigram feature with (current word / given word)
            biProbabilities[biFeature] = feature
        # elif alpha != 0:
        #     if prevWord == "<START>":
        #         givenFeature = tokens["<STOP>"]
        #     elif prevWord in tokens:
        #         givenFeature = tokens[prevWord]
        #     else:
        #         givenFeature = 0
        #     feature = (0 + alpha) / (givenFeature + (len(tokens) * alpha))
        else:                                                                           # Else, feature does not exist
            feature = 0

        prevWord = word         # Update proceeded words

        if feature == 0:
            continue

        probability += np.log(feature)      # Sum log probabilities

    return np.exp(-(1 / (len(sentences))) * probability), biProbabilities

def trigramModel(sentences, tokens, biTokens, triTokens, alpha):
    probability = 0
    triProbabilities = {}
    feature = 0.0
    sentences = sentences.split()
    prevPrevWord = "<START>"
    prevWord = "<START>"
    for word in sentences:
        if prevWord == "<STOP>":
            prevPrevWord = "<START>"
            prevWord = "<START>"
        triFeature = (prevPrevWord, prevWord, word)
        if triFeature in triTokens:                                     # If token exists in dictionary, get the probability
            if prevPrevWord == "<START>" and prevWord == "<START>":     # If both prevPrevWord and prevWord are <START>, calculate the current and given feature like a bigram model
                givenFeature = tokens["<STOP>"]
                currFeature = biTokens[(prevWord, word)]
            elif prevPrevWord == "<START>":
                if ("<START>", prevWord) not in biTokens:
                    prevPrevWord = prevWord
                    prevWord = word
                    continue
                givenFeature = biTokens[("<START>", prevWord)]                 # Else if only prevPrevWord is <START>, use prevWord and word as the given feature
                currFeature = triTokens[triFeature]
            else:
                if (prevPrevWord, prevWord) not in biTokens:
                    prevPrevWord = prevWord
                    prevWord = word
                    continue
                givenFeature = biTokens[(prevPrevWord, prevWord)]                         # Else, use prevPrevWord and prevWord as the given feature
                currFeature = triTokens[triFeature]
            feature = (currFeature + alpha) / (givenFeature + (len(tokens) * alpha))    # Calculate probability of bigram feature with (current word / given word)
            triProbabilities[triFeature] = feature
        else:                                                                           # Else, feature does not exist
            feature = 0

        # Update proceeded words
        prevPrevWord = prevWord
        prevWord = word

        if feature == 0:
            continue

        probability += np.log(feature)      # Sum log probabilities

    return np.exp(-(1 / (len(sentences))) * probability), triProbabilities