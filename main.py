from re import T
import numpy as np
import argparse
from perplexity import *
from math import fsum

def main():
    test_frame = open("./data/1b_benchmark.test.tokens", "r+", encoding="utf8")
    dev_frame = open("./data/1b_benchmark.dev.tokens", "r+", encoding="utf8")
    train_frame = open("./data/1b_benchmark.train.tokens", "r+", encoding="utf8")
    debug_frame = open("./data/1b_benchmark.debug.tokens", "r+", encoding="utf8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help="Does debug test.", action="store_true")
    ifDebug = parser.parse_args()

    model = perplexity()

    model.alpha = float(input("Enter alpha value: "))
    if model.alpha < 0:
        print("Invalid value for alpha")
        return -1
    print()

    if model.alpha == 0: # Interpolation with no smoothing applied case
        model.lambdaUni = float(input("Enter lambda value (l1) between 0 and 1 for unigram: "))
        model.lambdaBi = float(input("Enter lambda value (l2) between 0 and 1 for bigram: "))
        model.lambdaTri = float(input("Enter lambda value (l3) between 0 and 1 for trigram: "))

        # Input validation: lambda values must be non-negative and sum to 1
        if model.lambdaTri < 0 or model.lambdaBi < 0 or model.lambdaUni < 0:
            print("Invalid values for l1, l2 or l3. Please enter positive decimal values!")
            return -1

        elif fsum([model.lambdaTri, model.lambdaBi, model.lambdaUni]) != 1: 
            # Caution: lambdaUni + lambdaBi + lambdaTri exhibits unpredictable behavior with floating pt values
            print("Invalid values for l1, l2 or l3. Please ensure values sum to 1!")
            return -1

        else:
            print("Your values are λ_1 =", model.lambdaUni, "; λ_2 =", model.lambdaBi, "; λ_3 =", model.lambdaTri)
        print()
    
    model.preprocess()

    # File pointer reaches end upon file read in printPerplexity
    if ifDebug.debug == True:
        debugData, debugUniPrbs, debugBiPrbs, debugTriPrbs = model.printPerplexity("debug", debug_frame)
    else:
        trainData, trainUniPrbs, trainBiPrbs, trainTriPrbs = model.printPerplexity("train", train_frame)
        devData, devUniPrbs, devBiPrbs, devTriPrbs = model.printPerplexity("dev", dev_frame)
        testData, testUniPrbs, testBiPrbs, testTriPrbs = model.printPerplexity("test", test_frame)

    # Results for interpolation perplexities, no smoothing
    if model.alpha == 0:
        print("===== Linear Interpolation =====")
        print("Interpolating with values λ_1 =", model.lambdaUni, "; λ_2 =", model.lambdaBi, "; λ_3 =", model.lambdaTri, "...")
        print()

        # Compute perplexity for interpolated probabilities for train, dev, and test data sets respectively.
        if ifDebug.debug == True:
            model.printInterpolate("debug", debugData, debugUniPrbs, debugBiPrbs, debugTriPrbs)
        else:
            model.printInterpolate("train", trainData, trainUniPrbs, trainBiPrbs, trainTriPrbs)
            model.printInterpolate("dev", devData, devUniPrbs, devBiPrbs, devTriPrbs)
            model.printInterpolate("test", testData, testUniPrbs, testBiPrbs, testTriPrbs)
    return 0

main()
