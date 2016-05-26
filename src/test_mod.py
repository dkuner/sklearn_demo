import csv
import math
import pickle
import random

def load_csv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]        
    return dataset


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg, 2) for x in numbers]) / float(len(numbers)-1)
    return math.sqrt(variance)


def calculate_probability(x, mean, stdev):
      exponent = math.exp(-(math.pow(x-mean,2) / (2*math.pow(stdev,2))))
      return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = input_vector[i]
            probabilities[class_value] *= calculate_probability(x, mean, stdev)
    return probabilities


def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
                  if best_label is None or probability > best_prob:
                      best_prob = probability
                      best_label = class_value
    return best_label


def get_predictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(test_set))) * 100.0


def main():

    # params
    datafile = '../dataset/pima-indians-diabetes.data.csv'
    resultfile = '../results/result.csv'
    modelfile = '../models/model_nb.pkl'
 
    # load data
    data_set = load_csv(datafile)
    test_set = data_set
    print('Load {0} items'.format(len(data_set)))

    # load model
    summaries = pickle.load(open(modelfile, 'r'))
    print('Summaries: {0}'.format(summaries))

    # do predictions
    predictions = get_predictions(summaries, data_set)
    print('Predictions: {0}'.format(predictions))
   
    # appraise result
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: {0}%'.format(accuracy))

    # save result
    csvfile = file(resultfile, 'w')
    writer = csv.writer(csvfile)
    writer.writerow(predictions)


if __name__ == "__main__":
    main()
