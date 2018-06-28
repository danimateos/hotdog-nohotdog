from itertools import chain
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_training_histories(*histories):
    '''Abstract away the plotting of training histories'''
    
    figure, axes = plt.subplots(2,1)
    figure.set_size_inches(8,8)
    
    x_axis = range(sum([len(history.epoch) for history in histories]))
    
    for ax, metrics in enumerate([['loss', 'val_loss'], ['acc', 'val_acc']]):

        for metric in metrics:

            this_metric = [history.history[metric] for history in histories]
            merged = list(chain.from_iterable(this_metric))

            axes[ax].plot(x_axis, merged, label=metric)

        axes[ax].legend()
    
    return figure


def confusion_matrix(model, validation_generator):
    '''Plot the confusion matrix, return precision and recall'''
    
    predictions = model.predict_generator(validation_generator)

    C = confusion_matrix(validation_generator.classes, predictions > .5)
    ax = sns.heatmap(C, annot=True, square=True)
    ax.set_ylabel('True class')
    ax.set_xlabel('Predicted class')

    precision = C[0,0] / (C[0,0] + C[1,0])
    recall = C[0][0] / (C[0,0] + C[0,1])

    return ax, precision, recall