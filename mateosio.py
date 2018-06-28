from itertools import chain
import matplotlib.pyplot as plt

def plot_training_histories(*histories):

    

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