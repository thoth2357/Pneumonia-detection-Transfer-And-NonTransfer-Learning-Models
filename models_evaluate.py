import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection



def plot_loss(trained_model, xlabel):
    '''
    arguments:trained_model, axis, xlabel, ylabel
    purpose: plot the loss of the trained model against the epoch
    return: None
    '''
    model_history = trained_model.history
    model_history.update(
        {'epoch': list(
            range(
                len(model_history['val_loss'])
                )
            )
        }
    )
    model_history = pd.DataFrame.from_dict(model_history)
    best_epoch = model_history.sort_values(
        by = 'val_loss',
        ascending = True,
    ).iloc[0]['epoch']

    if not None:    
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))

    sns.lineplot(
        x = 'epochs',
        y = 'val_loss',
        ax = ax,
        data = model_history,
        label = 'Validation'
    )

    sns.lineplot(
        x = 'epochs',
        y = 'loss',
        ax = ax,
        data = model_history,
        label = 'Training', 
    )

    ax.axvline(
        x = best_epoch, 
        linestyle = '--', 
        color = 'green',
        label = 'Best Epoch'
    )
    ax.legend(loc = 1)    
    ax.set_ylim([0.1, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Loss (Fraction)')
    plt.show()


def plot_accuracy(trained_model, xlabel):
    '''
    arguments:trained_model, axis, xlabel, ylabel
    purpose: plot the loss of the trained model against the epoch
    return: None
    '''
    model_history = trained_model

    model_history.update(
        {'epoch': list(
            range(
                len(model_history['val_accuracy'])
                )
            )
        }
    )
    model_history = pd.DataFrame.from_dict(model_history)
    best_epoch = model_history.sort_values(
        by = 'val_accuracy',
        ascending = False,
    ).iloc[0]['epoch']
    
    if not None:
        fig, ax = plt.subplots(1, 1)

    sns.lineplot(
        x = 'epoch',
        y = 'val_accuracy',
        ax = ax,
        data = model_history,
        label = 'Validation'
    )

    sns.lineplot(
        x = 'epoch',
        y = 'accuracy',
        ax = ax,
        data = model_history,
        label = 'Training', 
    )
    ax.axhline(0.5, 
    linestyle = '--',
    color='red', 
    label = 'Chance')

    ax.axvline(
        x = best_epoch, 
        linestyle = '--', 
        color = 'green',
        label = 'Best Epoch'
    )
    ax.legend(loc = 1)    
    ax.set_ylim([0.5, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy (Fraction)')
    plt.show()