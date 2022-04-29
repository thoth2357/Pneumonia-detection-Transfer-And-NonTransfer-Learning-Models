from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class callbacks():
    def __init__(self, monitor, patience):
        self.monitor = monitor
        self.patience = patience

    def early_stopping(self):
        '''
        argument: None
        purpose: Helps to stop model training when the generalization gap is getting worse.
        return: early_stopping object
        '''
        stop = EarlyStopping(
            monitor = self.monitor,
            patience = self.patience
        )
        return stop
    
    def learning_reducer(self):
        '''
        argument: None
        purpose: helps to reduce learning rate when a metric has stopped improving.
        return: learning reducer object
        '''
        learning_reducer = ReduceLROnPlateau(
            monitor = self.monitor,
            patience = self.patience
        )
        return learning_reducer
    
    def model_checkpoint(self):
        '''
        argument: None
        purpose: helps to save best performing models when an epoch increases the metrics ends
        '''
