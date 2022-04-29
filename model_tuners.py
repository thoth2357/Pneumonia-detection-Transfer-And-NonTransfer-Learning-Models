import tensorflow as tf
from tf.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tf.keras.metrics import BinaryAccuracy, Precision, Recall, AUC

class Callbacks():
    def __init__(self, monitor, patience):
        self.monitor = monitor
        self.patience = patience
        self.loss = 'binary_crossentropy'
        self.optimizer = 'accuracy'
        self.metrics = [
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ] 
       

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
    
    def model_checkpoint(self, model_type):
        '''
        argument: None
        purpose: helps to save best performing models when an epoch increases the metrics ends
        return: checkpoint object
        '''
        checkpoint = ModelCheckpoint(
            filepath = f'{model_type}_weights.hdf5',
            save_best_only = True,
            save_weights_only = False,
            mode = 'auto'
        )
        return checkpoint

    def model_compiler(self, model):
        '''
        argument: model objects
        purpose: compiles the model with the loss, optimizer and metric
        return: compiled model
        '''
        compiled_model = model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = self.metrics
        )
        return compiled_model
    
