
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Reshape, GlobalAveragePooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import VGG19, VGG16



from preprocessing import *
from model_tuners import Callbacks

data_preprocessor = Preprocessing()
callback = Callbacks(monitor = 'val_loss' , patience = 2)
data_preprocessor.basic_descriptive_of_images()
train_image_aug, test_image_aug = data_preprocessor.data_augmentation()

train_set, test_set = data_preprocessor.dataset_splitting(train_image_aug, test_image_aug)


def model_create(model_type,):
    '''
    argument: model_type (which is going to be the type of Vgg model to work on either VGG19 or VGG 16)
    purpose: Create VGG model with necessary hyperparameters
    return: Created and compiled Vgg model waiting to be trained
    '''
    if model_type == 'VGG19':
        model = tf.keras.applications.VGG19(
            weights = 'imagenet',
            include_top = False
        )
        for layer in model.layers:
            layer.trainable = False
        
        flattened = Flatten()(model.output)
        predictions = Dense(2, activation='sigmoid')(x)

        model_final = Model(inputs=model.input, outputs=predictions)

        checkpoint = callback.model_checkpoint(model_type = 'VGG19')
        learning_reducer = callback.learning_reducer()
        early_stop = callback.early_stopping()

        compiled_model = callback.model_compiler(model_final)
       
        trained_model = compiled_model.fit(
            train_set,
            epochs = data_preprocessor.,

        )