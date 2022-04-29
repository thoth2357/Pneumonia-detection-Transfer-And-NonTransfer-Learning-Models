
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Reshape, GlobalAveragePooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import ResNet101, ResNet50



def model_create_and_train(model_type,data_preprocessor, callback, train_set, test_set):
    '''
    argument: model_type (which is going to be the type of Vgg model to work on either VGG19 or VGG 16)
    purpose: Create VGG model with necessary hyperparameters
    return: Created,compiled and trained ResNet model
    '''
    if model_type == 'ResNet101':
        model_name = ResNet101
    elif model_type == 'ResNet50':
        model_name = ResNet50
    else:
        return f'Error on Model type'
        quit()

    model = model_name(
        weights = 'imagenet',
        include_top = False
    )
    for layer in model.layers:
        layer.trainable = False
    
    x = model.output
    predictions = Dense(1, activation='sigmoid')(x)

    model_final = Model(inputs=model.input, outputs=predictions)

    checkpoint = callback.model_checkpoint(model_type = 'VGG19')
    learning_reducer = callback.learning_reducer()
    early_stop = callback.early_stopping()

    compiled_model = callback.model_compiler(model_final)
    
    trained_model = compiled_model.fit(
        train_set,
        epochs = data_preprocessor.EPOCHS,
        steps_per_epoch = train_set.samples // data_preprocessor.BATCH_SIZE,
        batch_size = data_preprocessor.batch_size,

        validation_data = test_set,
        validation_steps = test_set.samples // data_preprocessor.BATCH_SIZE - 10,
        callback = [checkpoint, learning_reducer, early_stop]
    )
    return trained_model

