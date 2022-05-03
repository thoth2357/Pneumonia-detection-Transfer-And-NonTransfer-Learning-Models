
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Reshape, GlobalAveragePooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import ResNet101, ResNet50V2



def model_create_and_train(model_type,data_preprocessor, callback, train_set, test_set, valid_set):
    '''
    argument: model_type (which is going to be the type of Vgg model to work on either VGG19 or VGG 16)
    purpose: Create RES model with necessary hyperparameters
    return: Created,compiled and trained ResNet model
    '''
    if model_type == 'ResNet101':
        model_name = ResNet101
    elif model_type == 'ResNet50V2':
        model_name = ResNet50V2
    else:
        return f'Error on Model type'
        quit()

    model = model_name(
        weights = 'imagenet',
        include_top = False,
        input_shape = (224, 224, 3)

    )
    for layer in model.layers:
        layer.trainable = False
    
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model_final = Model(inputs=model.input, outputs=predictions)
    model_final.summary()

    checkpoint = callback.model_checkpoint(model_type = f'{model_name}')
    learning_reducer = callback.learning_reducer()
    early_stop = callback.early_stopping()

    compiled_model = callback.model_compiler(model_final)
    
    trained_model = compiled_model.fit(
        train_set,
        epochs = data_preprocessor.EPOCHS,
        steps_per_epoch = train_set.samples // data_preprocessor.BATCH_SIZE,
        batch_size = data_preprocessor.BATCH_SIZE,

        validation_data = valid_set,
        validation_steps = test_set.samples // data_preprocessor.BATCH_SIZE - 10,
        callbacks = [checkpoint, learning_reducer, early_stop]
    )
    model_test_eval = compiled_model.evaluate(test_set)
    return trained_model, model_test_eval  

