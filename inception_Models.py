from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Reshape, GlobalAveragePooling2D


def model_create_and_train(model_type, data_preprocessor, callback, train_set, test_set, valid_set):
    '''
    argument:
    purpose: Create Inference model with necessary hyperparameters
    return: Created,compiled and trained Inference model
    '''

    model = InceptionV3(
        input_shape = (224, 224, 3),
        include_top = False,
        weights = 'imagenet'
    )

    for layer in model.layers:
        layer.trainanle = False
    
    x = Flatten()(model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(1, activation='sigmoid')(x)

    model_final = Model(inputs=model.input, outputs=x)

    checkpoint = callback.model_checkpoint(model_type = 'InceptionV3')
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

