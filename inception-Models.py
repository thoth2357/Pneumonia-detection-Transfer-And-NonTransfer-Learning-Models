from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Reshape, GlobalAveragePooling2D


def model_create_and_train(model_type, data_preprocessor, callback, train_set, test_set):
    '''
    argument: model_type (which is going to be the type of Vgg model to work on either VGG19 or VGG 16)
    purpose: Create Inference model with necessary hyperparameters
    return: Created,compiled and trained Inference model
    '''

    model = InceptionV3(
        input_shape = (150, 150, 3),
        include_top = False,
        weights = 'imagenet'
    )

    for layer in model.layers:
        layer.trainanle = False
    
    x = Flatten()(model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(1, activation='sigmoid')(x)

    model_final = Model(inputs=model.input, output=x)

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