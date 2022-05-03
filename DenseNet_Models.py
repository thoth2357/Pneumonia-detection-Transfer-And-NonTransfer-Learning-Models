from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Reshape, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def model_create_and_train(model_type, data_preprocessor, callback, train_set, test_set):
    '''
    argument: 
    purpose: Create DenseNet model with necessary hyperparameters
    return: Created,compiled and trained DenseNEt model
    '''

    model = DenseNet121(
        weights = 'imagenet',
        include_top = False,
        input_shape = (224, 224, 3)

    )

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(1, activation='sigmoid')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model_final = Model(inputs=model.input, outputs=predictions)

    checkpoint = callback.model_checkpoint(model_type = 'DenseNet121')
    learning_reducer = callback.learning_reducer()
    early_stop = callback.early_stopping()

    compiled_model = callback.model_compiler(model_final)

    trained_model = compiled_model.fit(
        train_set,
        epochs = data_preprocessor.EPOCHS,
        steps_per_epoch = train_set.samples // data_preprocessor.BATCH_SIZE,
        batch_size = data_preprocessor.BATCH_SIZE,

        validation_data = test_set,
        validation_steps = test_set.samples // data_preprocessor.BATCH_SIZE - 10,
        callbacks = [checkpoint, learning_reducer, early_stop]
    )
    return trained_model
