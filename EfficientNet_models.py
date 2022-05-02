from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Reshape, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def model_create_and_train(model_type, data_preprocessor, callback, train_set, test_set):
    '''
    argument: 
    purpose: Create VGG model with necessary hyperparameters
    return: Created,compiled and trained Vgg model
    '''
    model = EfficientNetB0(
        input_shape = (224, 224, 3),
        include_top = False,
        weights = 'imagenet'
    )

    for layer in model.layers:
        layer.trainable = False
    
    x  = Flatten()(model.output)
    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = Dense(1024, activation='relu')(x)

    # Add a dropout rate of 0.5
    x = Dropout(0.5)(x)

    # Add a final sigmoid layer with 1 node for classification output
    x = Dense(1, activation='sigmoid')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model_final = Model(inputs=model.input, outputs=predictions)
    
    checkpoint = callback.model_checkpoint(model_type = 'VGG19')
    learning_reducer = callback.learning_reducer()
    early_stop = callback.early_stopping()

    compiled_model = callback.model_compiler(model_final)

    trained_model = compiled_model.fit_generator(
        train_set,
        epochs = data_preprocessor.EPOCHS,
        steps_per_epoch = train_set.samples // data_preprocessor.BATCH_SIZE,
        batch_size = data_preprocessor.BATCH_SIZE,

        validation_data = test_set,
        validation_steps = test_set.samples // data_preprocessor.BATCH_SIZE - 10,
        callbacks = [checkpoint, learning_reducer, early_stop]
    )
    return trained_model
    