import efficientnet.keras as efn

def model_create_and_train(model_type,data_preprocessor, callback, train_set, test_set):
    '''
    argument: model_type (which is going to be the type of Vgg model to work on either VGG19 or VGG 16)
    purpose: Create VGG model with necessary hyperparameters
    return: Created,compiled and trained Vgg model
    '''