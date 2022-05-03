import warnings 
from preprocessing import *
from model_tuners import Callbacks
from models_evaluate import plot_loss, plot_accuracy

import VGG_Models as vgg_mod
import RES_Models as res_mod
import inception_Models as inception_mod
import EfficientNet_models as efficientNet_mod
import DenseNet_Models as denseNet_mod

warnings.filterwarnings("ignore")

data_preprocessor = Preprocessing()
callback = Callbacks(monitor = 'val_loss' , patience = 2)

data_preprocessor.basic_descriptive_of_images()
train_image_aug, test_image_aug = data_preprocessor.data_augmentation()

train_set, test_set, valid_set = data_preprocessor.dataset_splitting(train_image_aug, test_image_aug)

def main():
    models_available = {
        '1': {
            'model_type':'VGG16',
            'command': vgg_mod.model_create_and_train
        },
        '2': {
            'model_type':'VGG19',
            'command': vgg_mod.model_create_and_train
        },
        '3': {
            'model_type':'ResNet50V2',
            'command': res_mod.model_create_and_train
        },
        '4': {
            'model_type':'ResNet101',
            'command': res_mod.model_create_and_train
        },
        '5': {
            'model_type': 'InceptionV3',
            'command': inception_mod.model_create_and_train
        },
        '6': {
            'model_type':'EfficientNetB0',
            'command': efficientNet_mod.model_create_and_train
        },
        '7': {
            'model_type':'DenseNet121',
            'command': denseNet_mod.model_create_and_train
        }
    }

    print(f'''\nWhat your model are you interested in training and fitting on our data\n
    Here Are The Following Models Available to train and evaluate\n
    {[model['model_type'] for model in models_available.values()]}''')

    try:
        model_choosed = str(input('\n Choose Model: '))
        model_choosed_key = ''
        assert model_choosed in [model['model_type'] for model in models_available.values()]
        for key, value in models_available.items():
            if model_choosed == value['model_type']:
                model_choosed_key = key
        #calling the respective command to start the model training based on the model choosen by user
        model_trained = models_available[model_choosed_key]['command'](model_choosed,data_preprocessor, callback, train_set, test_set, valid_set)
        
        #evaluating model performance
        plot_accuracy(model_trained.history, 'Epoch')
        
    except AssertionError:
        print('Model choosen does not exist in the model options available. Note this could be caused by a wrong spelling')
        main()
        
main()