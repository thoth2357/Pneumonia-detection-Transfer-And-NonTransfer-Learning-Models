from curses import reset_prog_mode
from preprocessing import *
from model_tuners import Callbacks

import VGG_Models as vgg_mod
import RES_Models as res_mod
import inception_Models as inception_mod
import EfficientNet_models as efficientNet_mod
import DenseNet_Models as denseNet_mod

data_preprocessor = Preprocessing()
callback = Callbacks(monitor = 'val_loss' , patience = 2)
data_preprocessor.basic_5descriptive_of_images()
train_image_aug, test_image_aug = data_preprocessor.data_augmentation()

train_set, test_set = data_preprocessor.dataset_splitting(train_image_aug, test_image_aug)

def main():
    models_available = {
        '1': {
            'model_type':'VGG16',
            'command': vgg_mod.model_create_and_train
        },
        '2': 'VGG19',
        '3': 'ResNet50',
        '4': 'ResNet101',
        '5': 'InceptionV3',
        '6': 'EfficientNetB0',
        '7': 'DenseNet121'
    }

    print(f'''What your model are you interested in training and fitting on our data\n
    Here Are The Following Models Available to train and evaluate\n
    {[model for model in models_available.values()]}''')

    try:
        model_choosed = str(input('\n Choose Model: '))
        assert model_choosed in models_available.values()
        for key, value in models_available.items():
            if model_choosed == value:
                model_choosed_key = key
    except AssertionError:
        print('Model choosen does not exist in the model options available. Note this could be caused by a wrong spelling')
        main()
    
    
    
    models_available[model_choosed]()
