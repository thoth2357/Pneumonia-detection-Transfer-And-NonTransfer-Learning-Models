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
 
print('''What your model are you interested in training and fitting on our data\n
Here Are The Following Models Available to train and evaluate''')
