from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
from keras.optimizers import Adam
import pickle as pkl
import os

def gen_model():
    emotion_model=Sequential()

    emotion_model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
    emotion_model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2,2)))
    emotion_model.add(Dropout(0.25))

    emotion_model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2,2)))
    emotion_model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2,2)))
    emotion_model.add(Dropout(0.25))

    emotion_model.add(Flatten())
    emotion_model.add(Dense(1024,activation='relu'))
    emotion_model.add(Dropout(0.5))
    emotion_model.add(Dense(7,activation='softmax'))

    emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001,decay=1e-6),metrics=['accuracy'])

    return emotion_model

def model_save(model,itr,emotion_dict):
    model_path=os.path.join(f'checkpoint','emotion_detection_model_epoch_{itr}.pkl')
    with open(model_path, 'wb') as f:
        pkl.dump({"model":model,"epoch":itr,"emotion_dict":emotion_dict}, f, protocol=pkl.HIGHEST_PROTOCOL)

def model_load(model_path):
    with open(model_path, 'rb') as f:
        mp = pkl.load(f)
    model=mp['model']
    epoch=mp['epoch']
    emotion_dict=mp['emotion_dict']
    
    return model,epoch,emotion_dict