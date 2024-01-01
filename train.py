import argparse
import os
from keras.preprocessing.image import ImageDataGenerator
from src.utils import *

def main(args):
    DATA_DIR=args.input_path

    train_data_gen=ImageDataGenerator(rescale=1./255)
    validation_data_gen=ImageDataGenerator(rescale=1./255)

    train_generator=train_data_gen.flow_from_directory(
        os.path.join(DATA_DIR,'train'),
        target_size=(48,48),
        batch_size=args.batch_size,
        color_mode='grayscale',
        class_mode='categorical'
    )
    validation_generator=validation_data_gen.flow_from_directory(
        os.path.join(DATA_DIR,'test'),
        target_size=(48,48),
        batch_size=args.batch_size,
        color_mode='grayscale',
        class_mode='categorical'
    )

    if args.resume_training==None:
        emotion_model=gen_model()
        emotion_dict={0:'Angry',1:'Disgusted',2:'Fearful',3:'Happy',4:'Neutral',5:'Sad',6:'Surprised'}
    else:
        emotion_model,epoch,emotion_dict=model_load(args.resume_training)


    n_epochs=args.epochs
    batch_size=int(args.batch_size)
    train_steps=os.listdir(os.path.join(DATA_DIR,'train'))//batch_size
    test_steps=os.listdir(os.path.join(DATA_DIR,'test'))//batch_size

    print('STARTING TRAINING...')
    for i in range(n_epochs):
        emotion_model_info=emotion_model.fit_generator(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=1,
            validation_data=validation_generator,
            validation_steps=test_steps)

        if (epoch+i+1)%5==0:
            model_save(emotion_model,epoch+i+1,emotion_dict)
            print('MODEL SAVED AT EPOCH :',epoch+i+1)
    
    model_save(emotion_model,epoch+i+1,emotion_dict)
    print('MODEL SAVED AT EPOCH :',epoch+i+1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Caption Generator", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input_path", type=str, default=None, help="path to the input folder")
    parser.add_argument("-e", "--epochs", type=str, default="10", help="Number of Epochs")
    parser.add_argument("-b", "--batch_size", type=str, default="64", help="Batch size")
    parser.add_argument("-r", "--resume_training", type=str, default=None, help="To Resume the Training, give the model path")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)