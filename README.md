# Emotion Detector
Emotion recognition is the process of identifying human emotion. People vary widely in their accuracy at recognizing the emotions of others. Use of technology to help people with emotion recognition.
Emotion recognition software not only improves human-computer interfaces but also improves the actions made by computers in response to user feedback.
## Installation
```
pip install -r requirements.txt
```
The following project also uses haarcascade_frontalface_default opencv cascade for face detection for emotion recognition.
## Train
```
python train.py -i <INPUT DIR PATH> -e <EPOCHS> -b <BATCH SIZE> -r <RESUME TRAINING MODEL PATH>
```
## Test
```
python predict_emotion.py -i <INPUT IMG FILE> -m <MODEL PATH>
```