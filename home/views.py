from django.shortcuts import render, HttpResponse
from django.core.files.storage import FileSystemStorage
from home.forms import UploadFileForm
from home.models import File_upload
from keras.models import model_from_json
import numpy as np
import librosa
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")
import requests

json_file = open('./savedModel/model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./savedModel/model1.h5")
print("Loaded model from disk")


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0.0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)


loaded_model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=['accuracy', precision, recall, fmeasure])

CLASSES = ['a', 'am', 'bm', 'c', 'd', 'dm', 'e', 'em', 'f', 'g']

result = ""


def index(request):
    if request.method == 'POST':
        c_form = UploadFileForm(request.POST, request.FILES)
        if c_form.is_valid():
            y, sr = librosa.load(request.FILES['file'])
            total_duration = librosa.get_duration(y=y, sr=sr)
            iterate = math.floor(total_duration/2)
            predVal = {}
            for i in range(0, iterate):
                start_time = i*2
                start_index = int(start_time * sr)
                end_index = int(start_index + (sr*2))
                segment = y[start_index:end_index]
                ps = librosa.feature.melspectrogram(y=segment, sr=sr)
                ps = np.array(ps.reshape(1, 128, 87, 1))
                predicted = loaded_model.predict(ps)
                predicted = np.argmax(predicted, axis=1)
                predVal[int(end_index/sr)] = CLASSES[predicted[0]]
            keys = list(predVal.keys())
            values = list(predVal.values())
            heatData = np.zeros((len(CLASSES), len(keys)))
            k = 0
            for i, j in predVal.items():
                if k != len(keys):
                    heatData[CLASSES.index(j)][k] = 1
                    k = k+1
            hm = sns.heatmap(heatData, cmap="cividis", yticklabels=CLASSES,
                             xticklabels=keys, linewidths=0.5, linecolor='white')
            plt.xlabel("Time (s)")
            plt.ylabel("Chords")
            plt.savefig('./static/images/result.png')
            plt.close()
            return render(request, 'answer.html', {'result': values})
    else:
        content = {
            'forms': UploadFileForm()
        }
    return render(request, 'index.html', content)
