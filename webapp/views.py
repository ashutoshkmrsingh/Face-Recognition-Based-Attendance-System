from django.shortcuts import render
import cv2
from Model.retrain import Retrain
import pickle
import os
from Model.output import Classify

# Create your views here.
def index(request):
    return render(request,"index.html")

def login(request):
    username = request.POST['username']
    password = request.POST['pass']

    if username == 'admin' and password == '12345':
        return render(request, "main.html")
    else:
        return render(request, "index.html", {'flag':False})

def retrain(request):
    try:
        if not os.path.exists("./Model/student_dir_captured"):
            return render(request, "main.html", {'accuracy':{'not exists':0},
                                            '     retrain':True,})
        else:
            obj = Retrain()
            obj.start_training()
            threshold_path='./Model/prediction_threshold/threshold.pickle'
            with open(threshold_path, 'rb') as handle:
                threshold = pickle.load(handle)
            print(threshold)
            return render(request, "main.html", {'accuracy':threshold,
                                                'retrain':True,})
    except Exception:
        return render(request, "main.html", {'accuracy':{'not exists':0},
                                            '     retrain':True,})

students = []
def start(request):
    global students
    obj = Classify(test_image='./Model/test/image2.jpeg', use_webcam=False)
    # obj = Classify()
    detected = list(obj.result())
    students+=detected
    return render(request, "main.html", {'image':True,
                                            'students':set(students)})