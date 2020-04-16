from Model.align import Align_images
from Model.cleaner import Cleaner
from Model.classifier import Embedding_and_Training_Classifier
from Model.threshold import Create_Th
import pickle

class Retrain:
    def __init__(self):
        self.align_obj = Align_images()
        # self.cleaner_obj = Cleaner()
        self.train_obj = Embedding_and_Training_Classifier()
        self.thresh_obj = Create_Th()

    def start_training(self):
        self.align_obj.start_alignment()
        # self.cleaner_obj.start_cleaner()
        self.train_obj.start_train()
        self.thresh_obj.set_threshold()