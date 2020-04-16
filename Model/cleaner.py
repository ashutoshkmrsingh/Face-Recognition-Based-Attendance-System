import cv2
from mtcnn import MTCNN
import Model.facenet as facenet
import os

class Cleaner:
    def __init__(self, path_to_aligned_images='./Model/student_dir_aligned'):
        self.detector = MTCNN()
        self.input_datadir = path_to_aligned_images
        
    def start_cleaner(self):
        dataset = facenet.get_dataset(self.input_datadir)

        for data in dataset:
            for path in data.image_paths:
                image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                try:
                    result = self.detector.detect_faces(image)
                except IndexError:
                    os.remove(path)