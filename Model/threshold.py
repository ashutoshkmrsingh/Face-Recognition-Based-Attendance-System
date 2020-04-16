from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow.compat.v1 as tf
import numpy as np
import Model.facenet as facenet
import Model.detect_face as detect_face
import Model.classifier as classifier
import re
import pickle
import time
import cv2

tf.disable_v2_behavior()


class Create_Th:
    def __init__(self, input_datadir='./Model/student_dir_aligned', model='./Model/model_facenet/20170511-185253.pb',\
                     classifier='./Model/SVC_classifier/classifier.pkl'):
        self.input_datadir = input_datadir
        self.model = model
        self.classifier = classifier

    def set_threshold(self):
        dataset = facenet.get_dataset(self.input_datadir)
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, './Model/data')

            image_size = 180
            input_image_size = 160

            facenet.load_model(self.model)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            emb_array = np.zeros((1, embedding_size))

            self.classifier_exp = os.path.expanduser(self.classifier)
            with open(self.classifier_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
                
            threshold_dict = {}

            index = 0
            for data in dataset:
                total = 0
                for path in data.image_paths:
                    frame = cv2.imread(path, 0)
                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    cropped = facenet.flip(frame, False)
                    scaled.append(cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_LINEAR))
                    scaled = cv2.resize(scaled[0], (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape.append(scaled.reshape(-1, input_image_size, input_image_size, 3))
                    feed_dict = {images_placeholder: scaled_reshape[0], phase_train_placeholder: False}
                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    if best_class_indices[0] != index:
                        total-=predictions[0][best_class_indices[0]]
                    else:
                        total+=predictions[0][index]
                try:
                    threshold_dict[str(data.name)] = total/len(data.image_paths)
                except ZeroDivisionError:
                    pass
                index+=1
        try:
            with open('./Model/prediction_threshold/threshold.pickle', 'wb') as handle:
                pickle.dump(threshold_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except FileNotFoundError:
            os.mkdir('./Model/prediction_threshold')
            with open('./Model/prediction_threshold/threshold.pickle', 'wb') as handle:
                pickle.dump(threshold_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

