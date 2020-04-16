from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf

import numpy as np
import Model.facenet as facenet
import Model.detect_face as detect_face
import os
import time
import pickle
import cv2
import re
import urllib.request


class Classify:
    def __init__(self, test_image=None, model='./Model/model_facenet/20170511-185253.pb',\
                 classifier='./Model/SVC_classifier/classifier.pkl', data='./Model/data',\
                 train_dir='./Model/student_dir_aligned', threshold_path='./Model/prediction_threshold/threshold.pickle',\
                     url='http://192.168.43.1:8080/shot.jpg', use_webcam=True):
        self.use_webcam = use_webcam
        if use_webcam == True:
            self.url = url
            
        else:
            self.img_path = test_image
        self.model = model
        self.classifier = classifier
        self.data = data
        self.label_dir = train_dir
        with open(threshold_path, 'rb') as handle:
            self.threshold = pickle.load(handle)
        self.students = []
        
    def result(self):
        # if os.path.exists('./static/frame.jpg'):
        #     os.remove('./static/frame.jpg')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, self.data)

                minsize = 20  # minimum size of face
                threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                factor = 0.709  # scale factor
                
                margin = 44
                frame_interval = 10
                batch_size = 1000
                image_size = 180
                input_image_size = 160

                student_names = os.listdir(self.label_dir)

                while True:
                    label_string = ''.join(student_names)
                    invalid_label = re.search(r'bounding_boxes_\d[0-9]*.txt', label_string)
                    try:
                        student_names.remove(label_string[invalid_label.span()[0]:invalid_label.span()[1]])
                    except AttributeError:
                        break

                facenet.load_model(self.model)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                self.classifier_exp = os.path.expanduser(self.classifier)
                with open(self.classifier_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
                    
                c = 0
                prevTime = 0
                if self.use_webcam:
                    imgResp=urllib.request.urlopen(self.url)
                    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
                    frame=cv2.imdecode(imgNp, -1)
                else:
                    frame = cv2.imread(self.img_path, 0)
                if frame.shape[0] > 1920:
                    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  


                curTime = time.time() + 1  # calc fps
                timeF = frame_interval

                if (c % timeF == 0):
                    find_results = []

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print('Face Detected: %d' % nrof_faces)

                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]

                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))

                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('face is too close')
                                continue

                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(cv2.resize(cropped[i], (image_size, image_size), interpolation=cv2.INTER_LINEAR))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                   interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            
                            predictions = model.predict_proba(emb_array)
                            print('predictions',predictions)
                            
                            best_class_indices = np.argmax(predictions, axis=1)
                            print('best class index',best_class_indices)
                            
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            print('best_class_prob',best_class_probabilities)
                            
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 1)  # boxing face

                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                        
                            print('Result Indices: ', best_class_indices[0])
                            print('student names',student_names)
                            for H_i in student_names:
                                print('H_i', H_i)
                                print('student_names[best_class_indices[0]]',student_names[best_class_indices[0]])
                                print('self.threshold[student_names[best_class_indices[0]]]',self.threshold[student_names[best_class_indices[0]]])
                                if student_names[best_class_indices[0]] == H_i:
                                    if best_class_probabilities[0] < self.threshold[student_names[best_class_indices[0]]]:
                                        result_names = 'Unk'
                                        print(result_names)
                                    else:
                                        result_names = student_names[best_class_indices[0]]
                                        self.students.append(result_names)

                                    cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=2)
                                    print('\n\n')
                    else:
                        print('Unable to align')
                # cv2.imshow('Image', color_frame)
                if frame.shape[1] > 800:
                    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) 
                cv2.imwrite('./static/frame.jpg', frame)
                print('frame write successful')

                # if cv2.waitKey(1000000) & 0xFF == ord('q'):
                #     print('done')
                # cv2.destroyAllWindows()
                
            return set(self.students)