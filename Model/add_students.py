import cv2
import os
import shutil
import numpy as np
from mtcnn import MTCNN
import copy
import urllib.request


detector = MTCNN()

class Add_Student:
    def __init__(self, enrollment_no, student_dir = './student_dir_captured', url='http://192.168.43.1:8080/shot.jpg', use_webcam=False):
        self.enrollment_number = enrollment_no.upper()

        if not os.path.exists(student_dir):
            os.makedirs(student_dir)
                
        curr_dir = os.path.abspath(os.path.join(os.path.expanduser(student_dir), self.enrollment_number))
                
        if os.path.exists(curr_dir):
            shutil.rmtree(curr_dir)
            
        os.makedirs(curr_dir)
        self.student_dir = curr_dir

        self.url = url
        self.use_webcam = use_webcam
        
    def capture_image(self, delay=15, no_of_images=50):
        os.chdir(self.student_dir)

        delay = delay
        frame_delay = delay
        count = 0
        total_images = no_of_images
        if not self.use_webcam:
            camera = cv2.VideoCapture(0)

            camera.set(3, 640)
            camera.set(4, 480)
        
        flag = 0
        
        def write_frame(image, text, position, color, linetype):
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            
            cv2.putText(image,
                            text, 
                            position, 
                            font, 
                            fontScale,
                            color,
                            linetype)
        while True:
            if self.use_webcam:
                imgResp=urllib.request.urlopen(self.url)
                imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
                image=cv2.imdecode(imgNp,-1)
            else:
                retval, image = camera.read()
                assert retval == True, 'Camera not detected'

            try:
                original_image = copy.deepcopy(image)
                result = detector.detect_faces(image)
                
                if len(result) <= 1:
                    flag = 0
                    result = result[0]

                    bounding_box = result['box']
                    keypoints = result['keypoints']

                    center_coordinates = (bounding_box[0]+(bounding_box[2]//2), bounding_box[1]+(bounding_box[3]//2))
                    axesLength = (bounding_box[2]//2, bounding_box[3]//2)
                    angle = 0
                    startAngle = 0
                    endAngle = 360
                    color = 255,255,86
                    thickness = 1
                    cv2.ellipse(image, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness) #face

                    cv2.ellipse(image,(keypoints['left_eye']), (bounding_box[2]//10, bounding_box[3]//20), angle, startAngle, endAngle, (252,255,255), 1)

                    cv2.ellipse(image,(keypoints['right_eye']), (bounding_box[2]//10, bounding_box[3]//20), angle, startAngle, endAngle, (252,255,255), 1)

                    cv2.circle(image,(keypoints['nose']), 2, (102, 0, 204), 8)

                    center_coordinates = ((keypoints['mouth_left'][0]+((keypoints['mouth_right'][0]-keypoints['mouth_left'][0])//2)),\
                                        abs((keypoints['mouth_left'][1]+((keypoints['mouth_right'][1]-keypoints['mouth_left'][1])//2))))

                    axesLength = ((keypoints['mouth_right'][0]-keypoints['mouth_left'][0])//2, 8)
                    angle = 0
                    startAngle = 10
                    endAngle = 170
                    color = (0,0,255)  
                    thickness = 3
                    cv2.ellipse(image, center_coordinates, axesLength, 
                               angle, startAngle, endAngle, color, thickness)

                    cv2.circle(image,(keypoints['mouth_left']), 2, (0,0,255), 1)
                    cv2.circle(image,(keypoints['mouth_right']), 2, (0,0,255), 1)

                    if frame_delay % delay == 0:
                        frame_delay = 0
                        count += 1
                        write_frame(image, 'CLICKED', (40, 40), (255,0,0), 2)
                        cv2.imwrite('%d.png'%count, original_image)

                    frame_delay += 1

                    write_frame(image, 'FRAME DELAY: '+str(frame_delay), (350, 40), (255,0,0), 2)
                    write_frame(image, 'CAPTURED: '+str(count)+' / '+str(total_images), (165, 460), (0,255,0), 2)
                else:
                    flag += 1
                    if flag >= 5:
                        write_frame(image, 'MORE THAN 1 PERSON DETECTED', (60, 40), (255,0,255), 2)
            except IndexError:
                write_frame(image, "PLEASE DON'T MOVE!", (150, 200), (0,0,255), 4)
                pass
                        
            cv2.imshow('Press Q to quit', image[::1])
            
            if count == total_images:
                if not self.use_webcam:
                    camera.release()
                cv2.destroyAllWindows()
                break
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                if not self.use_webcam:
                    camera.release()
                break
        
        os.chdir('..')
        os.chdir('..')