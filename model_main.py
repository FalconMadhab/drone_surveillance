# Import packages
import os
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from imutils.video import VideoStream,FileVideoStream
from imutils.video import FPS
from skimage.measure import compare_ssim
import json
from datetime import datetime


# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

def detect_low_light(frame,light_thresh):
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    is_light = np.mean(gray_frame) > light_thresh
    if is_light:
        return 0
    else:
        1
'''
def loitering_prediction(interpreter,input_details,output_details,image,CLASSIFIER_CKPT):
    classifier_interpreter = Interpreter(model_path=CLASSIFIER_CKPT)
    classifier_interpreter.allocate_tensors()
    input_details_classifier = classifier_interpreter.get_input_details()
    output_details_classifier = classifier_interpreter.get_output_details()
    _, input_dim, _, _ = input_details[0]['shape']
    _, mp_dim, _, ky_pt_num = output_details[0]['shape']
    image1 = cv2.resize(image , (192,192))
    image1 = np.reshape(image1 , (1,192,192,3))
    # image1 = tf.cast(image1 ,tf.float32)
    image1 = np.asarray(image1  ,dtype = np.float32)
    interpreter.set_tensor(input_details[0]['index'], image1)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])
    feat_array= []
    # Process result and create feature array
    res = result.reshape(1, mp_dim**2, ky_pt_num)
    max_idxs = np.argmax(res, axis=1)
    coords = list(map(lambda x: divmod(x, mp_dim), max_idxs))
    feature_vec = np.vstack(coords).T.reshape(2 * ky_pt_num, 1)
    feat_array.append(feature_vec)
    sample = np.array(feat_array).squeeze()
    sample = sample.reshape((1,28,1))
    sample = np.asarray(sample  ,dtype = np.float32)
    classifier_interpreter.set_tensor(input_details_classifier[0]['index'],sample)
    classifier_interpreter.invoke()
    # print(output_details_classifier)
    result_classifier = classifier_interpreter.get_tensor(output_details_classifier[0]['index'])
    # print(result_classifier)
    if(result_classifier[0][0]> 0.7 ):
        return 1
    else:
        return 0
'''

# MODEL_HELMET = 'models\\helmet_model\\data'
# MODEL_MASK='models\\mask_model\\data'
MODEL_ANIMAL = 'models\\animal_model\\data'
MODEL_CROWD='models\\crowd_model\\data'
# MODEL_DOOR='models\\door_model\\data'
# MODEL_LOITERING='models\\loitering_model\\data'
# CLASSIFIER_GRAPH_NAME='lstm.tflite'
# POSE_GRAPH_NAME='pose.tflite'
GRAPH_NAME = 'float_model.tflite' #In each data folder there is float_model.tflite
LABELMAP_NAME = 'labels.txt' #In each data folder there is labels.txt
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
# HELMET_CKPT = os.path.join(CWD_PATH,MODEL_HELMET,GRAPH_NAME)
# MASK_CKPT = os.path.join(CWD_PATH,MODEL_MASK,GRAPH_NAME)
ANIMAL_CKPT = os.path.join(CWD_PATH,MODEL_ANIMAL,GRAPH_NAME)
CROWD_CKPT = os.path.join(CWD_PATH,MODEL_CROWD,GRAPH_NAME)
# DOOR_CKPT = os.path.join(CWD_PATH,MODEL_DOOR,GRAPH_NAME)
# LOITERING_CKPT = os.path.join(CWD_PATH,MODEL_LOITERING,POSE_GRAPH_NAME)
# CLASSIFIER_CKPT = os.path.join(CWD_PATH,MODEL_LOITERING,CLASSIFIER_GRAPH_NAME)
# Path to label map file
# HELMET_LABELS = os.path.join(CWD_PATH,MODEL_HELMET,LABELMAP_NAME)
# MASK_LABELS = os.path.join(CWD_PATH,MODEL_MASK,LABELMAP_NAME)
ANIMAL_LABELS = os.path.join(CWD_PATH,MODEL_ANIMAL,LABELMAP_NAME)
CROWD_LABELS = os.path.join(CWD_PATH,MODEL_CROWD,LABELMAP_NAME)
# DOOR_LABELS = os.path.join(CWD_PATH,MODEL_DOOR,LABELMAP_NAME)

# Load the label map
# with open(HELMET_LABELS, 'r') as f:
#     helmet_labels = [line.strip() for line in f.readlines()]
# with open(MASK_LABELS, 'r') as f:
#     mask_labels = [line.strip() for line in f.readlines()]
with open(ANIMAL_LABELS, 'r') as f:
    animal_labels = [line.strip() for line in f.readlines()]
with open(CROWD_LABELS, 'r') as f:
    crowd_labels = [line.strip() for line in f.readlines()]
# with open(DOOR_LABELS, 'r') as f:
#     door_labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
# if helmet_labels[0] == '???':
#     del(helmet_labels[0])
# if mask_labels[0] == '???':
#     del(mask_labels[0])
if animal_labels[0] == '???':
    del(animal_labels[0])
if crowd_labels[0] == '???':
    del(crowd_labels[0])
# if door_labels[0] == '???':
#     del(door_labels[0])

# helmet_interpreter = Interpreter(model_path=HELMET_CKPT)
# mask_interpreter = Interpreter(model_path=MASK_CKPT)
animal_interpreter = Interpreter(model_path=ANIMAL_CKPT)
crowd_interpreter = Interpreter(model_path=CROWD_CKPT)
# door_interpreter = Interpreter(model_path=DOOR_CKPT)
# loitering_interpreter = Interpreter(model_path=LOITERING_CKPT)

# helmet_interpreter.allocate_tensors()
# mask_interpreter.allocate_tensors()
animal_interpreter.allocate_tensors()
crowd_interpreter.allocate_tensors()
# door_interpreter.allocate_tensors()
# loitering_interpreter.allocate_tensors()
# Get model details
input_details = animal_interpreter.get_input_details() #Since mask, animal, helmet and door input details are same for all model
crowd_input_details=crowd_interpreter.get_input_details()
# loitering_input_details=loitering_interpreter.get_input_details()

# helmet_output_details = helmet_interpreter.get_output_details()
# mask_output_details = mask_interpreter.get_output_details()
animal_output_details = animal_interpreter.get_output_details()
crowd_output_details = crowd_interpreter.get_output_details()
# door_output_details = helmet_interpreter.get_output_details()
# loitering_output_details = loitering_interpreter.get_output_details()

height = input_details[0]['shape'][1]
width =input_details[0]['shape'][2]
crowd_height = crowd_input_details[0]['shape'][1]
crowd_width =crowd_input_details[0]['shape'][2]
#uncoment below 3 lines for floating model
floating_model = (input_details[0]['dtype'] == np.float32)
crowd_floating_model = (crowd_input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5


# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

#socket program server side 
# import socket                
# host = "10.175.1.148"
# # the port, let's use 5001
# port = 12345
# # next create a socket object 
# s = socket.socket()          
# print("Socket successfully created")
# print(f"[+] Connecting to {host}:{port}")
# s.connect((host, port))
# print("[+] Connected.")

# Initialize video stream
vid_path='17-05-46.flv'
# videostream = FileVideoStream(vid_path).start()
videostream = cv2.VideoCapture(vid_path)
time.sleep(1)
start_time = time.time()
frame_count=0
min_conf_threshold=0.75
light_thresh=90
nop_thresh=0
alert_dict={'time':0,'animal':0,'crowd':0}
#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    #t1 = cv2.getTickCount()

    # Grab frame from video stream
    # frame1 = videostream.read()
    ret,frame1 = videostream.read()
    if ret==False:
        break
    frame_count+=1
    if detect_low_light(frame1,light_thresh)==1:
        s.send('Low Light Detected'.encode('utf-8'))
    else:
        imH,imW=frame1.shape[:2]
        h_s,w_s=int(imW/10),int(imW/10)

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        helmet=0
        mask=0
        crowd=0
        animal=0
        door=0
        loitering=0

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        crowd_frame_resized = cv2.resize(frame_rgb, (crowd_width, crowd_height))
        #frame_resized =frame_resized.astype(np.uint8)         # added for np.UINT8 quantized model
        input_data = np.expand_dims(frame_resized, axis=0)
        crowd_input_data = np.expand_dims(crowd_frame_resized, axis=0)
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        if crowd_floating_model:
            crowd_input_data = (np.float32(crowd_input_data) - input_mean) / input_std
        # Perform the actual detection by running the model with the image as input
        # helmet_interpreter.set_tensor(input_details[0]['index'],input_data)
        # helmet_interpreter.invoke()

        # # Retrieve detection results
        # helmet_boxes = helmet_interpreter.get_tensor(helmet_output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        # helmet_classes = helmet_interpreter.get_tensor(helmet_output_details[1]['index'])[0] # Class index of detected objects
        # helmet_scores = helmet_interpreter.get_tensor(helmet_output_details[2]['index'])[0] # Confidence of detected objects
        # #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
        # # print('helmet_scores:',helmet_scores)

        # # Loop over all detections and draw detection box if confidence is above minimum threshold
        # for i in range(len(helmet_scores)):
        #     if ((helmet_scores[i] > min_conf_threshold) and (helmet_scores[i] <= 1.0)):

        #         '''
        #         # Get bounding box coordinates and draw box
        #         # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
        #         helmet_ymin = int(max(1,(helmet_boxes[i][0] * imH)))
        #         helmet_xmin = int(max(1,(helmet_boxes[i][1] * imW)))
        #         helmet_ymax = int(min(imH,(helmet_boxes[i][2] * imH)))
        #         helmet_xmax = int(min(imW,(helmet_boxes[i][3] * imW)))

        #         cv2.rectangle(frame, (helmet_xmin,helmet_ymin), (helmet_xmax,helmet_ymax), (10, 255, 0), 2)

        #         # Draw label
        #         object_name1 = helmet_labels[int(helmet_classes[i])] # Look up object name from "labels" array using class index
        #         helmet_label = '%s: %d%%' % (object_name1, int(helmet_scores[i]*100)) # Example: 'person: 72%'
        #         helmet_labelSize, helmet_baseLine = cv2.getTextSize(helmet_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        #         helmet_label_ymin = max(helmet_ymin, helmet_labelSize[1] + 10) # Make sure not to draw label too close to top of window
        #         cv2.rectangle(frame, (helmet_xmin, helmet_label_ymin-helmet_labelSize[1]-10), (helmet_xmin+helmet_labelSize[0], helmet_label_ymin+helmet_baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        #         cv2.putText(frame, helmet_label, (helmet_xmin, helmet_label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        #         '''
        #         object_name1 = helmet_labels[int(helmet_classes[i])] # Look up object name from "labels" array using class index
        #         if object_name1=='helmet':
        #             helmet=1        
        
        # mask_interpreter.set_tensor(input_details[0]['index'],input_data)
        # mask_interpreter.invoke()

        # # Retrieve detection results
        # mask_boxes = mask_interpreter.get_tensor(mask_output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        # mask_classes = mask_interpreter.get_tensor(mask_output_details[1]['index'])[0] # Class index of detected objects
        # mask_scores = mask_interpreter.get_tensor(mask_output_details[2]['index'])[0] # Confidence of detected objects
        # #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
        # # print('mask_scores:',mask_scores)

        # # Loop over all detections and draw detection box if confidence is above minimum threshold
        # for i in range(len(mask_scores)):
        #     if ((mask_scores[i] > min_conf_threshold) and (mask_scores[i] <= 1.0)):

        #         '''
        #         # Get bounding box coordinates and draw box
        #         # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
        #         mask_ymin = int(max(1,(mask_boxes[i][0] * imH)))
        #         mask_xmin = int(max(1,(mask_boxes[i][1] * imW)))
        #         mask_ymax = int(min(imH,(mask_boxes[i][2] * imH)))
        #         mask_xmax = int(min(imW,(mask_boxes[i][3] * imW)))

        #         cv2.rectangle(frame, (mask_xmin,mask_ymin), (mask_xmax,mask_ymax), (10, 255, 0), 2)

        #         # Draw label
        #         object_name2 = mask_labels[int(mask_classes[i])] # Look up object name from "labels" array using class index
        #         mask_label = '%s: %d%%' % (object_name2, int(mask_scores[i]*100)) # Example: 'person: 72%'
        #         mask_labelSize, mask_baseLine = cv2.getTextSize(mask_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        #         mask_label_ymin = max(mask_ymin, mask_labelSize[1] + 10) # Make sure not to draw label too close to top of window
        #         cv2.rectangle(frame, (mask_xmin, mask_label_ymin-mask_labelSize[1]-10), (mask_xmin+mask_labelSize[0], mask_label_ymin+mask_baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        #         cv2.putText(frame, mask_label, (mask_xmin, mask_label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        #         '''
        #         object_name2 = mask_labels[int(mask_classes[i])] # Look up object name from "labels" array using class index  
        #         if object_name2=='with_mask':
        #             mask=1
        
        # door_interpreter.set_tensor(input_details[0]['index'],input_data)
        # door_interpreter.invoke()

        # # Retrieve detection results
        # door_boxes = door_interpreter.get_tensor(door_output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        # door_classes = door_interpreter.get_tensor(door_output_details[1]['index'])[0] # Class index of detected objects
        # door_scores = door_interpreter.get_tensor(door_output_details[2]['index'])[0] # Confidence of detected objects
        # #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
        # # print('door_scores:',door_scores)

        # # Loop over all detections and draw detection box if confidence is above minimum threshold
        # for i in range(len(door_scores)):
        #     if ((door_scores[i] > min_conf_threshold) and (door_scores[i] <= 1.0)):

        #         '''
        #         # Get bounding box coordinates and draw box
        #         # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
        #         door_ymin = int(max(1,(door_boxes[i][0] * imH)))
        #         door_xmin = int(max(1,(door_boxes[i][1] * imW)))
        #         door_ymax = int(min(imH,(door_boxes[i][2] * imH)))
        #         door_xmax = int(min(imW,(door_boxes[i][3] * imW)))

        #         cv2.rectangle(frame, (door_xmin,door_ymin), (door_xmax,door_ymax), (10, 255, 0), 2)

        #         # Draw label
        #         object_name2 = door_labels[int(door_classes[i])] # Look up object name from "labels" array using class index
        #         door_label = '%s: %d%%' % (object_name2, int(door_scores[i]*100)) # Example: 'person: 72%'
        #         door_labelSize, door_baseLine = cv2.getTextSize(door_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        #         door_label_ymin = max(door_ymin, door_labelSize[1] + 10) # Make sure not to draw label too close to top of window
        #         cv2.rectangle(frame, (door_xmin, door_label_ymin-door_labelSize[1]-10), (door_xmin+door_labelSize[0], door_label_ymin+door_baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        #         cv2.putText(frame, door_label, (door_xmin, door_label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        #         '''
        #         object_name2 = door_labels[int(door_classes[i])] # Look up object name from "labels" array using class index  
        #         if object_name2=='open':
        #             door=1
        
        # #loitering 
        # loitering = loitering_prediction(loitering_interpreter,loitering_input_details,loitering_output_details,frame,CLASSIFIER_CKPT)

        animal_interpreter.set_tensor(input_details[0]['index'],input_data)
        animal_interpreter.invoke()

        # Retrieve detection results
        animal_boxes = animal_interpreter.get_tensor(animal_output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        animal_classes = animal_interpreter.get_tensor(animal_output_details[1]['index'])[0] # Class index of detected objects
        animal_scores = animal_interpreter.get_tensor(animal_output_details[2]['index'])[0] # Confidence of detected objects
        animal_output_data = animal_interpreter.get_tensor(animal_output_details[0]['index'])
        num = animal_interpreter.get_tensor(animal_output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
        # print('animal_scores:',animal_scores)
        ac=0
        # print(num)
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(animal_scores)):
            if ((animal_scores[i] > min_conf_threshold) and (animal_scores[i] <= 1.0)):

                '''
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                mask_ymin = int(max(1,(mask_boxes[i][0] * imH)))
                mask_xmin = int(max(1,(mask_boxes[i][1] * imW)))
                mask_ymax = int(min(imH,(mask_boxes[i][2] * imH)))
                mask_xmax = int(min(imW,(mask_boxes[i][3] * imW)))

                cv2.rectangle(frame, (mask_xmin,mask_ymin), (mask_xmax,mask_ymax), (10, 255, 0), 2)

                # Draw label
                object_name2 = mask_labels[int(mask_classes[i])] # Look up object name from "labels" array using class index
                mask_label = '%s: %d%%' % (object_name2, int(mask_scores[i]*100)) # Example: 'person: 72%'
                mask_labelSize, mask_baseLine = cv2.getTextSize(mask_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                mask_label_ymin = max(mask_ymin, mask_labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (mask_xmin, mask_label_ymin-mask_labelSize[1]-10), (mask_xmin+mask_labelSize[0], mask_label_ymin+mask_baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, mask_label, (mask_xmin, mask_label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                '''
                object_name3 = animal_labels[int(animal_classes[i])] # Look up object name from "labels" array using class index  
                if object_name3=='animal':
                    animal=1
                    ac+=1

        crowd_interpreter.set_tensor(crowd_input_details[0]['index'],crowd_input_data)
        crowd_interpreter.invoke()
        crowd_output_data = crowd_interpreter.get_tensor(crowd_output_details[0]['index'])
        # print('crowd_output_data shape:',crowd_output_data[0][0].shape)

        nop=0
        for det in crowd_output_data[0]:
            if det[5]>0.5 and det[6]==1:
                nop+=1
        if nop>nop_thresh:
            crowd=1
        # print(nop)
        curr_time=(time.time() - start_time)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        alert_dict={'time':current_time,'animal':ac,'crowd':nop}
        # print(alert_dict)
        with open('detections.csv', 'a') as f:
            f.write('{}\n'.format(alert_dict)) 
        # bytes_to_send=json.dumps(alert_dict).encode('utf-8')
        # s.sendall(bytes_to_send)

        '''
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
        '''
        # fps.stop()
        # print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# stop the timer and display FPS information
# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# Clean up
cv2.destroyAllWindows()
# videostream.stop()
videostream.release()
