# Import 
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp

# Keypoints using MP Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_pose(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    )

# Extract Keypoint 
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
    results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return np.concatenate([pose])

DATA_PATH = os.path.join('MP_Data_V2') # export data(numpy array)
actions = np.array(['left','walk','right']) # detect data

no_sequences = 795 # number videos worth of data
sequence_length = 60 # 60 frames in length

# import for create labels and features
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

label_map = {label:num for num, label in enumerate(actions)}
label_map
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Build and Train LSTM neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
log_dir = os.path.join('Logs_V3')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(60,132)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Predictions
res = model.predict(X_test)
[np.argmax(res[4])]
actions[np.argmax(y_test[4])]

# load weights
model.load_weights('action_V3.h5')

# Evaluation using confusion matrix and accuracy
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
multilabel_confusion_matrix(ytrue, yhat)
accuracy_score(ytrue, yhat)

# For real time
colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,305+num*40), (int(prob*100), 335+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num]+f" {int(prob*100)}%", (0, 330+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

sequence = []
sentence = []
threshold = 0.9 

num_left = 0 # collect_number_of_left
num_right = 0 # collect_number_of_right
count_left = 0 # count_frame_left
count_right = 0 # count_frame_left

# First camera(camera in macbook)
cap1 = cv2.VideoCapture(1)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Second camera(first webcam)
cap2 = cv2.VideoCapture(0)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 180)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 148)

# Third camera(second webcam)
cap3 = cv2.VideoCapture(2)
cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 180)
cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 148)

# mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap1.isOpened():

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        image1, results1 = mediapipe_detection(frame1, holistic) # make detections

        # Predicition logic
        keypoints = extract_keypoints(results1)
        sequence.append(keypoints)
        sequence = sequence[-60:]

        # dispaly message 
        cv2.imshow('left',frame2)
        cv2.imshow('right',frame3)

        if len(sequence) == 60:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            image1 = prob_viz(res, actions, image1, colors)

            if(int(res[0]*100) >= 90): # check percentage of left than more 90% 
                count_left +=1 # accumulate amount frame
                # every 15 frames that will make picture --> {count_left}_left.jpg
                if (count_left%15 ==0): # 
                    cv2.imwrite("left/camera_"+ str(num_left) +"_left.jpg", frame2)
                    num_left += 1
            elif(int(res[2]*100) >= 90): # check percentage of right than more 90% 
                count_right +=1 # accumulate amount frame
                # every 15 frames that will make picture --> {count_right}_right.jpg
                if (count_right%15 ==0):
                    cv2.imwrite("right/camera_"+ str(num_right) +"_right.jpg", frame3)
                    num_right += 1

        cv2.imshow('middle', image1)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap1.release()
    cap2.release()
    cap3.release()
    cv2.destroyAllWindows()