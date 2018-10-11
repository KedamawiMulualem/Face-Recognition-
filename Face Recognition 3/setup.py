import tensorflow as tf
import numpy as np
import dlib
import cv2
import matplotlib.image as mpimg

# Hyperparameter
hidden_size1 = 200
hidden_size2 = 100


# Create placeholders
X  = tf.placeholder(tf.float32, [None, 128])
Y_ = tf.placeholder(tf.float32, [None, 2])
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

# Create variables
w1 = tf.Variable(tf.truncated_normal([128, hidden_size1], stddev=0.1), tf.float32)
b1 = tf.Variable(tf.ones([hidden_size1])/10)

w2 = tf.Variable(tf.truncated_normal([hidden_size1, hidden_size2], stddev=0.1), tf.float32)
b2 = tf.Variable(tf.ones([hidden_size2])/10)

w3 = tf.Variable(tf.truncated_normal([hidden_size2, 2], stddev=0.1), tf.float32)
b3 = tf.Variable(tf.ones([2])/10)

Y1 = tf.matmul(X, w1) + b1
Y1 = tf.nn.relu(Y1)

Y2 = tf.matmul(Y1, w2) + b2
Y2 = tf.nn.relu(Y2)

Ydrop = tf.nn.dropout(Y2, pkeep)

Ylogits = tf.matmul(Ydrop, w3) + b3
Y = tf.nn.softmax(Ylogits)

# Loss (Cross-entropy)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)

# Optimizer
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

# Accuracy
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Initializer
init = tf.global_variables_initializer()
saver = tf.train.Saver()


face_detector = dlib.get_frontal_face_detector()

shape_predictor = dlib.shape_predictor('Files/shape_predictor_68_face_landmarks.dat')

face_recognition_model = dlib.face_recognition_model_v1('Files/dlib_face_recognition_resnet_model_v1.dat')


known_faces = []

im = mpimg.imread("Khoi1.jpg")
faces = face_detector(im, 1)
shapes = [shape_predictor(im, face) for face in faces]
if len(shapes) != 0:
    face_ids = [face_recognition_model.compute_face_descriptor(im, shape, 1) for shape in shapes]
    face_ids_array = np.array(face_ids)
    face_ids_list = face_ids_array.tolist()[0]
    known_faces.append(face_ids_list)

im = mpimg.imread("KD1.jpg")
faces = face_detector(im, 1)
shapes = [shape_predictor(im, face) for face in faces]
if len(shapes) != 0:
    face_ids = [face_recognition_model.compute_face_descriptor(im, shape, 1) for shape in shapes]
    face_ids_array = np.array(face_ids)
    face_ids_list = face_ids_array.tolist()[0]
    known_faces.append(face_ids_list)
    

    
print ("Known faces: ", len(known_faces))

def detect_face(im, sess, known_faces):
    images_test = []
    faces = face_detector(im, 1)
    shapes = [shape_predictor(im, face) for face in faces]
    face_ids = [face_recognition_model.compute_face_descriptor(im, shape, 1) for shape in shapes]
    face_ids_array = np.array(face_ids)
    
#     in_test_set = True
#     unknown_faces = []
    
#     for i, face in enumerate(face_ids_array):
#         for known_face in known_faces:
#             thresh_hold = np.linalg.norm(known_face - face)
#             if thresh_hold > 0.45:
#                 in_test_set = False
#             else:
#                 in_test_set = True
#                 break
        
#         if not in_test_set:
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             cv2.putText(im,"Unknown Face",(faces[i].left()-50,faces[i].top()-50), font, 2,(0,255,0),2,cv2.LINE_AA)

#             cv2.rectangle(im,(faces[i].left(),faces[i].top()),(faces[i].right(),faces[i].bottom()),(255,0,0),2)
#             unknown_faces.append(i)
            
        



    face_ids_list = face_ids_array.tolist()
    
#     print (len(face_ids_list))
#     print ("ID: ", unknown_faces)
    
#     if len(unknown_faces) != 0 and len(face_ids_list) != 0:
#         for i, index in enumerate(unknown_faces):
#             face_ids_list.pop(index - i)

    
    [images_test.append(face_id) for face_id in face_ids_list]
    
    if len(images_test) != 0:
        test_input = np.array(images_test)

        test_dict = {X: test_input, pkeep: 0.9}

        Y_predict = sess.run(Y, feed_dict=test_dict)

        for i, result in enumerate(Y_predict):
            confidence = round(result[np.argmax(result)] * 100, 2)
            if np.argmax(result) == 0:
                if np.linalg.norm(test_input[i] - known_faces[0]) < 0.45:
                    predicted_face = "Khoi face " + str(confidence) + " %"
                else:
                    predicted_face = "Unknown"
                    
            elif np.argmax(result) == 1:
                if np.linalg.norm(test_input[i] - known_faces[1]) < 0.45:
                    predicted_face = "KD face " + str(confidence) + " %"
                else:
                    predicted_face = "Unknown"

            font = cv2.FONT_HERSHEY_SIMPLEX
            
            if predicted_face == "Unknown":
                cv2.putText(im,predicted_face,(faces[i].left()-50,faces[i].top()-50), font, 2,(0,0,255),2,cv2.LINE_AA)
            else:
                cv2.putText(im,predicted_face,(faces[i].left()-50,faces[i].top()-50), font, 2,(0,255,0),2,cv2.LINE_AA)
            cv2.rectangle(im,(faces[i].left(),faces[i].top()),(faces[i].right(),faces[i].bottom()),(255,0,0),2)
    
    return im

cap = cv2.VideoCapture(0)

with tf.Session() as sess:
    saver.restore(sess, "Checkpoint/model.ckpt")
    print("Model restored.")

    while True:
        ret, frame = cap.read()
        cv2.imshow('Live stream', detect_face(frame, sess, known_faces))
        if cv2.waitKey(1) == 13:
            break

cap.release()
cv2.destroyAllWindows() 