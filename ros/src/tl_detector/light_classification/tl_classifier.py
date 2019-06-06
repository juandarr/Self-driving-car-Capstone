from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import datetime 
import cv2



class TLClassifier(object):
    def __init__(self):
        path_to_graph = r'light_classification/ssd_mobilenet_v1_optimized.pb' 

        self.image_tensor = None
        self.boxes = None
        self.scores = None
        self.classes = None
        self.num_detections = None
        self.sess = None

        self.c = ['RED', 'GREEN', 'YELLOW']

        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name(
                'num_detections:0')

        self.sess = tf.Session(graph=self.graph)
        
        self.camera_feed = False

        self.threshold = .3


    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name(
                'num_detections:0')

        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.graph.as_default():
            
            image_expanded = np.expand_dims(image, axis=0)
            
            start =  datetime.datetime.now()
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})
            end = datetime.datetime.now()

            c = end - start
            #print('Inference time: ', c.total_seconds())

        score_f = scores[0]
        class_f = classes[0].astype(np.uint8)
        #print('SCORES: ', score_f)
        #print('CLASSES: ', class_f)
        
        # Show camera feed when flag is true
        if self.camera_feed:
            cv2.imshow('my webcam', image)
            cv2.waitKey(1)
        
        
        if score_f[0] > self.threshold:
            if class_f[0] == 1:
                print('GREEN light detected.')
                return TrafficLight.GREEN
            elif class_f[0] == 2:
                print('RED light detected.')
                return TrafficLight.RED
            elif class_f[0] == 3:
                print('YELLOW light detected.')
                return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN
