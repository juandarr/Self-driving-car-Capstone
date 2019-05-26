from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf


class TLClassifier(object):
    def __init__(self):
        ## Test model
        path_to_graph = r'traffic_light.pb' 
        path_to_labels = r'udacity_label_map.pbtxt'
        num_classes = 4
        IMAGE_SIZE = (12, 8)

        self.graph = load_graph(path_to_graph)

        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        print(category_index)

    def load_graph(graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph


    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.graph.as_default():
            with tf.Session(graph= self.graph) as sess:
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detect_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detect_classes = detection_graph.get_tensor_by_name('detection_classes:0')
 
                
                image_np = load_image_into_numpy_array(image)
                image_expanded = np.expand_dims(image_np, axis=0)
                
                (scores, classes) = sess.run(
                    [detect_scores, detect_classes],
                    feed_dict={image_tensor: image_expanded})
                
                print('SCORES')
                print(scores[0])
                print('CLASSES')
                print(classes[0])
        return TrafficLight.UNKNOWN
