# import tensorflow as tf
import sys
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

loc = os.path.dirname(os.path.realpath(__file__))
label_path = os.path.join(loc, "nn_files/retrained_labels.txt")
graph_path = os.path.join(loc, "nn_files/retrained_graph.pb")

# change this as you see fit
image_path = sys.argv[1]

# Read in the image_data
image_data = tf.gfile.GFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip()
               for line in tf.gfile.GFile(label_path)]


def predict(image_path, verbose=False):
    image_path = os.path.abspath(image_path)
    # Read in the image_data
    image_data = tf.gfile.GFile(image_path, 'rb').read()

    # Unpersists graph from file
    with tf.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(
            softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        if verbose:
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))

        return label_lines[top_k[0]]


print(predict(image_path))
