
from flask import Flask, render_template, request, make_response
from werkzeug import secure_filename
from sklearn.ensemble import RandomForestRegressor
import base64
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import os
import sys
import PIL
from io import BytesIO
from numpy import arange, sin, pi
import tensorflow as tf
from utils import label_map_util
from PIL import Image
from sklearn.externals import joblib


sys.path.append("..")
LARGE_FONT = ("Verdana", 12)
IMAGE_SIZE = (12, 8)

UPLOAD_FOLDER = 'upload'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def start_page():
    """

    :return: render_template : Display index.html page
    """
    return render_template('index1.html')

filename = 'finalized_model.sav'
#joblib.dump(reg, filename)
reg = joblib.load(filename)


def detect_object(file, received_param,reg):
    """
    This function does
    :param file: input images
    :param received_param: Dictionary use to initialize variable such as list_result,count,count_brown,count_yellow,count_green

    :return: received_param : Dictionary
    """
    list_result = received_param["result_print"]
    count = received_param["count"]
    count_brown = received_param["count_brown"]
    count_yellow = received_param["count_yellow"]
    count_green = received_param["count_green"]
    print("file", file)
    file_split = (os.path.splitext(os.path.basename(file))[0])
    print("File", file_split)
    final_file_split_1 = file_split.rpartition('.')[0]
    print("File Name", final_file_split_1)
    # print("Hello",a)
    final_file_split = final_file_split_1.rpartition('.')[0]
    print("Final File Name", final_file_split)
    global image_np1
    global count1
    global lbl
    image_np1 = 0
    # What model to download.
    MODEL_NAME = 'inference_graph'
    MODEL_FILE = MODEL_NAME + '.tar.gz'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('training', 'labelmap.pbtxt')

    NUM_CLASSES = 3

    # ## Download Model

    # ## Load a (frozen) Tensorflow model into memory.

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # ## Helper code

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    # # Detection
    TEST_IMAGE_PATHS = [file]
    # Size, in inches, of the output images.

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in TEST_IMAGE_PATHS:
                image = PIL.Image.open(image_path, "r")
                # image=Image.load(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
            final_score = np.squeeze(scores)
            final_classes = np.squeeze(classes)
            for i in range(100):
                if scores is None or final_score[i] > 0.1:
                    count = count + 1
                    if final_classes [i]== 1.0:
                        count_green = count_green + 1

                    elif final_classes[i] == 2.0:
                        count_yellow = count_yellow + 1
                    elif final_classes[i] == 3.0:
                        count_brown = count_brown + 1

    count_brown_final = count_brown + count_yellow
    count_mature_new = count_green + count_brown_final

# Draw the results of the detection (aka 'visulaize the results')
    print("Overall Count", count)
    print("GreenCount",count_green)
    print("YellowCount",count_yellow)
    print("BrownCount",count_brown)
    print("BrownYellowCount", count_brown_final)
    print("MatureCount", count_mature_new)

    image_np1 = image_np
    print("The approximate count of Coconuts are",count_mature_new)
    received_param["count"]=count_mature_new
    f = Figure(figsize=(IMAGE_SIZE))

    a = f.add_subplot(111)

    t = arange(0.0, 3.0, 0.01)
    s = sin(2 * pi * t)
    a.imshow(image_np1)

    canvas = FigureCanvas(f)

    png_output = BytesIO()
    canvas.print_png(png_output)
    png_output.seek(0)
    png_response = base64.b64encode(png_output.getvalue())
    result1 = str(png_response)[2:-1]
    list_result.append(result1)
    received_param["result"] = list_result
    return received_param


@app.route("/takeFiles",methods=['GET', 'POST'])
def post():
    """

    :return: render_template : index.html file(Display output)
    """
    global pred_green,pred_brown,accuracy
    if request.method=='POST':
     file=request.files.getlist('exampleInputFile')
     accuracy = request.form['myRange']
     filenames = []
     for f in  file:
         filename = secure_filename(f.filename)
         f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
         filenames.append(filename)
    received_param={"count":0,"count_brown":0,"count_brown_new":0,"count_yellow":0,"count_green":0,"result_print":[]}

    for f in filenames:
        print(f)

        print(os.path.join(app.config['UPLOAD_FOLDER'],f))
        returned_result=detect_object(os.path.join(app.config['UPLOAD_FOLDER'],f),received_param,reg)
        os.remove("{}/{}".format(UPLOAD_FOLDER, f))
        received_param=dict(returned_result)
    return render_template("index1.html", result_print=received_param["result_print"], count=received_param["count"],
                           count_green=received_param["count_green"], count_yellow=received_param["count_yellow"],
                           count_brown=received_param["count_brown"])


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8189,threaded=True,debug=True)

