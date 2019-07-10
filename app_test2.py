from flask import Flask, render_template, request, make_response
from wtforms import Form, FileField, SubmitField
from werkzeug import secure_filename
from werkzeug.datastructures import CombinedMultiDict
import tempfile

import base64
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import sys
import numpy as np
import random
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import numpy
import PIL
import zipfile
from io import StringIO, BytesIO
from PIL import ImageDraw
from base64 import b64encode
from numpy import array, arange, sin, pi
import tensorflow as tf
import urllib.request
import urllib.parse
from matplotlib import pyplot as plt
from utils import label_map_util
from PIL import Image
from utils import visualization_utils as vis_util

sys.path.append("..")
LARGE_FONT = ("Verdana", 12)
IMAGE_SIZE = (12, 8)

UPLOAD_FOLDER = 'upload'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# class PhotoForm(Form):
#     image_path=FileField()
# import secondary_model as sm
# import detection_image as fdm

# count,count_green,count_yellow,count_brown,count_brown_new,final_file_split = fdm.return_primary_model_output()
# final_matured_nuts,brown_coconuts,green_coconuts = sm.sec_model(count_green, count, final_file_split, count_brown)
# print("___Brown___",brown_coconuts)
@app.route("/")
def start_page():
    """

    :return: render_template : Display index.html page
    """
    return render_template('index.html')


def detect_object(file, received_param):
    """
    This function does
    :param file: input images
    :param received_param: Dictionary use to initialize variable
    :return: received_param : Dictionary
    """
    list_result = received_param["result_print"]
    count = received_param["count"]
    count_brown = received_param["count_brown"]
    count_yellow = received_param["count_yellow"]
    count_green = received_param["count_green"]
    # count_brown_new = received_param["count_brown_new"]
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

    # In[5]:

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

    # In[6]:

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # ## Helper code

    # In[7]:

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    # # Detection

    # In[8]:

    # For the sake of simplicity we will use only 2 images:
    # image1.jpg
    # image2.jpg
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = 'test_images'
    # TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1,3) ]
    # TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'chilli7.jpg')]
    TEST_IMAGE_PATHS = [file]
    # Size, in inches, of the output images.

    # In[9]:

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
                if scores is None or final_score[i] > 0.001:
                    count = count + 1
                    received_param["count"] = count
                    if final_classes[i] == 1.0:
                        count_green = count_green + 1

                    elif final_classes[i] == 2.0:
                        count_yellow = count_yellow + 1

                    elif final_classes[i] == 3.0:
                        count_brown = count_brown + 1

    count_brown_final = count_brown + count_yellow
    count_mature_new = count_green + count_brown_final
    # Draw the results of the detection (aka 'visulaize the results')
    #print("Overall Count", count)
    print("GreenCount", count_green)
    print("YellowCount", count_yellow)
    print("BrownCount",count_brown)
    print("BrownYellowCount", count_brown_final)
    print("MatureCount", count_mature_new)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.80)

    sec_df = pd.read_csv("D:/Marico/tensorflow2_main/models/research/object_detection/carePackage.csv")
    #print(sec_df)

    X = sec_df.drop('actual_nuts',axis =1)
    Y = sec_df['actual_nuts']

    print("columns",X.columns)

    reg = LinearRegression().fit(X, Y)
    print("Score:",reg.score(X, Y))
    print("coeff;",reg.coef_)
    print("intercept",reg.intercept_)

    green = count_green
    brown = count_brown_final
    tree_no = final_file_split
    #tree_no = fdm.post()
    #green = int(green)
    #brown = int(brown)
    #tree_no = int(tree_no)
    #brown = fdm.count_brown_new
    pred_count = reg.predict(np.array([[1, green, brown]]))
    print("Tree No:",tree_no)
    print("Final count",pred_count)
    mean_intercept = reg.intercept_/2
    pred_green = green + mean_intercept
    final_green = int(pred_green)
    pred_brown = brown + mean_intercept
    final_brown = int(pred_brown)
    print("pred_green",final_green)
    received_param["count_green"]=final_green
    print("pred_brown",final_brown)
    received_param["count_brown"]= final_brown
    final_matured_nuts = final_green+final_brown
    print("final_matured_nuts",final_matured_nuts)
    sec_df_new = pd.read_csv("D:/Marico/tensorflow2_main/models/research/object_detection/Book1.csv")
    print("sec_df_new",sec_df_new)
    print("validation",sec_df_new.columns)
    actual_count_nuts = sec_df_new[sec_df_new['tree_no'] == tree_no]['actual']
    print("actual_count_nuts!!!",actual_count_nuts)
    print("actual_count_nuts type!!!",actual_count_nuts)
    #actual = 49
    mape = np.mean(np.abs(actual_count_nuts - final_matured_nuts)/actual_count_nuts)

    print("actual_count_nuts",actual_count_nuts)
    print("final_matured_nuts",final_matured_nuts)

    print(mape)
    print("Acc %", (1 - mape)*100)
    image_np1 = image_np
    #count1 = str(count)
    #total_matured_nuts = round(final_matured_nuts)
    #total_matured_nuts=int(final_matured_nuts)
    print("The approximate count of Coconuts are",final_matured_nuts)
    #print(total_matured_nuts)
    received_param["count"]=final_matured_nuts
    f = Figure(figsize=(IMAGE_SIZE))

    a = f.add_subplot(111)

    t = arange(0.0, 3.0, 0.01)
    s = sin(2 * pi * t)
    a.imshow(image_np1)

    canvas = FigureCanvas(f)

    png_output = BytesIO()
    canvas.print_png(png_output)
    # plt.savefig(png_output, format='png')
    png_output.seek(0)
    png_response = base64.b64encode(png_output.getvalue())
    result1 = str(png_response)[2:-1]
    list_result.append(result1)
    received_param["result"] = list_result
    # print("Final Count:",received_param["result"])
    # response = make_response(png_output.getvalue())
    # response.headers['Content-Type'] = 'image/png'
    return received_param


@app.route("/takeFiles",methods=['GET', 'POST'])
def post():
    """

    :return: render_template : index.html file(Display output)
    """
    global pred_green,pred_brown
    if request.method=='POST':
     file=request.files.getlist('exampleInputFile')
     accuracy = request.form['myRange']
     filenames = []
     for f in  file:
         filename = secure_filename(f.filename)
         f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
         filenames.append(filename)

    # result1='edhggg'
    received_param={"count":0,"count_brown":0,"count_brown_new":0,"count_yellow":0,"count_green":0,"result_print":[]}

    for f in filenames:
        print(f)

        print(os.path.join(app.config['UPLOAD_FOLDER'],f))
        returned_result=detect_object(os.path.join(app.config['UPLOAD_FOLDER'],f),received_param)
        os.remove("{}/{}".format(UPLOAD_FOLDER, f))
        received_param=dict(returned_result)

        # print("base64",received_param[""])
    return render_template("index.html", result_print=received_param["result_print"], count=received_param["count"],
                           count_green=received_param["count_green"], count_yellow=received_param["count_yellow"],
                           count_brown=received_param["count_brown"])


if __name__ == '__main__':
    app.run(port=8080, debug=True)
