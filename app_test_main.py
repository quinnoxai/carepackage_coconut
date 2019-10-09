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
final_list = []
final_green_list = []
final_brown_list = []


@app.route("/")
def start_page():
    """

    :return: render_template : Display index.html page
    """
    return render_template('index1.html')


filename = 'finalized_model.sav'
# joblib.dump(reg, filename)
reg = joblib.load(filename)


def detect_object(file, received_param, reg):
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
    file_split = (os.path.splitext(os.path.basename(file))[0])
    final_file_split_1 = file.rpartition('.')[0]
    final_file_split = final_file_split_1.rpartition('.')[0]

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
                if scores is None or final_score[i] > 0.01:
                    count = count + 1
                    if final_classes[i] == 1.0:
                        count_green = count_green + 1
                    # elif final_classes[i] == 2.0:
                    # count_yellow = count_yellow + 1
                    elif final_classes[i] == 3.0 or final_classes[i] == 2.0:
                        count_brown = count_brown + 1

    # count_brown_final = count_brown + count_yellow
    # count_mature_new = count_green + count_brown_final

    # Draw the results of the detection (aka 'visulaize the results')
    print("Overall Count", count)
    print("GreenCount", count_green)
    print("BrownCount", count_brown)
    sec_df = pd.read_csv("carePackage.csv")

    trainX = sec_df.drop('actual_nuts', axis=1)
    trainY = sec_df['actual_nuts']

    # reg = RandomForestRegressor()
    # reg.fit(trainX , trainY )
    # filename = 'finalized_model.sav'
    # joblib.dump(reg, filename)
    # reg = joblib.load(filename)

    # please install this below
    from treeinterpreter import treeinterpreter as ti
    # fit a scikit-learn's regressor model
    # rf = RandomForestRegressor()
    # rf.fit(trainX, trainY)

    # bias is intercepet here
    # add bias to each contribution, you will get three contibutions one for brown and other for green coconut and one for image count
    # count_green = green contribution + bias
    # same for brown etc...

    green = count_green
    brown = count_brown
    tree_no = final_file_split
    testX = np.array([[1, green, brown]])
    testX = testX.reshape(1, -1)

    prediction, bias, contributions = ti.predict(reg, testX)
    contributions = pd.DataFrame(contributions)
    print("Contri",contributions)
    contributions.columns = ['no', 'green', 'brown']
    contri_green = int(contributions['green'].iloc[0])
    pred_green = int(green) + contri_green + int(bias / 2)
    final_green = int(pred_green)
    print("pred_green", final_green)
    contri_brown = int(contributions['brown'].iloc[0])
    pred_brown = int(brown) + contri_brown + int(bias / 2)
    final_brown = int(pred_brown)
    print("pred_brown", final_brown)

    final_matured_nut = final_green + final_brown
    print("final_matured_nuts", final_matured_nut)
    final_green_list.append(final_green)
    final_green_count = sum(final_green_list)
    final_brown_list.append(final_brown)
    final_brown_count = sum(final_brown_list)
    final_list.append(final_matured_nut)
    print("final_list", final_list)
    final_matured_nuts = sum(final_list)
    sec_df_new = pd.read_csv("Book1.csv")
    actual_count_nuts = sec_df_new[sec_df_new['tree_no'] == tree_no]['actual']
    mape = np.mean(np.abs(actual_count_nuts - final_matured_nuts) / actual_count_nuts)
    image_np1 = image_np
    print("The approximate count of Coconuts are", final_matured_nuts)
    received_param["green"] = final_green
    received_param["brown"] = final_brown
    received_param["total"] = final_matured_nut
    

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


@app.route("/takeFiles", methods=['GET', 'POST'])
def post():
    """

    :return: render_template : index.html file(Display output)
    """
    global pred_green, pred_brown, accuracy
    if request.method == 'POST':
        file = request.files.getlist('exampleInputFile')
        accuracy = request.form['myRange']
        filenames = []
        for f in file:
            filename = secure_filename(f.filename)
            print("filename",filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    received_param = {"count": 0, "count_brown": 0, "count_brown_new": 0, "count_yellow": 0, "count_green": 0,
                      "final_brown": 0,
                      "final_green": 0, "final_matured_nuts": 0, "green": 0, "brown": 0, "total": 0, "result_print": []}
    dataf = pd.DataFrame(columns=['tree_no', 'Green Nuts', 'Brown Nuts', 'Total Nuts'])
    print(dataf)

    for image_name in filenames:
        print("Image Name", image_name)
        final_file_split_1 = image_name.rpartition('.')[0]
        final_file_split_2 = final_file_split_1.rpartition('.')[0]
        final_file_split = final_file_split_2.rpartition('.')[0]
        print("Final File Name", final_file_split)
        samplef = pd.DataFrame(columns=['tree', 'tree_no', 'Green Nuts', 'Brown Nuts', 'Total Nuts'])

        print(os.path.join(app.config['UPLOAD_FOLDER'], image_name))
        returned_result = detect_object(os.path.join(app.config['UPLOAD_FOLDER'], image_name), received_param, reg)

        os.remove("{}/{}".format(UPLOAD_FOLDER, image_name))
        received_param = dict(returned_result)
        tree_single_brown = received_param["brown"]
        tree_single_green = received_param["green"]
        total_nuts_tree = received_param["total"]
        print("tree single brown", tree_single_brown)
        samplef.loc[0, 'tree'] = final_file_split
        samplef.loc[0, 'tree_no'] = image_name
        samplef.loc[0, 'Brown Nuts'] = tree_single_brown
        samplef.loc[0, 'Green Nuts'] = tree_single_green
        samplef.loc[0, 'Total Nuts'] = total_nuts_tree
        dataf = dataf.append(samplef)
    print("Total list : ", dataf)

    dataf['tree_no'] = dataf['tree_no'].astype('category')
    dataf['tree'] = dataf['tree'].astype('category')
    dataf['Brown Nuts'] = dataf['Brown Nuts'].astype('int')
    dataf['Green Nuts'] = dataf['Green Nuts'].astype('int')
    dataf['Total Nuts'] = dataf['Total Nuts'].astype('int')

    print(dataf.dtypes)
    dataf = dataf.groupby('tree').mean()
    dataf = (round(dataf))
    dataf = dataf.reset_index()
    print("dataf['Total Nuts']",dataf['Total Nuts'])
    sum_green = dataf['Green Nuts'].sum()
    sum_brown = dataf['Brown Nuts'].sum()
    sum_final=dataf['Total Nuts'].sum()
    print("dataf['Total Nuts']",sum_final)
    received_param["final_green"] = sum_green
    received_param["final_brown"] = sum_brown
    received_param["final_matured_nuts"] = sum_final
    print(dataf)
    os.chdir("/home/sherlock/tensorflow2_main/models/research/object_detection/static")
    dataf.to_csv("final_data_output.csv")

    return render_template("index1.html", result_print=received_param["result_print"], count=received_param["count"],
                           count_green=received_param["count_green"], count_yellow=received_param["count_yellow"],
                           count_brown=received_param["count_brown"], final_brown=received_param["final_brown"],
                           final_green=received_param["final_green"],
                           final_matured_nuts=received_param["final_matured_nuts"])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8189, threaded=True, debug=True)
