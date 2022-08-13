#!/usr/bin/python
import json
import os
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from skimage.io import imsave
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

def dice_metric(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def jaccard_metric(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    Jaccard = (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)
    return Jaccard

def jaccard_loss(y_true, y_pred):
    return 1-jaccard_metric(y_true, y_pred)

def display_imarge(img):
    imarge = []
    for row in img:
        imarge_row = []
        for pix in row:
            # get idx max of 8 class ?
            if pix[0] > 0.5: merge = .1
            elif pix[1] > 0.5: merge = .2
            elif pix[2] > 0.5: merge = .3
            elif pix[3] > 0.5: merge = .4
            elif pix[4] > 0.5: merge = .5
            elif pix[5] > 0.5: merge = .6
            elif pix[6] > 0.5: merge = .7
            else: merge = .8
            imarge_row.append(merge) 
        imarge.append(imarge_row)
    imarge = np.array(imarge)
    return imarge

# ==============================
# ========= FLASK ==============
# ==============================  
from flask import Flask, render_template, url_for, request
import requests
import json

app = Flask(__name__)

@app.route('/')
def home():
  concaten = ""
  liste = []
  for parent, dnames, fnames in os.walk("static/"):
    for fname in fnames:
      filename = os.path.join(fname)
      liste.append(filename)   
  return render_template('machines.html', liste=liste)

@app.route('/<path:path>')
def machines(path):
    url = url_for("static", filename=path)
    img_arr_np = call_API_with_img_id(path)
    imsave('static/MASK.png',img_arr_np)
    #imsave('static/MASK.png',img_arr_np[0])
    return render_template('contenu.html', contenu=[url]) 



from segmentation_models.metrics import iou_score
custom_object = {"jaccard_loss": jaccard_loss, "dice_metric":dice_metric, "iou_score":iou_score }
model = load_model('PSP', custom_objects = custom_object)
#model = load_model('testModelsimple20Mo')

def call_API_with_img_id(fileName = 'berlin_000000_000019_leftImg8bit.png'):
    url = "http://192.168.1.12:8080/api_pred2"
    myobj = {}

    input_images = load_img('static/'+fileName, color_mode='rgb', target_size= (384,384))
    input_images = np.array(input_images)
    input_images_np = np.array([input_images])
    myobj['image'] = input_images_np.tolist()

    
    x = requests.post(url, json = myobj)
    jsonString = json.dumps(x.json())
    #recup
    img_arr = json.loads(jsonString)
    img_arr_np = np.array(img_arr)
    imarge = display_imarge(img_arr_np[0])

    return imarge




@app.route('/api_pred2', methods=["POST"])
def API_pred2():
    variable_name = request.get_json( )
    key_variable = variable_name['image']

    predd = model.predict(key_variable)
    predd_list = predd.tolist()
    jsonString = json.dumps(predd_list)

    return jsonString


@app.route('/test')
def test():
  return 'test'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)