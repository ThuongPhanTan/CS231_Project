from flask import Flask, render_template, Response, request, make_response, send_from_directory
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
from utils.effect import Effects

dict_effect = {
    "time_warp_horizontal":0,
    "time_warp_vertical":0,
    "thug_life_ef":0,
    'lipstick':0
}

convert_name = {
    "time_warp_horizontal": "Time warp horizontal",
    "time_warp_vertical":"Time warp vertical",
    "thug_life_ef": "Thug life",
    "lipstick": "Lipstick"
}

choose =0
rec=0


#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


def change_values_dict(dic,key):
    keys = list(dic.keys())
    for i in keys:
        if i==key:
            dic[i]=1
        else:
            dic[i]=0
    print(dic)

def name_effect_choosed(dic):
    for key,value in dic.items():
      if value==1:
          return convert_name[key]

 
def gen_frames():  
    camera = cv2.VideoCapture(0)
    global out, capture,rec_frame
    count_frame = 0
    while True:
        success, frame = camera.read() 
        count_frame +=1
        if success:                    
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except:
                pass         

def record_frame():
    name_effect = ""
    if dict_effect["time_warp_horizontal"]:
        name_effect = "time_warp_scan_horizontal"

    elif dict_effect["time_warp_vertical"]:
        name_effect = "time_warp_scan_vertical"

    elif dict_effect["thug_life_ef"]:
        name_effect = "thug_life"

    elif dict_effect["lipstick"]:
        name_effect = "lipstick"

    recorder = Effects(effects=name_effect)
    return recorder


    
@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    if not rec:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(record_frame().record_video_capture(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global zoom_in_ef, sepia_ef,time_warp_horizontal, time_warp_vertical,face_recognition
    global rec, demo, choose

    if request.method == 'POST':

        if  request.form.get('effect') == 'time_warp_scan_vertical':   
            change_values_dict(dict_effect,"time_warp_vertical") 

        elif  request.form.get('effect') == 'time_warp_scan_horizontal':   
            change_values_dict(dict_effect,"time_warp_horizontal")   

        elif  request.form.get('effect') == 'thug_life_effect':     
            change_values_dict(dict_effect,"thug_life_ef") 

        elif  request.form.get('effect') == 'lipstick':     
            change_values_dict(dict_effect,"lipstick")
            
        elif  request.form.get('rec') == 'Rec':
            global rec
            rec = not rec
            
      
 
    elif request.method=='GET':
        return render_template("index.html",effedt_choosed = name_effect_choosed(dict_effect))
    return render_template("index.html",effedt_choosed = name_effect_choosed(dict_effect))

@app.route("/static/<path:path>")
def static_dir(path):
    return send_from_directory("static", path)


if __name__ == '__main__':
    app.run(debug=True)
    
