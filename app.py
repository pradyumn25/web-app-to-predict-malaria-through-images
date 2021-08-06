from flask import Flask,render_template,request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('mal_detector.h5')

@app.route("/",methods=['POST','GET'])
def index():
    if(request.method=='POST'):
        cell_img = request.form.get('img')
        my_image = image.load_img(cell_img,target_size=(130,130,3))
        my_image = np.expand_dims(my_image, axis=0)
        pred = model.predict(my_image)

        if(pred[0][0]==0.):
            new_pred = 'infected'
        else:
            new_pred = 'not infected'
        return render_template("index.html",pred=new_pred)

    return render_template("index.html")


app.run(debug=True)