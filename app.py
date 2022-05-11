import os

from flask import render_template, request, Flask
from werkzeug.utils import redirect

from process import predict_img

app = Flask(__name__)
base_dir = os.path.abspath(os.path.dirname(__file__))


@app.route('/')
def home():
    return render_template('index.html',title='Home')

@app.route("/predict")
def predict():
    return render_template("predict.html",title="Predict")

@app.route("/upload", methods=["GET","POST"])
def upload():
    target = os.path.join(base_dir, 'temp/')
    if request.method == 'POST':
        file = request.files['img'] # 'img' is the id passed in input file form field
        filename = file.filename
        # filename = filename(filename)
        file.save("".join([target, filename])) #saving file in temp folder
        print("upload Completed") #printing on terminal

        return redirect('/prediction/{}'.format(filename))

@app.route("/prediction/<filename>",methods=["GET","POST"])
def prediction(filename):
    #imported process.py
    x=predict_img(filename) #imported from process file
    return render_template('output.html',results=x)

if __name__=='__main__':
    app.run(debug=True)