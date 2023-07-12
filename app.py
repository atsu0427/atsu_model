from flask import Flask, request, jsonify,render_template
from model import load_model, recommend
import model

app = Flask(__name__)

vectorizer = model.load_model()
app.jinja_env.globals.update(zip=zip)

@app.route("/",methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    elif request.method == 'POST':
        data = request.form['name']
        recommendations = recommend(vectorizer, data)
        image_filenames = [f"{name}.jpeg" for name in recommendations]
        return render_template('result.html',data=data,result=recommendations,image=image_filenames )

if __name__ == "__main__":
    app.run(debug=True)


