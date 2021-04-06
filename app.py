from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
app = Flask(__name__)

cv = pickle.load(open("cv.pkl", "rb"))
loaded_model = pickle.load(open("review.pkl", "rb"))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST' and 'message' in request.form:
        review = str(request.form.get('message'))
        if len(review)==0:
            return render_template('index.html',prediction_text="Please try again")
        else:    
            data = [review]
            vect = cv.transform(data).toarray() 
            pred = loaded_model.predict(vect) 

            if pred[0]==1:
                return render_template('index.html',prediction_text="The given review for movie is having Positive Feedback")
            else:
                return render_template('index.html',prediction_text="The given review for movie is having Negative Feedback")
    else:
        return render_template('index.html',prediction_text="Please try again")
    
if __name__=="__main__":
    app.run(debug=True)
