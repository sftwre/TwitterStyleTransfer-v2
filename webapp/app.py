from flask import Flask,send_from_directory,request,jsonify
from flask_cors import CORS, cross_origin
from fastai.text import *
from random import random

app = Flask(__name__, static_folder='client/build',static_url_path='')
cors = CORS(app)
STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

@app.route('/api')
@cross_origin()
def Welcome():
    return "Welcome to the API!!!"

@app.route('/api/style-tweet')
@cross_origin()
def style_tweet():
    style = request.args.get('style')
    text = request.args.get('text')
    res_text = ""
    orig_words = 4
    gen_words = 20

    # Ignore stop words in the provided text
    for word in text.split():
        if orig_words < 0:
            break

        if word.lower() not in STOP_WORDS:
            orig_words -= 1

        res_text += (word + " ")

    res_text = res_text[:-1] # Remove trailing space

    x = load_learner('', 'model/' + style + '.pkl') # Load the PTM

    # Find the length of tweet to generate
    predict_len = gen_words + (int(random() * 10) - 5)
    predict_len = max(predict_len, len(text))

    # Return the generated tweet
    predict = x.predict(res_text, predict_len, temperature=0.75)
    predict = predict.replace("xxbos ", "")
    return jsonify(predict)

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
