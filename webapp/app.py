from flask import Flask,send_from_directory,request,jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__, static_folder='client/build',static_url_path='')
cors = CORS(app)

@app.route('/api')
@cross_origin()
def Welcome():
    return "Welcome to the API!!!"

@app.route('/api/style-tweet')
@cross_origin()
def style_tweet():
    style = request.args.get('style')
    text = request.args.get('text')

    return jsonify(style + ":" + text)

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
