from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

import webcam_cv3
import FaceClustering

UPLOAD_FOLDER = '/Users/artrial/desktop/face_detect/face_clustering/FaceImageClustering/video'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['mp4', 'mov'])


#Example for Root Route
@app.route("/")
def index():
    return "Index!"

#Example for Resource Route
@app.route("/hello")
def hello():
    return "Hello World!"

#Example to pass path parameters
@app.route("/members/<string:name>/")
def getMember(name):
    return name

#Example for Query Params
@app.route('/query', methods=['GET', 'POST'])
def query_example():
    language = request.args.get('language') #if key doesn't exist, returns None
    if (language == None):
        return jsonify({'success': True})
    else:
        return '''<h1>The language value is: {}</h1>'''.format(language)

@app.route('/', methods=['POST'])
def post_example():
    print("Request Body: " + str(request.files['image'].filename));
    file = request.files['image']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    webcam_cv3.webcamCrop(request.files['image'].filename)
    FaceClustering.faceCluster()
    return "Video Uploaded SuccessFully."

if __name__ == "__main__":
    app.run(debug=True, port=5000)
