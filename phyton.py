                                             
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'welcome to amzu version2'

app.run(host='0.0.0.0', port=8080)
