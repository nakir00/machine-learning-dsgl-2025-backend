from flask import Flask, request, jsonify
import json, os

port = int(os.environ.get("PORT", 8080))
app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", static_url_path="/static", port=port)
