from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__, template_folder='templates')

@app.route('/')
def userauth():
    return render_template('test.html')

if __name__ == '__main__':
    app.run()