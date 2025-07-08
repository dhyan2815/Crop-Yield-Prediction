from flask import Flask,render_template,request


# create a simple flask application

app=Flask(__name__)

@app.route("/",methods=["GET"])
def welcome():
    return("<h1>Welcome to Home page</h1>")

@app.route("/index",methods=["GET"])
def index():
    return("<h2>Welcome to the Index page</h2>")

@app.route('/success/<int:score>')
def success(score):
    return "The person has passed and score is: "+ str(score)

@app.route('/fail/<int:score>')
def fail(score):
    return "The person has failed and score is: "+ str(score)

@app.route('/form', methods=["GET","POST"])
def form():
    if request.method=="GET":
        return render_template('form.html')




if __name__=="__main__":
    app.run(debug=True)