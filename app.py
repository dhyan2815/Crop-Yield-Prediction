from flask import Flask,render_template,request

# create a simple flask application
app=Flask(__name__)

# Index route
@app.route("/",methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        crop = request.form.get('crop')
        area = request.form.get('area')
        region = request.form.get('region')
        year = request.form.get('year')

        print(f"Prediction request received:")
        print(f"Crop: {crop}")
        print(f"Area: {area} hectares")
        print(f"Region: {region}")
        print(f"Year: {year if year else 'Current Year'}")

if __name__=="__main__":
    app.run(debug=True)