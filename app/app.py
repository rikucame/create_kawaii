#Flaskとrender_template（HTMLを表示させるための関数）をインポート
from flask import Flask,render_template, request
import sys
sys.path.append("./transformer/translate/")
from translate import Predicter
#Flaskオブジェクトの生成
app = Flask(__name__)

weight = "./app/model_result_0033001iteration.pt"
sp = "./app/en_ja_8000.model"
test = Predicter(weight_path=weight, sp_path=sp)

#「/」へアクセスがあった場合に、"Hello World"の文字列を返す
@app.route("/", methods=["GET", "POST"])
def hello():
    input = ""
    if(request.form.get("input")):
        input = request.form.get("input")
    else :
        input = "メガネ"
    output = test.predict(input)
    return render_template("index.html", input=input, output=output)
    


#「/index」へアクセスがあった場合に、「index.html」を返す
@app.route("/index")
def index():
    return render_template("index.html")


#おまじない
if __name__ == "__main__":
    app.run(debug=True)