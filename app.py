#Flaskとrender_template（HTMLを表示させるための関数）をインポート
from flask import Flask,render_template, request
import os
from translater.predict import Predicter
# from transfomer import Predicter
#Flaskオブジェクトの生成
app = Flask(__name__)

weight = "./best_model.pt"
sp = "./tokenizer.model"
test = Predicter(dim=512,head_num=8,layer_num=4,pad_id=32000,seq_len=128,weight_path=weight, sp_path=sp)

#「/」へアクセスがあった場合に、"Hello World"の文字列を返す
@app.route("/", methods=["GET", "POST"])
def hello():
    input = ""
    if(request.form.get("input")):
        input = request.form.get("input")
    else :
        input = "メガネ"
    output = test.predict(input+".")
    return render_template("index.html", input=input, output=output)
    


#「/index」へアクセスがあった場合に、「index.html」を返す
@app.route("/index")
def index():
    return render_template("index.html")


#おまじない
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)