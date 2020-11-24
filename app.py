#Flaskとrender_template（HTMLを表示させるための関数）をインポート
from flask import Flask,render_template, request
import os
from PIL import Image
import torch
from gan_model.model import Generator
from gan_model.generator import generate
import random

#Flaskオブジェクトの生成
app = Flask(__name__,static_folder="./outputs/")

latent = 512
n_mlp = 8
size = 512
weight_path = "./pretrained_weight.pt"
device = "cpu"
truncation_mean = 4096

g_ema = Generator(size, latent, n_mlp, channel_multiplier=2)

g_ema.load_state_dict(torch.load(weight_path))
g_ema = g_ema.to(device)

with torch.no_grad():
    mean_latent = g_ema.mean_latent(truncation_mean)

#「/」へアクセスがあった場合に、"Hello World"の文字列を返す
@app.route("/", methods=["GET", "POST"])
def hello():
    file_name = "test.png"
    rand = random.randint(100,999)
    if(request.form.get("input")) is None:
        input = "生成してません"
    else :
        input = "生成してます"
        sample = generate(g_ema, device, mean_latent)
        Image.fromarray(sample).save(f'./outputs/generate_{rand}.png')
        file_name = f'generate_{rand}.png'

    return render_template("index.html", input=input, file_name=file_name)

#「/index」へアクセスがあった場合に、「index.html」を返す
@app.route("/index")
def index():
    return render_template("index.html")


#おまじない
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)