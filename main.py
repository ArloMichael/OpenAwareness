import gen
import time
from flask import Flask, request
app = Flask(__name__)

model = gen.NewModel()

@app.route("/")
def home():
    try:
        return {"code":"success", "response":"welcome",  "time":0}
    except Exception as e:
        return {"code":"error", "response":str(e), "time":0}

@app.route("/train")
def train_on():
    try:
        training_data = request.args.get("data")
        start = time.time()
        model.train(training_data,epochs=int(request.args.get("epochs")))
        return {"code":"success", "response":"trained", "time":round(time.time()-start,2)}
    except Exception as e:
        return {"code":"error", "response":str(e), "time":round(time.time()-start,2)}

@app.route("/generate")
def generate_text():
    start = time.time()
    try:
        if request.args.get("max"):
            generated_text = model.generate(request.args.get("seed"), max_words=int(request.args.get("max")),fix_gram=True)
        else:
            generated_text = model.generate(request.args.get("seed"),fix_gram=True)

        return {"code":"success", "response":str(generated_text), "time":round(time.time()-start,2)}
    except Exception as e:
        return {"code":"error", "response":str(e), "time":round(time.time()-start,2)}

@app.route("/load")
def load_pretrained():
    start = time.time()
    try:
        model.load()
        return {"code":"success", "response":"loaded", "time":round(time.time()-start,2)}
    except Exception as e:
        return {"code":"error", "response":str(e), "time":round(time.time()-start,2)}

@app.route("/reset")
def reset_model():
    start = time.time()
    try:
        model.reset()
        return {"code":"success", "response":"reset", "time":round(time.time()-start,2)}
    except Exception as e:
        return {"code":"error", "response":str(e), "time":round(time.time()-start,2)}

app.run(host='0.0.0.0', port=5000)