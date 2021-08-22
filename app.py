import flask
from flask import render_template, url_for
import pandas as pd
import dill
from sklearn.ensemble import GradientBoostingClassifier

param_dict_sm = {
    'year': {'2018': 2, '2017': 1},
    'mileage': {'14827': 1, '26676': 2, '62794': 3},
    'tax': {'145': 9,
                          '160': 4},
    'mpg': {'42.8': 4,
                               '51.4': 9,
                               '72.4': 6},
    'engineSize': {'2.0': 2, '3.0': 3, '1.5': 1},
}

def load_pipeline():
    with open('grad_boost_pipeline.dill', 'rb') as in_strm:
        pipeline = dill.load(in_strm)
    return pipeline


# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None


@app.route("/", methods=["GET"])
@app.route('/index_bmw.html', methods=["GET"])
def index():
    return render_template('form_bmw.html')


@app.route("/predict_bmw.html", methods=["GET", 'POST'])
def predict1():
    try:
        # print(flask.request.form)
        data = dict(flask.request.form)
        data_convert = dict()
        for key, value in data.items():
            data_convert[key] = [param_dict[key][value]]
        pipeline = load_pipeline()
        preds = pipeline.predict(pd.DataFrame(data_convert))
        #if preds[:, 1][0] > 0.5:
            #text = 'Вероятность того что гриб ядовит -'
            #result = round(preds[:, 1][0] * 100, 2)
        #else:
            #text = '''Скорее всего гриб не ядовит.
#Bероятность -'''
            result = round(preds)
        return render_template('itog_bmw.html', text=text, result=result)
    except Exception as e:
        return f'''Ошибка ввода данных {e} {dict(flask.request.form)}'''


@app.route("/predict_get", methods=['POST'])
def predict():
    try:
        data_out = {"success": False}
        data = (flask.request.get_json())
        data_convert = dict()
        for key, value in data.items():
            data_convert[key] = [param_dict[key][value]]
        pipeline = load_pipeline()
        preds = pipeline.predict(pd.DataFrame(data_convert))
        data_out["predictions"] = preds
        # indicate that the request was a success
        data_out["success"] = True
        return flask.jsonify(data_out)
    except:
        return f'''Ошибка ввода данных'''


if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    # pipeline = load_pipeline() здесь нельзя инициировать так как heroku не видит pipline если из пайчарма то нормально
    app.run()
