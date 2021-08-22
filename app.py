import flask
from flask import render_template, url_for
import pandas as pd
import dill
from sklearn.ensemble import GradientBoostingClassifier

param_dict = {'bruises?': {'f': 2, 't': 1},
 'cap-color': {'b': 7,
  'c': 9,
  'e': 5,
  'g': 4,
  'n': 1,
  'p': 6,
  'r': 10,
  'u': 8,
  'w': 3,
  'y': 2},
 'cap-shape': {'b': 2, 'c': 6, 'f': 4, 'k': 5, 's': 3, 'x': 1},
 'cap-surface': {'f': 3, 'g': 4, 's': 1, 'y': 2},
 'gill-attachment': {'a': 2, 'f': 1},
 'gill-color': {'b': 9,
  'e': 8,
  'g': 3,
  'h': 6,
  'k': 1,
  'n': 2,
  'o': 12,
  'p': 4,
  'r': 10,
  'u': 7,
  'w': 5,
  'y': 11},
 'gill-size': {'b': 2, 'n': 1},
 'gill-spacing': {'c': 1, 'w': 2, 'd': 3},
 'habitat': {'d': 4, 'g': 2, 'l': 7, 'm': 3, 'p': 5, 'u': 1, 'w': 6},
 'odor': {'a': 2,
  'c': 6,
  'f': 5,
  'l': 3,
  'm': 9,
  'n': 4,
  'p': 1,
  's': 8,
  'y': 7},
 'population': {'a': 3, 'c': 6, 'n': 2, 's': 1, 'v': 4, 'y': 5},
 'ring-number': {'n': 3, 'o': 1, 't': 2},
 'ring-type': {'e': 2, 'f': 4, 'l': 3, 'n': 5, 'p': 1},
 'spore-print-color': {'b': 9,
  'h': 4,
  'k': 1,
  'n': 2,
  'o': 7,
  'r': 6,
  'u': 3,
  'w': 5,
  'y': 8},
 'stalk-color-above-ring': {'b': 5,
  'c': 8,
  'e': 6,
  'g': 2,
  'n': 4,
  'o': 7,
  'p': 3,
  'w': 1,
  'y': 9},
 'stalk-color-below-ring': {'b': 4,
  'c': 9,
  'e': 6,
  'g': 3,
  'n': 5,
  'o': 8,
  'p': 2,
  'w': 1,
  'y': 7},
 'stalk-root': {'?': 5, 'b': 3, 'c': 2, 'e': 1, 'r': 4},
 'stalk-shape': {'e': 1, 't': 2},
 'stalk-surface-above-ring': {'f': 2, 'k': 3, 's': 1, 'y': 4},
 'stalk-surface-below-ring': {'f': 2, 'k': 4, 's': 1, 'y': 3},
 'veil-color': {'n': 2, 'o': 3, 'w': 1, 'y': 4},
 'veil-type': {'p': 1}}

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
