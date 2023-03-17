import pickle
from urllib import request
import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# scaler = pickle.load(open('scaler_model.pkl','rb'))
# model = pickle.load(open('my_model.pkl','rb'))
model = pickle.load(open('xgb_pred_model.pkl','rb'))
# df2 = pd.read_csv("Mod_acc_data.csv")

# X = df2[['Category_Code', 'Accident_Type_Code', 'Year', 'Month']]
# y = df2['Value']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Parse the JSON request body
    data = request.get_json()

    # Extract the values for each feature
    category_code = data['category_code']
    acc_type_code = data['accident_type_code']
    year = data['year']
    month = data['month']

    #prediction
    input_data = np.array([[category_code, acc_type_code, year, month]])
    result = model.predict(input_data)
    response = {'predictions': result.tolist()}
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

