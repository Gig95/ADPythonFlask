# import main Flask class and request object
import json

from flask import Flask, request, jsonify
import model

# create the Flask app
app = Flask(__name__)


# GET requests will be blocked
@app.route('/recommend', methods=['POST'])
def recommendations():
    request_data = request.get_json()

    foodType = ""
    carbTypeString = ""
    proteinTypeString = ""

    if request_data:
        if 'foodType' in request_data:
            foodType = request_data['foodType']

        if 'carbType' in request_data:
            for carb in request_data['carbType']:
                carbTypeString += carb + " "

        if 'proteinType' in request_data:
            for protein in request_data['proteinType']:
                proteinTypeString += protein + " "

    top_stalls_list = model.recommendation(foodType, carbTypeString, proteinTypeString)
    topstalls = []
    for stall in top_stalls_list:
        stallJson = stall.to_json()
        stallData = json.loads(stallJson)
        topstalls.append(stallData)

    return jsonify(topstalls)


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
