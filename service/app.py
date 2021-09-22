from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
import numpy as np
import pandas as pd
import nltk
import string
import re
import ktrain 
import logging
import os

import tensorflow as tf
tf.get_logger().setLevel('INFO')

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


flask_app = Flask(__name__)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "ML React App", 
		  description = "Predict results using a trained model")

name_space = app.namespace('prediction', description='Prediction APIs')

model = app.model('Prediction params', 
				  {'textField1': fields.String(required = True, 
				  							   description="Enter the tweet here", 
    					  				 	   help="Tweet field cannot be blank")})


classifier = ktrain.load_predictor('/home/naim/Documents/Transformer-BERT-SMS-Spam-Detection/saved_models/ONLP_model_1')

@name_space.route("/")
class MainClass(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response

	@app.expect(model)		
	def post(self):
		try: 
			formData = request.json
			data = [val for val in formData.values()]
			test_tweet = data
			test_tweet =  re.sub(r'http\S+', '', str(test_tweet))
			test_tweet =  re.sub(r'[^a-zA-Z\s\w.?!,#@]+', '', str(test_tweet))
			test_tweet =  re.sub('[0-9]+', '', str(test_tweet))


			prediction = classifier.predict(test_tweet)
			response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result": "Tweeter: " + ", ".join(prediction)
				})
			response.headers.add('Access-Control-Allow-Origin', '*')
			return response
		except Exception as error:
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})





