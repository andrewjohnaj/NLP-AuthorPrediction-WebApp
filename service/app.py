from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import string
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


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

classifier = joblib.load('NBmodel.joblib')
# Load the object from the file 
cv_from_joblib = joblib.load('CVobject.joblib')
cv_teststring = CountVectorizer(vocabulary=cv_from_joblib.get_feature_names(),lowercase=True, ngram_range = (1,1), analyzer = "word")


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
			stop = stopwords.words('english')
			test_tweet = ' '.join([word for word in test_tweet.split() if word not in stop])
			test_tweet = word_tokenize(test_tweet)
			lemmatizer = WordNetLemmatizer()
			test_tweet_lem = ' '.join([lemmatizer.lemmatize(w) for w in test_tweet])
			test_tweet_bow = cv_teststring.fit_transform([test_tweet_lem])

			prediction = classifier.predict(test_tweet_bow)
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