from flask import Flask, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
import config

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(config)

# Initialize extensions
api = Api(
    app,
    version='1.0',
    title='DrunkTester API',
    description='A sophisticated API for intoxication detection',
    doc='/docs' if config.ENABLE_API_DOCUMENTATION else False
)
CORS(app, resources={r"/*": {"origins": config.CORS_ORIGINS}})
jwt = JWTManager(app)
db = SQLAlchemy(app)

# Configure logging
if not os.path.exists(config.LOGS_FOLDER):
    os.makedirs(config.LOGS_FOLDER)

file_handler = RotatingFileHandler(
    config.LOG_FILE,
    maxBytes=config.LOG_MAX_SIZE,
    backupCount=config.LOG_BACKUP_COUNT
)
file_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
app.logger.addHandler(file_handler)
app.logger.setLevel(config.LOG_LEVEL)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[config.API_RATE_LIMIT]
)

# API Models
intoxication_model = api.model('IntoxicationResult', {
    'is_intoxicated': fields.Boolean(required=True, description='Whether the person is intoxicated'),
    'confidence': fields.Float(required=True, description='Confidence score'),
    'face_analysis': fields.Nested(api.model('FaceAnalysis', {
        'score': fields.Float(required=True),
        'features': fields.List(fields.String, required=True)
    })),
    'speech_analysis': fields.Nested(api.model('SpeechAnalysis', {
        'score': fields.Float(required=True),
        'slurring_detected': fields.Boolean(required=True),
        'speech_clarity': fields.Float(required=True)
    })),
    'timestamp': fields.DateTime(required=True)
})

# Namespaces
ns = api.namespace('analysis', description='Intoxication analysis operations')

@ns.route('/face')
class FaceAnalysis(Resource):
    @ns.doc('analyze_face')
    @ns.response(200, 'Success', intoxication_model)
    @ns.response(400, 'Invalid input')
    @ns.response(401, 'Unauthorized')
    @ns.response(429, 'Too many requests')
    @limiter.limit("5 per minute")
    def post(self):
        """Analyze a face image for signs of intoxication"""
        try:
            # Implementation will be added
            return jsonify({
                'is_intoxicated': False,
                'confidence': 0.0,
                'face_analysis': {
                    'score': 0.0,
                    'features': []
                },
                'speech_analysis': None,
                'timestamp': datetime.utcnow()
            })
        except Exception as e:
            app.logger.error(f"Error in face analysis: {str(e)}")
            return {'message': 'Internal server error'}, 500

@ns.route('/speech')
class SpeechAnalysis(Resource):
    @ns.doc('analyze_speech')
    @ns.response(200, 'Success', intoxication_model)
    @ns.response(400, 'Invalid input')
    @ns.response(401, 'Unauthorized')
    @ns.response(429, 'Too many requests')
    @limiter.limit("5 per minute")
    def post(self):
        """Analyze speech for signs of intoxication"""
        try:
            # Implementation will be added
            return jsonify({
                'is_intoxicated': False,
                'confidence': 0.0,
                'face_analysis': None,
                'speech_analysis': {
                    'score': 0.0,
                    'slurring_detected': False,
                    'speech_clarity': 1.0
                },
                'timestamp': datetime.utcnow()
            })
        except Exception as e:
            app.logger.error(f"Error in speech analysis: {str(e)}")
            return {'message': 'Internal server error'}, 500

@ns.route('/combined')
class CombinedAnalysis(Resource):
    @ns.doc('analyze_combined')
    @ns.response(200, 'Success', intoxication_model)
    @ns.response(400, 'Invalid input')
    @ns.response(401, 'Unauthorized')
    @ns.response(429, 'Too many requests')
    @limiter.limit("5 per minute")
    def post(self):
        """Perform combined face and speech analysis"""
        try:
            # Implementation will be added
            return jsonify({
                'is_intoxicated': False,
                'confidence': 0.0,
                'face_analysis': {
                    'score': 0.0,
                    'features': []
                },
                'speech_analysis': {
                    'score': 0.0,
                    'slurring_detected': False,
                    'speech_clarity': 1.0
                },
                'timestamp': datetime.utcnow()
            })
        except Exception as e:
            app.logger.error(f"Error in combined analysis: {str(e)}")
            return {'message': 'Internal server error'}, 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )
