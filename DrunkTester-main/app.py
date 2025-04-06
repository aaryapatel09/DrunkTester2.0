from flask import Flask, request, jsonify, send_file
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
import config
from face_analyzer import FaceAnalyzer
from speech_analyzer import SpeechAnalyzer
import hashlib
import json

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(config)

# Initialize extensions
api = Api(
    app,
    version='1.0',
    title='DrunkTester API',
    description='Advanced intoxication detection system',
    doc='/docs'
)
CORS(app)
jwt = JWTManager(app)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[config.API_RATE_LIMIT]
)

# Initialize analyzers
face_analyzer = FaceAnalyzer()
speech_analyzer = SpeechAnalyzer()

# API Models
intoxication_model = api.model('IntoxicationResult', {
    'timestamp': fields.DateTime(required=True),
    'confidence': fields.Float(required=True),
    'is_intoxicated': fields.Boolean(required=True),
    'features': fields.Raw(required=True),
    'raw_prediction': fields.List(fields.Float, required=True)
})

analysis_request_model = api.model('AnalysisRequest', {
    'file_path': fields.String(required=True),
    'analysis_type': fields.String(required=True, enum=['face', 'speech', 'combined'])
})

# Namespaces
ns = api.namespace('api/v1', description='Intoxication detection operations')

@ns.route('/analyze')
class AnalysisEndpoint(Resource):
    @ns.expect(analysis_request_model)
    @ns.response(200, 'Success', intoxication_model)
    @ns.response(400, 'Invalid input')
    @ns.response(401, 'Unauthorized')
    @ns.response(429, 'Too many requests')
    @jwt_required()
    @limiter.limit("5 per minute")
    def post(self):
        """Analyze a file for signs of intoxication."""
        try:
            data = request.get_json()
            file_path = data.get('file_path')
            analysis_type = data.get('analysis_type')
            
            if not file_path or not analysis_type:
                return {'error': 'Missing required parameters'}, 400
            
            if not os.path.exists(file_path):
                return {'error': 'File not found'}, 404
            
            # Perform analysis based on type
            if analysis_type == 'face':
                result = face_analyzer.analyze_face(file_path)
            elif analysis_type == 'speech':
                result = speech_analyzer.analyze_speech(file_path)
            elif analysis_type == 'combined':
                face_result = face_analyzer.analyze_face(file_path)
                speech_result = speech_analyzer.analyze_speech(file_path)
                
                # Combine results
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'confidence': (face_result['confidence'] + speech_result['confidence']) / 2,
                    'is_intoxicated': face_result['is_intoxicated'] or speech_result['is_intoxicated'],
                    'face_analysis': face_result,
                    'speech_analysis': speech_result
                }
            else:
                return {'error': 'Invalid analysis type'}, 400
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error in analysis endpoint: {str(e)}")
            return {'error': str(e)}, 500

@ns.route('/history')
class HistoryEndpoint(Resource):
    @ns.response(200, 'Success')
    @ns.response(401, 'Unauthorized')
    @jwt_required()
    def get(self):
        """Get analysis history."""
        try:
            face_history = face_analyzer.get_analysis_history()
            speech_history = speech_analyzer.get_analysis_history()
            
            return jsonify({
                'face_analysis_history': face_history,
                'speech_analysis_history': speech_history
            })
            
        except Exception as e:
            logger.error(f"Error getting history: {str(e)}")
            return {'error': str(e)}, 500

@ns.route('/clear-cache')
class CacheEndpoint(Resource):
    @ns.response(200, 'Success')
    @ns.response(401, 'Unauthorized')
    @jwt_required()
    def post(self):
        """Clear analysis caches."""
        try:
            face_analyzer.clear_cache()
            speech_analyzer.clear_cache()
            
            return {'message': 'Caches cleared successfully'}
            
        except Exception as e:
            logger.error(f"Error clearing caches: {str(e)}")
            return {'error': str(e)}, 500

@ns.route('/train')
class TrainEndpoint(Resource):
    @ns.response(200, 'Success')
    @ns.response(401, 'Unauthorized')
    @jwt_required()
    def post(self):
        """Train models on new data."""
        try:
            # This is a placeholder for actual training implementation
            # In a real implementation, you would:
            # 1. Load training data
            # 2. Preprocess data
            # 3. Train models
            # 4. Save updated models
            
            return {'message': 'Training completed successfully'}
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            return {'error': str(e)}, 500

# Authentication
@ns.route('/login')
class LoginEndpoint(Resource):
    @ns.response(200, 'Success')
    @ns.response(401, 'Invalid credentials')
    def post(self):
        """Login and get JWT token."""
        try:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
            
            # This is a placeholder for actual authentication
            # In a real implementation, you would verify credentials against a database
            if username == 'admin' and password == 'admin':
                access_token = create_access_token(
                    identity=username,
                    expires_delta=timedelta(hours=1)
                )
                return {'access_token': access_token}
            
            return {'error': 'Invalid credentials'}, 401
            
        except Exception as e:
            logger.error(f"Error in login: {str(e)}")
            return {'error': str(e)}, 500

if __name__ == '__main__':
    # Ensure required directories exist
    for folder in [
        config.UPLOAD_FOLDER,
        config.DATA_FOLDER,
        config.MODELS_FOLDER,
        config.LOGS_FOLDER,
        config.CACHE_FOLDER
    ]:
        os.makedirs(folder, exist_ok=True)
    
    # Start the application
    app.run(
        host='0.0.0.0',
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    ) 