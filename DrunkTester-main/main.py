from flask import Flask, render_template, request, flash, redirect, url_for, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import face_model
import voice_model
from werkzeug.utils import secure_filename
import logging
from typing import Optional, Tuple
import config
from datetime import datetime, timedelta

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

app = Flask(__name__)
app.secret_key = config.FLASK_SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def save_file(file, folder: str) -> Optional[str]:
    """Safely save an uploaded file and return its path."""
    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(folder, unique_filename)
        file.save(filepath)
        return filepath
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return None

@app.route('/')
def camera():
    session.permanent = True
    return render_template('camera.html')

@app.route('/upload_photo', methods=['POST'])
@limiter.limit("5 per minute")
def upload_photo():
    try:
        if 'photo' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('camera'))
        
        photo = request.files['photo']
        if photo.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('camera'))
        
        if not allowed_file(photo.filename):
            flash('Invalid file type. Allowed types: ' + ', '.join(config.ALLOWED_EXTENSIONS), 'error')
            return redirect(url_for('camera'))

        filepath = save_file(photo, config.UPLOAD_FOLDER)
        if not filepath:
            flash('Error saving file', 'error')
            return redirect(url_for('camera'))

        try:
            confidence, is_intoxicated = face_model.check_intoxicated(filepath)
            session['face_confidence'] = confidence
            session['is_intoxicated'] = is_intoxicated
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return render_template('voice_model.html')
        except Exception as e:
            logger.error(f"Error processing photo: {str(e)}")
            flash('Error processing photo. Please try again.', 'error')
            return redirect(url_for('camera'))
    
    except Exception as e:
        logger.error(f"Unexpected error in upload_photo: {str(e)}")
        flash('An unexpected error occurred', 'error')
        return redirect(url_for('camera'))

@app.route('/upload_video', methods=['POST'])
@limiter.limit("5 per minute")
def upload_video():
    try:
        if 'video' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('voice_model'))
        
        video = request.files['video']
        if video.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('voice_model'))
        
        if not allowed_file(video.filename):
            flash('Invalid file type. Allowed types: ' + ', '.join(config.ALLOWED_EXTENSIONS), 'error')
            return redirect(url_for('voice_model'))

        filepath = save_file(video, config.UPLOAD_FOLDER)
        if not filepath:
            flash('Error saving file', 'error')
            return redirect(url_for('voice_model'))

        try:
            phrase = voice_model.get_phrase()
            session['target_phrase'] = phrase
            
            if not voice_model.get_wav_file(filepath):
                flash('Error processing video file', 'error')
                return redirect(url_for('voice_model'))

            slurring_score = voice_model.check_slurring('test.wav', phrase)
            session['slurring_score'] = slurring_score
            
            # Get face analysis results from session
            face_confidence = session.get('face_confidence', 0)
            is_intoxicated = session.get('is_intoxicated', False)
            
            # Calculate combined score
            combined_score = (face_confidence + slurring_score) / 2
            session['combined_score'] = combined_score
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if combined_score >= config.MODEL_CONFIDENCE_THRESHOLD:
                return render_template('intoxicated.html')
            else:
                return render_template('not_intoxicated.html')
                
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            flash('Error processing video. Please try again.', 'error')
            return redirect(url_for('voice_model'))
    
    except Exception as e:
        logger.error(f"Unexpected error in upload_video: {str(e)}")
        flash('An unexpected error occurred', 'error')
        return redirect(url_for('voice_model'))

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('camera'))

@app.errorhandler(429)
def ratelimit_handler(e):
    flash('Too many requests. Please try again later.', 'error')
    return redirect(url_for('camera'))

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )
