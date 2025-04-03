# DrunkTester 2.0

A sophisticated web application that uses machine learning to detect potential intoxication through facial analysis and speech pattern recognition.

## Features

- **Facial Analysis**: Uses a deep learning model to analyze facial features for signs of intoxication
- **Speech Pattern Recognition**: Analyzes speech patterns for slurring and other indicators
- **Combined Analysis**: Provides a comprehensive assessment using both facial and speech analysis
- **Real-time Processing**: Fast and efficient processing of uploaded media
- **Secure**: Implements rate limiting and secure file handling
- **User-Friendly**: Simple web interface for easy interaction

## Prerequisites

- Python 3.8 or higher
- PyTorch
- Flask
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aaryapatel09/DrunkTester2.0.git
cd DrunkTester2.0
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The application can be configured through environment variables:

```bash
export SECRET_KEY='your-secret-key'
export FLASK_DEBUG=False
export PORT=5000
```

Or modify the `config.py` file directly.

## Usage

1. Start the application:
```bash
python main.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Follow the on-screen instructions:
   - Upload a photo for facial analysis
   - Upload a video for speech analysis
   - View the combined results

## Supported File Formats

- Images: JPG, JPEG, PNG
- Videos: MP4, MOV, AVI, MKV, WMV, FLV

## Technical Details

### Face Analysis
- Uses a CNN model trained on facial features
- Implements batch normalization and dropout for better accuracy
- Caches results for improved performance

### Speech Analysis
- Uses Google's Speech Recognition API
- Implements retry mechanism for better reliability
- Normalizes audio for consistent analysis
- Caches results for improved performance

### Security Features
- Rate limiting to prevent abuse
- Secure file handling
- Input validation
- Session management
- File size limits

## Error Handling

The application includes comprehensive error handling for:
- File upload issues
- Processing errors
- API failures
- Invalid inputs

## Logging

Logs are stored in `app.log` and include:
- Application events
- Error messages
- Processing results
- Security events

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch for the deep learning framework
- Flask for the web framework
- Google Speech Recognition API
- All contributors and users of the application
