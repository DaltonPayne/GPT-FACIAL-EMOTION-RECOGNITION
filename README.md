# GPT+FACIAL EMOTION RECOGNITION

This project is an emotion recognition system that combines GPT-4 (or GPT-3.5-turbo, which is the default model to improve response time) and a deep learning model trained on the FER2013 dataset. It detects facial emotions in real-time from a webcam feed and generates AI responses based on the user's emotion. The project is implemented using TensorFlow, OpenCV, and OpenAI's API.

## Installation Guide

### Prerequisites

- Python 3.7 or higher
- pip

### Dependencies

- numpy
- pandas
- tensorflow
- opencv-python
- openai
- scikit-learn
- tqdm

### Steps

1. Clone the repository:

```bash
git clone https://github.com/yourusername/GPT-Facial-Emotion-Recognition.git
cd GPT-Facial-Emotion-Recognition
```

2. Create a virtual environment and activate it (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use \`venv\Scripts\activate\`
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Download the FER2013 dataset from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) and place the `fer2013.csv` file in the root of the project folder.

5. Download the Haar Cascade XML file for face detection from [OpenCV's GitHub repository](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) and place it in the root of the project folder.

6. Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your_api_key_here"  # On Windows, use \`set OPENAI_API_KEY=your_api_key_here\`
```

7. Run the script:

```bash
python main.py
```

## How It Works

1. The script starts by checking if a trained model exists. If not, it loads the FER2013 dataset and creates a deep learning model using TensorFlow.
2. The model is trained on the dataset and saved to disk.
3. The script then starts the emotion recognition loop, capturing video frames from the user's webcam and converting them to grayscale.
4. Using OpenCV's Haar Cascade classifier, it detects faces in the frames and feeds them into the trained deep learning model.
5. The model predicts the user's emotion and displays it on the frame.
6. If the user's emotion changes, a prompt is generated based on the detected emotion and passed to OpenAI's GPT-4 API.
7. The AI's response is displayed on the frame for a set duration.
8. The loop continues until the user presses 'q' to exit.
