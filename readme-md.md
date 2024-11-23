# Health Data Analysis System

A web-based system for analyzing health data using natural language queries powered by the Phi-3.5 LLM model.

## Features

- Natural language to SQL query conversion
- Real-time analysis of health data
- Interactive web interface
- Query history tracking
- Streaming responses for better user experience

## Prerequisites

- Python 3.8+
- 8GB+ RAM
- ~4GB disk space for the model
- Internet connection for initial setup

## Installation

1. Clone the repository:
```bash
git clone https://github.com/piotrekobi/health-data-analysis
cd health-data-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the Phi-3.5 model:
```bash
mkdir -p models
wget https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q8_0.gguf -O models/Phi-3.5-mini-instruct-Q8_0.gguf
```

## Usage

1. Start the backend API server:
```bash
python .\src\api.py
```

2. In a new terminal, start the Streamlit frontend:
```bash
streamlit run app.py
```

3. Open your browser and navigate to `http://localhost:8501`

4. Enter your health-related questions in natural language, for example:
   - "What is the average BMI for patients with high stress levels?"
   - "Compare smoking rates between normal and abnormal blood pressure groups"
   - "Show the relationship between physical activity and blood pressure"


## Running Evaluation

To evaluate the system's performance on a set of test questions:

1. Create a text file `data/evaluation_questions.txt` with one question per line
2. Run the evaluation script:
```bash
python evaluation_framework.py
```

The script will:
- Process each question
- Generate and execute SQL queries
- Measure success rates and execution times
- Save detailed results in the `evaluation_results` directory