# Dental Diagnosis Expert System

An expert system application designed for enhanced dental disease diagnosis using machine learning techniques.

## Features

- Diagnoses 6 dental diseases based on 28 symptoms
- Uses K-Nearest Neighbors (KNN) algorithm
- Interactive web interface built with Streamlit
- Command-line interface option

## Diseases Diagnosed

- Gingivitis
- Karies Gigi (Dental Caries)
- Periodontitis
- Abses gigi (Dental Abscess)
- Pulpitis
- Stomatitis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dental-diagnosis-expert-system.git

# Navigate to the project directory
cd dental-diagnosis-expert-system

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Interface

```bash
streamlit run streamlit_app.py
```

### Network Access

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

### Command Line Interface

```bash
python dental_cli.py
```

## Dataset

The system uses a dataset of 100 patients with 28 symptoms to predict dental diseases.

## License

[MIT](LICENSE)