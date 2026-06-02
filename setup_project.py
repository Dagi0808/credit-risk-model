from pathlib import Path

# Project directories
directories = [
    ".github/workflows",
    "data/raw",
    "data/processed",
    "notebooks",
    "src/api",
    "tests",
]

# Project files
files = [
    ".github/workflows/ci.yml",
    "notebooks/eda.ipynb",
    "src/__init__.py",
    "src/data_processing.py",
    "src/train.py",
    "src/predict.py",
    "src/api/main.py",
    "src/api/pydantic_models.py",
    "tests/test_data_processing.py",
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    ".gitignore",
    "README.md",
]

# Create directories
for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Create files
for file in files:
    Path(file).touch(exist_ok=True)

print("✓ Project structure created successfully")