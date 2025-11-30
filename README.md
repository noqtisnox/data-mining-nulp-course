# Data Mining Labs (IAD)

This repository contains several laboratory exercises completed for a Data Mining course. Each subdirectory (`lab2-decision-trees/`, etc.) represents a self-contained lab focusing on a specific algorithm or concept.

---

## 🚀 Lab Structure

Each lab folder is structured uniformly:

| Folder/File | Description |
| :--- | :--- |
| `data/` | **Input datasets** used by the lab's scripts. Ignored by Git (check `.gitignore`). |
| `json/` | **Output** of the experiments, such as performance metrics, cross-validation scores, or model configurations. Ignored by Git. |
| `main.py` | The **entry point** script to run the full lab, usually handling data loading, model training, and result output. |
| `id3.py`, `c45.py` | Core scripts containing the implementation of the primary algorithms (e.g., ID3, C4.5). |
| `__init__.py` | Marks the directory as a **Python package** and controls its imports. |

---

## 🛠️ Setup and Installation

Follow these steps to set up the project environment and prepare for running the labs.

### 1. Clone the Repository

```bash
git clone https://github.com/noqtisnox/data-mining-nulp-course.git
cd data-mining-nulp-course
```

### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment (`venv`) to isolate dependencies.
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate
```

### 3. Install Dependencies
Install all necessary libraries (e.g., NumPy, Pandas, Scikit-learn).
```bash
pip install -r requirements.txt
```

## 🔬 How to Run a Lab
To run an experiment, simply navigate to the root directory and execute the `main.py` file within the specific lab folder.

**Example: Running Lab 2 (Decision Trees)**
```bash
# Ensure your virtual environment is active
python3 lab2-decision-trees/main.py
```
The results will be written to files inside the respective lab's `json/` folder.
