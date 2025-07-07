# SiStok: Fish Stock Analysis Application 🐟

SiStok (Stock Information System) is an interactive web application designed for fisheries data analysis. This application provides tools for researchers, fisheries managers, and academics to perform fish stock assessment using surplus production models.

The application is built using Streamlit and implements statistical models to evaluate fish stock status based on historical catch and fishing effort data.

## ✨ Key Features

### 1. 📊 Dashboard
- **Interactive Data Visualization**: Displays key metrics such as total catch, production value, total fishing days, and number of unique species
- **Dynamic Filtering**: Users can filter data by port, fish species, and year range
- **Analytical Charts**:
  - Annual total catch trends (Line Chart)
  - Top 10 fish species composition (Bar Chart)
  - Dominant fishing gear usage proportion (Pie Chart)
- **AI Assistant (Chatbot)**: Allows users to query basic information from displayed data using natural language (e.g., "What is the total squid catch in 2022?")
- **Data Preview**: Displays filtered raw data in table format

### 2. 🔬 Analysis
- **Data Upload**: Users can upload their own fisheries data in .csv format
- **Effort Standardization**: Automatically standardizes fishing effort from various types of fishing gear using Fishing Power Index (FPI)
- **Surplus Production Models**:
  - Implements two fundamental models: Schaefer Model (linear) and Fox Model (exponential)
  - Calculates key biological parameters: MSY (Maximum Sustainable Yield) and E_opt (Optimal Fishing Effort)
- **Model Comparison & Visualization**:
  - Displays comparison table of results from both models
  - Complete with evaluation metrics (R², RMSE, MAE, MAPE)
  - Visualizes surplus production curves and CPUE vs. Effort relationships for selected models
- **Management Recommendations**: Provides exploitation status (e.g., under-exploited, fully-exploited, or over-exploited) based on comparison between actual effort and E_opt
- **Stock Status Analysis**: Presents final conclusions on stock utilization status based on C/CMSY and E/E_opt matrix (Kobe Plot logic)

### 3. 📚 About
- **Complete Documentation**: In-depth explanation of the application, features, and usage instructions
- **Theoretical Background**: Mathematical explanations and characteristics of Schaefer and Fox Models
- **Glossary**: List of important terms in fisheries management (CPUE, MSY, etc.)
- **Evaluation Metrics**: Explanation of formulas and interpretation of statistical metrics used (R², RMSE, MAE, MAPE)

## 🛠️ Technology & Libraries

This application is built with Python and the following libraries:

- **Framework**: streamlit, streamlit_option_menu
- **Data Manipulation**: pandas, numpy
- **Visualization**: plotly, matplotlib
- **Models & Statistics**: scikit-learn, scipy
- **Utilities**: gdown (for downloading data from Google Drive)

## 🚀 Running the Application

### 1. Prerequisites
- Python 3.8 or newer
- pip (package installer for Python)

### 2. Installation

Clone this repository:
```bash
git clone https://github.com/tndry/sistok-streamlit.git
cd sistok-streamlit
```

Create and activate virtual environment (optional but recommended):
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Install all required libraries:
```bash
pip install -r requirements.txt
```

### 3. Running the Application

Run the Streamlit application from terminal:
```bash
streamlit run sistok_app.py
```

Open your browser and access the URL displayed in the terminal (usually `http://localhost:8501`).

## 📋 Data Format for Analysis

To use the Analysis feature with your own data, ensure your .csv file has the following columns:

| Column | Description | Data Type |
|--------|-------------|-----------|
| `tahun` | Year of data recording | Integer |
| `jenis_api` | Fishing gear type | String |
| `berat` | Catch weight in Kg | Float |
| `Jumlah Hari` | Trip duration or fishing effort in days | Integer |
| `Nilai Produksi` | Economic value of catch (Optional) | Float |

**Sample data format:**
```csv
tahun,jenis_api,berat,Jumlah Hari,Nilai Produksi
2020,Jaring Insang,150.5,3,2250000
2020,Pancing,75.2,2,1128000
2021,Jaring Insang,180.3,4,2704500
```

You can download sample data format from the Analysis page in the application.

## 📊 Requirements

```
streamlit
pandas
numpy
plotly
matplotlib
gdown
scikit-learn
scipy
streamlit-option-menu
```

## 🤝 Contributing

Contributions are welcome! Please create a pull request or open an issue for suggestions and improvements.

## 📄 License

## 📞 Contact

If you have questions, suggestions, or need assistance with this application, please contact:

📧 **Email**: tandrysimamora@gmail.com

---

**Note**: This application is developed for research and educational purposes in fisheries management. Analysis results should be interpreted carefully and validated with adequate field data.
