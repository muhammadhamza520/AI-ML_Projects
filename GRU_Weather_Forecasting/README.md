# GRU for Weather Forecasting 🌦️

## 📌 Overview
This project implements a **Gated Recurrent Unit (GRU)** based deep learning model to forecast weather conditions using **historical hourly weather data**. The model predicts **temperature, humidity, and pressure** based on past observations.

---

## 📊 Dataset
The dataset is sourced from Kaggle: [Historical Hourly Weather Data](https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data).

- **Features used**:  
  - Temperature (°C)  
  - Humidity (%)  
  - Pressure (hPa)  
- **Location selected**: *Vancouver*  
- **Time resolution**: Hourly data  

👉 Note: Download the dataset from Kaggle and place it in the project folder as:
```
/dataset/temperature.csv
/dataset/humidity.csv
/dataset/pressure.csv
```

---

## ⚙️ Project Workflow
1. **Data Preprocessing**
   - Handle missing values using forward-fill/backfill.
   - Normalize data using `MinMaxScaler`.
   - Sequence generation for supervised learning (24 past hours → next hour prediction).
2. **Model Architecture**
   - GRU (64 units, ReLU, return sequences).
   - Dropout regularization.
   - GRU (32 units, ReLU).
   - Dense output layer (predicts temperature, humidity, pressure).
3. **Training**
   - Optimizer: Adam (lr=0.001)  
   - Loss: Mean Squared Error (MSE)  
   - Metrics: Mean Absolute Error (MAE)  
   - Early stopping & model checkpoint callbacks.
4. **Evaluation**
   - Compare actual vs. predicted values.  
   - Visualization of model performance.

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/AI-ML-Projects.git
   cd AI-ML-Projects/GRU-Weather-Forecasting
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook weather_forecasting_gru.ipynb
   ```
4. The trained model will be saved as:
   ```
   weather_forecasting_gru_model.h5
   ```

---

## 📈 Results
- **Evaluation Metrics**:
  - Test Loss (MSE): ~ (varies on training)  
  - Test MAE: ~ (varies on training)  
- The model demonstrates strong ability to capture temporal dependencies in weather data.
- Example Visualization:  
  - Actual vs Predicted Temperature (first 100 test samples).

---

## 🔮 Future Work
- Extend forecasting horizon to predict multiple hours ahead.  
- Experiment with **LSTM** and **Transformer models** for improved accuracy.  
- Incorporate additional features (wind speed, weather condition, etc.).  
- Deploy the trained model via a REST API or a Streamlit web app.  

---

## 📌 Requirements
- Python 3.8+
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn

Install them via:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

---

## 📜 License
This project is open-source and available under the MIT License.
