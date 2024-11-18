import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Dropout
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

def get_bitcoin_data():
    try:
        # Fetch Bitcoin data
        btc = yf.download('BTC-USD', start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        if btc.empty:
            raise ValueError("Không thể lấy được dữ liệu Bitcoin.")

        # Check for MultiIndex and extract 'Close' column for 'BTC-USD'
        if isinstance(btc.columns, pd.MultiIndex):
            if ('Close', 'BTC-USD') in btc.columns:
                close_data = btc[('Close', 'BTC-USD')]  # Extract 'Close' prices
                print(f"Dữ liệu cột 'Close' đã được trích xuất thành công với {len(close_data)} dòng.")
                return close_data
            else:
                raise KeyError("Cột ('Close', 'BTC-USD') không tồn tại trong dữ liệu.")
        else:
            raise KeyError("Dữ liệu không chứa MultiIndex hợp lệ.")

    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu: {str(e)}")
        return None

def create_features(data):
    df = pd.DataFrame(data)
    df.columns = ['Close']  # Đặt tên cột
    df['Returns'] = df['Close'].pct_change()
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA14'] = df['Close'].rolling(window=14).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['STD7'] = df['Close'].rolling(window=7).std()
    df.fillna(method='bfill', inplace=True)
    return df

def arima_prediction(data, forecast_days=30):
    try:
        # Kiểm tra dữ liệu đầu vào
        if data is None or len(data) == 0:
            raise ValueError("Dữ liệu đầu vào không hợp lệ")

        # Fit ARIMA model
        model = ARIMA(data, order=(2, 1, 0))
        results = model.fit()

        # Dự đoán in-sample
        in_sample_pred = results.fittedvalues

        # Dự đoán tương lai
        forecast = results.forecast(steps=forecast_days)

        return in_sample_pred, forecast, (2,1,0)
    except Exception as e:
        print(f"Lỗi trong ARIMA prediction: {str(e)}")
        return None, None, None

def create_drcnn_model(sequence_length, n_features):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu',
               input_shape=(sequence_length, n_features)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_data_drcnn(data, sequence_length):
    try:
        # Tạo features
        df = create_features(data)

        # Chuẩn hóa dữ liệu
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
            y.append(scaled_data[i + sequence_length, 0])

        X = np.array(X)
        y = np.array(y)

        return X, y, scaler, df.columns
    except Exception as e:
        print(f"Lỗi trong prepare_data_drcnn: {str(e)}")
        return None, None, None, None

def drcnn_prediction(data, forecast_days=30):
    try:
        sequence_length = 60
        X, y, scaler, features = prepare_data_drcnn(data, sequence_length)

        if X is None or y is None:
            raise ValueError("Lỗi trong quá trình chuẩn bị dữ liệu")

        # Chia dữ liệu
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        y_train = y[:train_size]

        # Tạo và huấn luyện mô hình
        model = create_drcnn_model(sequence_length, X.shape[2])
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        # Dự đoán in-sample
        in_sample_pred = model.predict(X)
        in_sample_pred = scaler.inverse_transform(
            np.concatenate([in_sample_pred, np.zeros((len(in_sample_pred), len(features)-1))], axis=1)
        )[:, 0]

        # Dự đoán tương lai
        last_sequence = X[-1]
        future_pred = []

        for _ in range(forecast_days):
            next_pred = model.predict(last_sequence.reshape(1, sequence_length, -1))
            future_pred.append(next_pred[0])
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1] = np.append(next_pred, np.zeros(len(features)-1))

        future_pred = np.array(future_pred)
        future_pred = scaler.inverse_transform(
            np.concatenate([future_pred, np.zeros((len(future_pred), len(features)-1))], axis=1)
        )[:, 0]

        return in_sample_pred, future_pred
    except Exception as e:
        print(f"Lỗi trong DRCNN prediction: {str(e)}")
        return None, None

def calculate_metrics(actual, predicted):
    """Tính toán các metrics đánh giá"""
    # Đảm bảo dữ liệu là 1D array
    actual = np.array(actual).ravel()
    predicted = np.array(predicted).ravel()

    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def print_insights(data, arima_results, drcnn_results):
    """In ra các insight về dự đoán"""
    print("\n=== PHÂN TÍCH KẾT QUẢ DỰ ĐOÁN ===")

    # Thống kê cơ bản về dữ liệu
    print("\n1. Thống kê dữ liệu Bitcoin:")
    print(f"- Giá trung bình: ${float(data.mean()):.2f}")
    print(f"- Giá cao nhất: ${float(data.max()):.2f}")
    print(f"- Giá thấp nhất: ${float(data.min()):.2f}")
    print(f"- Độ biến động (std): ${float(data.std()):.2f}")

    # Đánh giá mô hình ARIMA
    arima_in_sample, arima_forecast, _ = arima_results
    print("\n2. Đánh giá mô hình ARIMA:")

    # Cắt dữ liệu để có cùng độ dài
    min_len = min(len(data[1:len(arima_in_sample)+1]), len(arima_in_sample))
    actual_arima = data[1:min_len+1].values.flatten()
    predicted_arima = np.array(arima_in_sample[:min_len]).flatten()

    arima_metrics = calculate_metrics(actual_arima, predicted_arima)
    print(f"- RMSE: ${arima_metrics['RMSE']:.2f}")
    print(f"- MAE: ${arima_metrics['MAE']:.2f}")
    print(f"- MAPE: {arima_metrics['MAPE']:.2f}%")

    # Đánh giá mô hình DRCNN
    drcnn_in_sample, drcnn_forecast = drcnn_results
    print("\n3. Đánh giá mô hình DRCNN:")

    # Cắt dữ liệu để có cùng độ dài
    start_idx = 60
    min_len = min(len(data[start_idx:start_idx+len(drcnn_in_sample)]), len(drcnn_in_sample))
    actual_drcnn = data[start_idx:start_idx+min_len].values.flatten()
    predicted_drcnn = np.array(drcnn_in_sample[:min_len]).flatten()

    drcnn_metrics = calculate_metrics(actual_drcnn, predicted_drcnn)
    print(f"- RMSE: ${drcnn_metrics['RMSE']:.2f}")
    print(f"- MAE: ${drcnn_metrics['MAE']:.2f}")
    print(f"- MAPE: {drcnn_metrics['MAPE']:.2f}%")

    # Dự đoán giá trong tương lai
    print("\n4. Dự đoán giá trong tương lai:")
    print(f"ARIMA dự đoán giá sau 30 ngày: ${float(arima_forecast[-1]):.2f}")
    print(f"DRCNN dự đoán giá sau 30 ngày: ${float(drcnn_forecast[-1]):.2f}")

    # Xu hướng
    current_price = float(data.iloc[-1])
    future_avg = (float(arima_forecast[-1]) + float(drcnn_forecast[-1]))/2
    trend = "TĂNG" if future_avg > current_price else "GIẢM"
    print(f"\n5. Xu hướng dự đoán: {trend}")
    print(f"   Giá hiện tại: ${current_price:.2f}")
    print(f"   Giá dự đoán trung bình: ${future_avg:.2f}")
    print(f"   Phần trăm thay đổi: {((future_avg - current_price)/current_price * 100):.2f}%")

    # Thêm thông tin về độ chính xác của mô hình
    print("\n6. So sánh độ chính xác mô hình:")
    if arima_metrics['RMSE'] < drcnn_metrics['RMSE']:
        print("ARIMA có độ chính xác cao hơn dựa trên RMSE")
    else:
        print("DRCNN có độ chính xác cao hơn dựa trên RMSE")



def visualize_predictions(data, arima_results, drcnn_results):
    try:
        # Kiểm tra dữ liệu đầu vào
        if (arima_results is None or drcnn_results is None or
            not all(x is not None for x in arima_results) or
            not all(x is not None for x in drcnn_results)):
            raise ValueError("Dữ liệu dự đoán không hợp lệ")

        plt.figure(figsize=(15, 10))

        # Plot dữ liệu thực tế
        plt.plot(data.index, data.values, label='Actual', color='black', alpha=0.7)

        # Plot dự đoán in-sample
        arima_in_sample, arima_forecast, _ = arima_results
        drcnn_in_sample, drcnn_forecast = drcnn_results

        # Đảm bảo các chỉ số phù hợp
        valid_indices = min(len(data.index[1:]), len(arima_in_sample))
        plt.plot(data.index[1:valid_indices], arima_in_sample[:valid_indices-1],
                label='ARIMA In-sample', alpha=0.5)

        valid_indices_drcnn = min(len(data.index[60:]), len(drcnn_in_sample))
        plt.plot(data.index[60:60+valid_indices_drcnn], drcnn_in_sample[:valid_indices_drcnn],
                label='DRCNN In-sample', alpha=0.5)

        # Plot dự đoán tương lai
        future_dates = pd.date_range(start=data.index[-1], periods=31)[1:]
        plt.plot(future_dates, arima_forecast, label='ARIMA Forecast', linestyle='--')
        plt.plot(future_dates, drcnn_forecast, label='DRCNN Forecast', linestyle='--')

        plt.title('Bitcoin Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Lỗi trong visualize_predictions: {str(e)}")

def plot_individual_predictions(data, arima_results, drcnn_results):
    """Vẽ biểu đồ riêng cho từng mô hình"""
    # Bỏ dòng plt.style.use('seaborn')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Plot ARIMA predictions
    arima_in_sample, arima_forecast, _ = arima_results

    # Đảm bảo độ dài khớp nhau cho ARIMA
    min_len_arima = min(len(data) - 1, len(arima_in_sample))
    ax1.plot(data.index, data.values, label='Actual', color='black', alpha=0.7)
    ax1.plot(data.index[1:min_len_arima+1], arima_in_sample[:min_len_arima],
             label='ARIMA In-sample', color='blue', alpha=0.5)

    # Tạo future dates cho dự đoán
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=len(arima_forecast)+1)[1:]
    ax1.plot(future_dates, arima_forecast,
             label='ARIMA Forecast', color='red', linestyle='--')

    ax1.set_title('ARIMA Model Predictions', fontsize=12)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Bitcoin Price (USD)', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot DRCNN predictions
    drcnn_in_sample, drcnn_forecast = drcnn_results

    # Đảm bảo độ dài khớp nhau cho DRCNN
    start_idx = 60
    min_len_drcnn = min(len(data[start_idx:]) - 1, len(drcnn_in_sample))
    ax2.plot(data.index, data.values, label='Actual', color='black', alpha=0.7)
    ax2.plot(data.index[start_idx:start_idx+min_len_drcnn],
             drcnn_in_sample[:min_len_drcnn],
             label='DRCNN In-sample', color='green', alpha=0.5)

    # Tạo future dates cho dự đoán DRCNN
    future_dates_drcnn = pd.date_range(start=last_date, periods=len(drcnn_forecast)+1)[1:]
    ax2.plot(future_dates_drcnn, drcnn_forecast,
             label='DRCNN Forecast', color='purple', linestyle='--')

    ax2.set_title('DRCNN Model Predictions', fontsize=12)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Bitcoin Price (USD)', fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



def main():
    # Lấy dữ liệu
    print("Đang lấy dữ liệu Bitcoin...")
    data = get_bitcoin_data()

    if data is None or len(data) == 0:
        print("Không thể tiếp tục vì không có dữ liệu")
        return

    print("Đang thực hiện dự đoán ARIMA...")
    arima_results = arima_prediction(data)

    print("Đang thực hiện dự đoán DRCNN...")
    drcnn_results = drcnn_prediction(data)

    # Kiểm tra kết quả dự đoán
    if (arima_results is not None and len(arima_results) == 3 and
        drcnn_results is not None and len(drcnn_results) == 2):

        # In ra insights
        print_insights(data, arima_results, drcnn_results)

        # Vẽ biểu đồ tổng hợp
        print("\nĐang tạo biểu đồ tổng hợp...")
        visualize_predictions(data, arima_results, drcnn_results)

        # Vẽ biểu đồ riêng
        print("Đang tạo biểu đồ chi tiết cho từng mô hình...")
        plot_individual_predictions(data, arima_results, drcnn_results)

    else:
        print("Không thể tạo biểu đồ do lỗi trong quá trình dự đoán")

if __name__ == "__main__":
    main()