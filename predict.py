import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os
from data_loader import load_data, prepare_data, HouseRentDataset
from model import RentPredictionModel, save_model, load_model
from utils import save_chart, logger, IMAGES_DIR, MODELS_DIR

warnings.filterwarnings('ignore')

# Налаштування відтворюваності результатів
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Перевірка доступності GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Використовуємо пристрій: {device}")

# Функція для оцінки моделі
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            running_loss += loss.item() * X_batch.size(0)
    
    return running_loss / len(data_loader.dataset)

# Функція для обчислення метрик
def calculate_metrics(y_true, y_pred):
    # Переводимо назад з логарифмічної шкали
    y_true_original = np.expm1(y_true)
    y_pred_original = np.expm1(y_pred)
    
    mse = mean_squared_error(y_true_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_original, y_pred_original)
    r2 = r2_score(y_true_original, y_pred_original)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def predict_and_evaluate(model_filename='rent_prediction_model.pth'):
    """
    Завантажує навчену модель і оцінює її на тестових даних
    """
    # Завантаження даних
    logger.info("Завантаження даних...")
    df = load_data()
    
    # Підготовка даних
    logger.info("Підготовка даних...")
    X_train, X_test, y_train, y_test, preprocessor, input_dim = prepare_data(df)
    
    # Створення DataLoader для тестових даних
    batch_size = 32
    test_dataset = HouseRentDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Завантаження моделі
    try:
        model = load_model(input_dim, model_filename)
    except FileNotFoundError as e:
        logger.error(f"{e}. Використовуйте train.py для навчання моделі.")
        return
    
    # Перенесення моделі на правильний пристрій
    model = model.to(device)
    
    # Оцінка моделі
    logger.info("Оцінка моделі на тестових даних...")
    criterion = torch.nn.MSELoss()
    val_loss = evaluate_model(model, test_loader, criterion, device)
    logger.info(f"Втрати на тестовій вибірці: {val_loss:.4f}")
    
    # Отримання передбачень
    logger.info("Генерація прогнозів...")
    model.eval()
    y_pred_list = []
    y_true_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            y_pred_list.extend(y_pred.flatten())
            y_true_list.extend(y_batch.numpy().flatten())
    
    # Обчислення метрик
    metrics = calculate_metrics(np.array(y_true_list), np.array(y_pred_list))
    logger.info("\nМетрики на тестовій вибірці:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # Візуалізація результатів
    logger.info("Створення графіків...")
    
    # Фактичні проти передбачених значень
    y_true_original = np.expm1(y_true_list)
    y_pred_original = np.expm1(y_pred_list)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true_original, y_pred_original, alpha=0.5)
    plt.plot([min(y_true_original), max(y_true_original)], 
             [min(y_true_original), max(y_true_original)], 
             'r--')
    plt.xlabel('Фактична вартість оренди')
    plt.ylabel('Прогнозована вартість оренди')
    plt.title('Фактичні проти прогнозованих значень')
    save_chart('true_vs_predicted.png')
    
    # Візуалізація розподілу помилок
    errors = y_pred_original - y_true_original
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel('Помилка прогнозу')
    plt.ylabel('Частота')
    plt.title('Розподіл помилок прогнозування')
    plt.axvline(x=0, color='r', linestyle='--')
    save_chart('error_distribution.png')
    
    # Залежність помилок від фактичних значень
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true_original, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Фактична вартість оренди')
    plt.ylabel('Помилка прогнозу')
    plt.title('Залежність помилок від фактичних значень')
    save_chart('error_vs_true.png')
    
    logger.info("\nОцінка моделі завершена успішно!")
    logger.info(f"Всі графіки збережені у директорії: {IMAGES_DIR}")

# Якщо файл запускається як скрипт, а не імпортується
if __name__ == "__main__":
    predict_and_evaluate()