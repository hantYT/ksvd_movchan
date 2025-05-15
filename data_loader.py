import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
from torch.utils.data import Dataset
import os
import requests
from io import StringIO
from utils import save_chart, logger

# Налаштування відтворюваності результатів
SEED = 42
np.random.seed(SEED)

# Клас для створення датасету PyTorch
class HouseRentDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Завантаження та огляд даних
def load_data(file_path=None):
    """
    Завантаження даних з файлу або з інтернету
    """
    # Якщо вказано локальний файл і він існує - використовуємо його
    if file_path and os.path.exists(file_path):
        logger.info(f"Завантаження даних з {file_path}")
        return pd.read_csv(file_path)
    
    # Спроба завантажити локальний файл з типовим ім'ям
    local_files = ['House_Rent_Dataset.csv', 'house_rent_dataset.csv']
    for local_file in local_files:
        if os.path.exists(local_file):
            logger.info(f"Завантаження даних з локального файлу {local_file}")
            return pd.read_csv(local_file)
    
    # Пробуємо завантажити дані з кількох джерел
    urls = [
        "https://raw.githubusercontent.com/datasets-io/house-rent-prediction/main/House_Rent_Dataset.csv",
        "https://raw.githubusercontent.com/iamsouravbanerjee/house-rent-prediction-dataset/main/House_Rent_Dataset.csv",
        "https://media.githubusercontent.com/media/iamsouravbanerjee/house-rent-prediction-dataset/main/House_Rent_Dataset.csv"
    ]
    
    # Спроба завантажити з Kaggle
    try:
        logger.info("Спроба завантажити дані з Kaggle...")
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('iamsouravbanerjee/house-rent-prediction-dataset', 
                                          path='./', unzip=True)
        if os.path.exists('House_Rent_Dataset.csv'):
            return pd.read_csv('House_Rent_Dataset.csv')
    except Exception as e:
        logger.warning(f"Не вдалося завантажити дані з Kaggle: {e}")

    # Спроба завантажити дані з GitHub або інших URL
    for url in urls:
        try:
            logger.info(f"Спроба завантажити дані з URL: {url}")
            response = requests.get(url)
            if response.status_code == 200:
                return pd.read_csv(StringIO(response.text))
        except Exception as e:
            logger.warning(f"Помилка завантаження з {url}: {e}")
    
    # Якщо всі попередні методи не спрацювали, створюємо тестовий датасет
    logger.warning("Не вдалося завантажити дані. Створюю тестовий датасет...")
    return create_test_dataset()

def create_test_dataset():
    """
    Створює тестовий датасет для випадку, коли не вдається завантажити реальні дані
    """
    np.random.seed(SEED)
    n_samples = 1000
    
    # Створення випадкових ознак
    bhk = np.random.randint(1, 6, n_samples)
    size = np.random.randint(500, 3000, n_samples)
    floor = np.random.randint(1, 10, n_samples)
    area_type = np.random.choice(['Carpet Area', 'Super Area', 'Built Area'], n_samples)
    city = np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Chennai'], n_samples)
    furnishing = np.random.choice(['Furnished', 'Semi-Furnished', 'Unfurnished'], n_samples)
    bathrooms = np.random.randint(1, 5, n_samples)
    
    # Створення цільової змінної (вартість оренди)
    # Додаємо деяку залежність від ознак для реалістичності
    rent = 5000 + 2000 * bhk + 10 * size + 1000 * bathrooms + np.random.normal(0, 5000, n_samples)
    rent = np.abs(rent)  # Забезпечуємо позитивні значення
    
    # Створення датафрейму
    df = pd.DataFrame({
        'BHK': bhk,
        'Size': size,
        'Floor': floor,
        'Area Type': area_type,
        'City': city,
        'Furnishing Status': furnishing,
        'Bathroom': bathrooms,
        'Rent': rent
    })
    
    logger.info("Створено тестовий датасет з формою: %s", df.shape)
    return df

def analyze_data(df):
    """
    Виконує базовий аналіз даних і повертає статистику
    """
    logger.info("Форма датасету: %s", df.shape)
    
    # Дослідження цільової змінної
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Rent'], kde=True)
    plt.title('Розподіл цільової змінної (Rent)')
    plt.xlabel('Вартість оренди')
    plt.ylabel('Частота')
    save_chart('rent_distribution.png')
    
    # Перевірка наявності викидів
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df['Rent'])
    plt.title('Boxplot вартості оренди')
    plt.ylabel('Вартість оренди')
    save_chart('rent_boxplot.png')
    
    # Кореляція між числовими даними
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Кореляційна матриця числових змінних')
    save_chart('correlation_matrix.png')
    
    return correlation_matrix

def prepare_data(df):
    """
    Підготовка даних для моделі
    """
    # Копіюємо датафрейм для безпеки
    data = df.copy()
    
    # Логарифмуємо цільову змінну для зменшення впливу викидів
    data['Rent'] = np.log1p(data['Rent'])
    
    # Виділяємо цільову змінну та ознаки
    X = data.drop('Rent', axis=1)
    y = data['Rent']
    
    # Визначаємо категоріальні та числові стовпці
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Створюємо препроцесор для даних
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ])
    
    # Розділення на тренувальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    # Застосовуємо препроцесор
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Якщо OneHotEncoder повернув розріджену матрицю, конвертуємо її в щільний масив
    if hasattr(X_train_preprocessed, "toarray"):
        X_train_preprocessed = X_train_preprocessed.toarray()
    if hasattr(X_test_preprocessed, "toarray"):
        X_test_preprocessed = X_test_preprocessed.toarray()
    
    # Перетворюємо дані в тензори PyTorch
    X_train_tensor = torch.FloatTensor(X_train_preprocessed)
    X_test_tensor = torch.FloatTensor(X_test_preprocessed)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)
    
    logger.info(f"Форма X_train_tensor: {X_train_tensor.shape}")
    logger.info(f"Форма X_test_tensor: {X_test_tensor.shape}")
    
    # Повертаємо також розмірність входу для створення моделі
    input_dim = X_train_tensor.shape[1]
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, preprocessor, input_dim

if __name__ == "__main__":
    # Для тестування модуля
    df = load_data()
    analyze_data(df)
    X_train, X_test, y_train, y_test, preprocessor, input_dim = prepare_data(df)
    logger.info(f"Розмірність вхідних даних: {input_dim}")
