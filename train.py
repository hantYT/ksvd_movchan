import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from data_loader import load_data, prepare_data, HouseRentDataset, analyze_data
from model import RentPredictionModel, save_model
from utils import save_chart, logger, MODELS_DIR

# Налаштування відтворюваності результатів
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Перевірка доступності GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Використовуємо пристрій: {device}")

def train_model(train_loader, test_loader, input_dim, device, num_epochs=100, learning_rate=0.001):
    """
    Функція для навчання моделі
    """
    # Створення моделі
    model = RentPredictionModel(input_dim)
    model.to(device)
    
    # Визначення функції втрат та оптимізатора
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Для зменшення швидкості навчання при відсутності прогресу
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            # Переносимо дані на відповідний пристрій
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Обчислюємо передбачення
            y_pred = model(X_batch)
            
            # Обчислюємо функцію втрат
            loss = criterion(y_pred, y_batch)
            
            # Обнуляємо градієнти, обчислюємо градієнти та оновлюємо ваги
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Обчислюємо втрати на валідаційній вибірці
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss = val_loss / len(test_loader.dataset)
        val_losses.append(val_loss)
        
        # Оновлюємо швидкість навчання
        scheduler.step(val_loss)
        
        # Додаємо власне логування зміни швидкості навчання
        current_lr = optimizer.param_groups[0]['lr']
        
        # Зберігаємо кращу модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, "best_rent_prediction_model.pth")
        
        # Виводимо прогрес лише для кожної 10-ї епохи щоб зменшити кількість логів
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f'Епоха [{epoch+1}/{num_epochs}], '
                  f'Втрати на тренувальних даних: {epoch_loss:.4f}, '
                  f'Втрати на тестових даних: {val_loss:.4f}, '
                  f'LR: {current_lr:.6f}')
    
    # Зберігаємо фінальну модель
    save_model(model, "rent_prediction_model.pth")
    
    training_time = time.time() - start_time
    logger.info(f"Навчання завершено за {training_time:.2f} секунд")
    
    # Візуалізація кривих навчання
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Тренувальна вибірка')
    plt.plot(range(1, num_epochs+1), val_losses, label='Тестова вибірка')
    plt.xlabel('Епохи')
    plt.ylabel('Втрати (MSE)')
    plt.title('Криві навчання')
    plt.legend()
    save_chart('learning_curves.png')
    
    return model

def main():
    # Завантаження даних
    logger.info("Завантаження даних...")
    df = load_data()
    
    # Аналіз даних
    logger.info("Аналіз даних...")
    analyze_data(df)
    
    # Підготовка даних
    logger.info("Підготовка даних...")
    X_train, X_test, y_train, y_test, preprocessor, input_dim = prepare_data(df)
    
    # Гіперпараметри
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 100
    
    # Створення DataLoader'ів
    train_dataset = HouseRentDataset(X_train, y_train)
    test_dataset = HouseRentDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Навчання моделі
    logger.info("Початок навчання моделі...")
    model = train_model(train_loader, test_loader, input_dim, device, num_epochs, learning_rate)
    
    logger.info("\nНавчання завершено успішно!")
    logger.info("Для оцінки моделі та отримання прогнозів використовуйте predict.py")

if __name__ == "__main__":
    main()
