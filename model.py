import torch
import torch.nn as nn
import os
from utils import logger, ensure_models_dir, MODELS_DIR

# Визначення моделі нейронної мережі
class RentPredictionModel(nn.Module):
    def __init__(self, input_dim):
        super(RentPredictionModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        x = self.relu(self.layer2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        x = self.relu(self.layer3(x))
        x = self.batch_norm3(x)
        x = self.dropout(x)
        
        x = self.layer4(x)
        return x

def save_model(model, filename="rent_prediction_model.pth"):
    """
    Зберігає навчену модель на диск в папку models
    """
    ensure_models_dir()  # Make sure the models directory exists
    path = os.path.join(MODELS_DIR, filename)
    torch.save(model.state_dict(), path)
    logger.info(f"Модель збережена за шляхом: {path}")
    return path

def load_model(input_dim, filename="rent_prediction_model.pth"):
    """
    Завантажує навчену модель з диску з папки models
    """
    ensure_models_dir()  # Make sure the models directory exists
    path = os.path.join(MODELS_DIR, filename)
    
    # Also check in the current directory for backward compatibility
    if not os.path.exists(path) and os.path.exists(filename):
        path = filename
        logger.warning(f"Використовуємо модель з поточної директорії: {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл моделі не знайдено: {path}")
        
    model = RentPredictionModel(input_dim)
    model.load_state_dict(torch.load(path))
    model.eval()  # Переводимо модель в режим оцінки
    logger.info(f"Модель завантажена з: {path}")
    return model
