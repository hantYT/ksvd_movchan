## Встановлення та налаштування

1. Клонуйте репозиторій:
```
git clone https://github.com/hantYT/ksvd_movchan.git
```

2. Встановіть необхідні залежності:
```
pip install -r requirements.txt
```
Отримайте API Token "kaggle.json"

Перемістіть у папку свого локального юзера windows

3. Запустіть графічний інтерфейс:
```
python app.py
```

Або використовуйте консольний інтерфейс:
```
python main.py --action train  # для навчання моделі
python main.py --action predict  # для прогнозування
```

## Вимоги до системи

- Python 3.6+
- PyQt5
- PyTorch
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn


