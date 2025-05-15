import argparse
import os
from train import main as train_main
from predict import predict_and_evaluate

def main():
    parser = argparse.ArgumentParser(description='Система прогнозування вартості оренди житла')
    parser.add_argument('--action', type=str, default='train', choices=['train', 'predict'], 
                        help='Дія: train - навчання моделі, predict - оцінка та прогнозування')
    parser.add_argument('--model_path', type=str, default='rent_prediction_model.pth',
                        help='Шлях до файлу моделі (для завантаження або збереження)')
    
    args = parser.parse_args()
    
    if args.action == 'train':
        print("Запуск навчання моделі...")
        train_main()
    elif args.action == 'predict':
        if not os.path.exists(args.model_path):
            print(f"Помилка: файл моделі '{args.model_path}' не знайдено.")
            print("Спочатку виконайте навчання моделі за допомогою команди --action train")
            return
        
        print(f"Запуск прогнозування з використанням моделі {args.model_path}...")
        predict_and_evaluate(args.model_path)
    
    print("Робота програми завершена.")

if __name__ == "__main__":
    main()
