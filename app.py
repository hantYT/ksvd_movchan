import sys
import os
import threading
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                           QWidget, QTabWidget, QLabel, QProgressBar, QTextEdit, QScrollArea,
                           QGridLayout, QFileDialog, QSpinBox, QDoubleSpinBox, QGroupBox,
                           QComboBox, QSplitter)
from PyQt5.QtGui import QPixmap, QTextCursor
from PyQt5.QtCore import Qt, pyqtSignal, QObject

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from data_loader import load_data, prepare_data, analyze_data, HouseRentDataset
from model import RentPredictionModel, save_model, load_model 
from utils import get_image_files, ensure_images_dir, ensure_models_dir, IMAGES_DIR, MODELS_DIR, save_chart
import logging

# Custom stream to redirect stdout to QTextEdit
class QTextEditLogger(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.setLevel(logging.INFO)
        self.setFormatter(logging.Formatter('%(message)s'))
        
    def emit(self, record):
        msg = self.format(record)
        self.text_widget.append(msg)
        self.text_widget.moveCursor(QTextCursor.End)

# Worker class to run training/prediction in a separate thread
class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    
    def __init__(self, task_type, params=None, console=None):
        super().__init__()
        self.task_type = task_type
        self.params = params or {}
        self.should_stop = False
        self.console = console
        
        # Configure logger
        self.logger = logging.getLogger("Worker")
        if console:
            handler = QTextEditLogger(console)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
    def run_training(self):
        try:
            # Get parameters from the GUI
            num_epochs = self.params.get('num_epochs', 100)
            learning_rate = self.params.get('learning_rate', 0.001)
            batch_size = self.params.get('batch_size', 32)
            
            # Load data
            self.logger.info("Завантаження даних...")
            df = load_data()
            
            # Analyze data
            self.logger.info("Аналіз даних...")
            analyze_data(df)
            
            # Prepare data
            self.logger.info("Підготовка даних...")
            X_train, X_test, y_train, y_test, preprocessor, input_dim = prepare_data(df)
            
            # Create datasets and dataloaders
            from torch.utils.data import DataLoader
            
            train_dataset = HouseRentDataset(X_train, y_train)
            test_dataset = HouseRentDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Train model
            self.logger.info(f"Початок навчання моделі з параметрами: epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}")
            
            # Determine device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"Використовуємо пристрій: {device}")
            
            # Custom training with progress updates
            self.train_with_progress(train_loader, test_loader, input_dim, device, num_epochs, learning_rate)
            
            self.logger.info("\nНавчання завершено успішно!")
            self.logger.info("Модель збережена в файлі rent_prediction_model.pth")
            
        except Exception as e:
            self.logger.error(f"Помилка під час навчання: {str(e)}")
        
        self.finished.emit()
    
    def train_with_progress(self, train_loader, test_loader, input_dim, device, num_epochs, learning_rate):
        # Create model
        model = RentPredictionModel(input_dim)
        model.to(device)
        
        # Loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # LR scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            if self.should_stop:
                self.logger.info("Навчання перервано користувачем")
                break
                
            model.train()
            running_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * X_batch.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)
            
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
            
            # Update LR
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, "best_rent_prediction_model.pth")
            
            # Report progress
            progress_percentage = int((epoch + 1) / num_epochs * 100)
            self.progress.emit(progress_percentage)
            
            # Log only every 10 epochs to reduce console clutter
            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.logger.info(f'Епоха [{epoch+1}/{num_epochs}], '
                        f'Втрати на тренувальних даних: {epoch_loss:.4f}, '
                        f'Втрати на тестових даних: {val_loss:.4f}, '
                        f'LR: {current_lr:.6f}')
        
        # Save final model
        save_model(model)
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Тренувальна вибірка')
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Тестова вибірка')
        plt.xlabel('Епохи')
        plt.ylabel('Втрати (MSE)')
        plt.title('Криві навчання')
        plt.legend()
        save_chart('learning_curves.png')
    
    def run_prediction(self):
        try:
            model_path = self.params.get('model_path', 'rent_prediction_model.pth')
            
            self.logger.info(f"Запуск прогнозування з моделлю: {model_path}")
            
            # Update progress bar
            self.progress.emit(10)
            
            # Simulate steps
            from predict import predict_and_evaluate
            
            # Temporarily redirect logging to our console
            root_logger = logging.getLogger()
            old_handlers = root_logger.handlers.copy()
            root_logger.handlers.clear()
            
            if self.console:
                handler = QTextEditLogger(self.console)
                root_logger.addHandler(handler)
            
            # Run prediction
            predict_and_evaluate(model_path)
            
            # Restore original handlers
            root_logger.handlers.clear()
            for handler in old_handlers:
                root_logger.addHandler(handler)
            
            self.progress.emit(100)
            
            self.logger.info("Прогнозування завершено успішно!")
            self.logger.info(f"Графіки збережені в директорії: {IMAGES_DIR}")
            
        except Exception as e:
            self.logger.error(f"Помилка під час прогнозування: {str(e)}")
        
        self.finished.emit()
    
    def run(self):
        if self.task_type == 'train':
            self.run_training()
        elif self.task_type == 'predict':
            self.run_prediction()
        else:
            self.logger.error(f"Невідомий тип завдання: {self.task_type}")
            self.finished.emit()
    
    def stop(self):
        self.should_stop = True
        self.logger.info("Зупинка процесу...")

class ImageGalleryWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        self.layout = QVBoxLayout(self)
        
        # Create scroll area for images
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        # Container widget for the grid layout
        self.scroll_content = QWidget()
        self.grid_layout = QGridLayout(self.scroll_content)
        
        self.scroll_area.setWidget(self.scroll_content)
        self.layout.addWidget(self.scroll_area)
        
        # Refresh button
        self.refresh_btn = QPushButton("Оновити галерею")
        self.refresh_btn.clicked.connect(self.load_images)
        self.layout.addWidget(self.refresh_btn)
        
        # Initialize with empty gallery
        self.load_images()
    
    def load_images(self):
        # Clear existing widgets from the grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        
        # Get image files
        image_files = get_image_files()
        
        if not image_files:
            label = QLabel("Немає зображень в директорії")
            label.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(label, 0, 0)
            return
        
        # Calculate grid dimensions
        max_cols = 2
        row, col = 0, 0
        
        for img_path in image_files:
            try:
                # Create container for image and label
                container = QWidget()
                container_layout = QVBoxLayout(container)
                
                # Create image label
                img_label = QLabel()
                pixmap = QPixmap(img_path)
                pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                img_label.setPixmap(pixmap)
                img_label.setAlignment(Qt.AlignCenter)
                
                # Create text label for the filename
                file_label = QLabel(os.path.basename(img_path))
                file_label.setAlignment(Qt.AlignCenter)
                
                # Add to container
                container_layout.addWidget(img_label)
                container_layout.addWidget(file_label)
                
                # Add to grid
                self.grid_layout.addWidget(container, row, col)
                
                # Update grid position
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
            
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

class RentPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker_thread = None
        self.worker = None
        
    def init_ui(self):
        self.setWindowTitle("Система прогнозування вартості оренди")
        self.resize(1000, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Create tabs
        self.train_tab = QWidget()
        self.predict_tab = QWidget()
        self.gallery_tab = QWidget()
        
        self.tabs.addTab(self.train_tab, "Навчання")
        self.tabs.addTab(self.predict_tab, "Прогнозування")
        self.tabs.addTab(self.gallery_tab, "Галерея зображень")
        
        # Setup train tab
        self.setup_train_tab()
        
        # Setup predict tab
        self.setup_predict_tab()
        
        # Setup gallery tab
        self.setup_gallery_tab()
        
        # Add tabs to main layout
        main_layout.addWidget(self.tabs)
        
        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # Create console output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(150)
        self.console.setStyleSheet("background-color: #f0f0f0; font-family: monospace;")
        main_layout.addWidget(self.console)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Ensure images directory exists
        ensure_images_dir()
        
        # Log app started
        self.log("Програма запущена")
        self.log(f"Директорія для зображень: {IMAGES_DIR}")
    
    def setup_train_tab(self):
        layout = QVBoxLayout(self.train_tab)
        
        # Parameters group box
        params_group = QGroupBox("Параметри навчання")
        params_layout = QGridLayout(params_group)
        
        # Epochs
        params_layout.addWidget(QLabel("Кількість епох:"), 0, 0)
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(100)
        params_layout.addWidget(self.epochs_spinbox, 0, 1)
        
        # Learning rate
        params_layout.addWidget(QLabel("Швидкість навчання:"), 1, 0)
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(0.0001, 0.1)
        self.lr_spinbox.setValue(0.001)
        self.lr_spinbox.setSingleStep(0.0001)
        self.lr_spinbox.setDecimals(4)
        params_layout.addWidget(self.lr_spinbox, 1, 1)
        
        # Batch size
        params_layout.addWidget(QLabel("Розмір пакету:"), 2, 0)
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(1, 256)
        self.batch_spinbox.setValue(32)
        params_layout.addWidget(self.batch_spinbox, 2, 1)
        
        layout.addWidget(params_group)
        
        # Buttons for training
        button_layout = QHBoxLayout()
        
        self.train_button = QPushButton("Почати навчання")
        self.train_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.train_button)
        
        self.stop_button = QPushButton("Зупинити")
        self.stop_button.clicked.connect(self.stop_task)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()
    
    def setup_predict_tab(self):
        layout = QVBoxLayout(self.predict_tab)
        
        # Model selection
        model_group = QGroupBox("Вибір моделі")
        model_layout = QHBoxLayout(model_group)
        
        self.model_path_edit = QTextEdit()
        self.model_path_edit.setMaximumHeight(30)
        self.model_path_edit.setText("rent_prediction_model.pth")
        model_layout.addWidget(self.model_path_edit)
        
        browse_button = QPushButton("Обрати...")
        browse_button.clicked.connect(self.browse_model)
        model_layout.addWidget(browse_button)
        
        layout.addWidget(model_group)
        
        # Predict button
        self.predict_button = QPushButton("Запустити прогнозування")
        self.predict_button.clicked.connect(self.start_prediction)
        layout.addWidget(self.predict_button)
        
        layout.addStretch()
    
    def setup_gallery_tab(self):
        layout = QVBoxLayout(self.gallery_tab)
        
        self.gallery = ImageGalleryWidget()
        layout.addWidget(self.gallery)
    
    def browse_model(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self, "Обрати модель", MODELS_DIR, "PyTorch Model (*.pth)"
        )
        if model_path:
            self.model_path_edit.setText(model_path)
    
    def start_training(self):
        # Get parameters from UI
        params = {
            'num_epochs': self.epochs_spinbox.value(),
            'learning_rate': self.lr_spinbox.value(),
            'batch_size': self.batch_spinbox.value()
        }
        
        # Start worker thread for training
        self.start_worker('train', params)
    
    def start_prediction(self):
        # Get model path from UI
        params = {
            'model_path': self.model_path_edit.toPlainText()
        }
        
        # Start worker thread for prediction
        self.start_worker('predict', params)
    
    def start_worker(self, task_type, params):
        # Disable buttons
        self.train_button.setEnabled(False)
        self.predict_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Reset progress bar
        self.progress_bar.setValue(0)
        
        # Clear console
        self.console.clear()
        
        # Log task starting
        self.log(f"Запуск завдання: {task_type}")
        
        # Create worker and thread
        self.worker = Worker(task_type, params, self.console)
        self.worker_thread = threading.Thread(target=self.worker.run)
        
        # Connect signals
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.progress.connect(self.update_progress)
        
        # Start thread
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def stop_task(self):
        if self.worker:
            self.worker.stop()
    
    def on_worker_finished(self):
        # Re-enable buttons
        self.train_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # Log task completed
        self.log("Завдання завершено")
        
        # Update gallery when task finishes
        if self.tabs.currentIndex() != 2:  # If not already on gallery tab
            self.gallery.load_images()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def log(self, message):
        self.console.append(message)
        self.console.moveCursor(QTextCursor.End)
    
    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker.stop()
            self.worker_thread.join(timeout=1.0)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RentPredictionApp()
    window.show()
    sys.exit(app.exec_())
