#!/usr/bin/env python3
"""
Hyperparameter Optimization GUI for the Learned Encoding Experiment

This application provides an interactive interface to tune the hyperparameters
of the LearnedEncodingModel and visualize its training performance in real-time.
"""

import sys
import time
from dataclasses import asdict

import torch
import torch.optim as optim
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit,
                             QMainWindow, QGroupBox)

# Matplotlib integration for plotting
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import necessary components from the main experiment script
from real_world_experiment import (
    ModelConfig,
    LearnedEncodingModel,
    get_dataloaders,
    train_epoch,
    evaluate
)


class TrainingWorker(QObject):
    """
    A worker that runs the model training in a separate thread to prevent
    the GUI from freezing.
    """
    # Signals to communicate with the main GUI thread
    progress = pyqtSignal(int, float, float, float)  # epoch, train_loss, val_loss, val_perplexity
    finished = pyqtSignal()
    log = pyqtSignal(str)

    def __init__(self, model_config: ModelConfig, epochs: int, lr: float):
        super().__init__()
        self.model_config = model_config
        self.epochs = epochs
        self.lr = lr
        self.is_running = True

    def run(self):
        """The main training logic that will be executed in the thread."""
        try:
            self.log.emit("--- Starting Training ---")

            # 1. Get Dataloaders
            self.log.emit("Loading data... (this may take a moment)")
            # Use a smaller batch size and sequence length for faster GUI feedback
            train_loader, val_loader, _, vocab_size = get_dataloaders(batch_size=16, seq_len=35)
            self.log.emit(f"Data loaded. Vocabulary size: {vocab_size}")

            # Update model_config with the actual vocab_size from the dataset
            self.model_config.vocab_size = vocab_size

            # 2. Initialize Model and Optimizer
            self.log.emit(f"Initializing model with config: {asdict(self.model_config)}")
            model = LearnedEncodingModel(self.model_config)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            criterion = torch.nn.CrossEntropyLoss()
            self.log.emit("Model and optimizer created.")

            # 3. Training Loop
            self.log.emit(f"Starting training for {self.epochs} epochs.")
            for epoch in range(self.epochs):
                if not self.is_running:
                    self.log.emit("Training stopped by user.")
                    break

                start_time = time.time()

                # Train for one epoch
                train_loss = train_epoch(model, train_loader, optimizer, criterion, self.model_config.vocab_size)

                # Evaluate on the validation set
                val_loss, val_ppl = evaluate(model, val_loader, criterion, self.model_config.vocab_size)

                epoch_duration = time.time() - start_time

                self.log.emit(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {train_loss:.3f} | "
                    f"Val Loss: {val_loss:.3f} | "
                    f"Val PPL: {val_ppl:7.3f} | "
                    f"Time: {epoch_duration:.2f}s"
                )

                # Emit progress signal to update the GUI
                self.progress.emit(epoch + 1, train_loss, val_loss, val_ppl)

            self.log.emit("--- Training Finished ---")

        except Exception as e:
            self.log.emit(f"An error occurred: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        self.is_running = False


class PlotCanvas(FigureCanvas):
    """A Matplotlib canvas widget to integrate into a PyQt6 application."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

    def plot(self, epoch, train_loss, val_loss):
        """Append new data and redraw the plot."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        self.axes.clear()
        self.axes.plot(self.epochs, self.train_losses, 'r-', label='Training Loss')
        self.axes.plot(self.epochs, self.val_losses, 'b-', label='Validation Loss')
        self.axes.set_title("Training and Validation Loss")
        self.axes.set_xlabel("Epoch")
        self.axes.set_ylabel("Loss")
        self.axes.legend()
        self.draw()

    def clear(self):
        """Clear the plot for a new run."""
        self.axes.clear()
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.draw()


class HyperparameterOptimizerGUI(QMainWindow):
    """The main window for the Hyperparameter Optimization GUI."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hyperparameter Optimizer for Learned Encoding")
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left side: Controls and Logs
        self.left_layout = QVBoxLayout()
        self.main_layout.addLayout(self.left_layout, 1)

        # Right side: Plot
        self.plot_canvas = PlotCanvas(self)
        self.main_layout.addWidget(self.plot_canvas, 2)

        # --- Controls Group ---
        self.controls_group = QGroupBox("Hyperparameters")
        self.form_layout = QFormLayout()
        self.controls_group.setLayout(self.form_layout)
        self.left_layout.addWidget(self.controls_group)

        # Hyperparameter input fields
        self.lr_input = QDoubleSpinBox()
        self.lr_input.setRange(0.00001, 0.1)
        self.lr_input.setSingleStep(0.0001)
        self.lr_input.setValue(0.001)
        self.lr_input.setDecimals(5)
        self.form_layout.addRow("Learning Rate:", self.lr_input)

        self.compressed_dim_input = QSpinBox()
        self.compressed_dim_input.setRange(4, 1024)
        self.compressed_dim_input.setValue(16)
        self.form_layout.addRow("Compressed Dimension:", self.compressed_dim_input)

        self.num_layers_input = QSpinBox()
        self.num_layers_input.setRange(1, 12)
        self.num_layers_input.setValue(2)
        self.form_layout.addRow("Num Transformer Layers:", self.num_layers_input)

        self.nhead_input = QSpinBox()
        self.nhead_input.setRange(1, 16)
        self.nhead_input.setValue(4)
        self.form_layout.addRow("Num Attention Heads:", self.nhead_input)

        self.hidden_dim_input = QSpinBox()
        self.hidden_dim_input.setRange(16, 2048)
        self.hidden_dim_input.setValue(64)
        self.form_layout.addRow("Hidden Dimension (FFN):", self.hidden_dim_input)

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 100)
        self.epochs_input.setValue(10)
        self.form_layout.addRow("Epochs:", self.epochs_input)

        # --- Action Buttons ---
        self.button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.stop_button)
        self.left_layout.addLayout(self.button_layout)

        # --- Log Area ---
        self.log_group = QGroupBox("Logs")
        self.log_layout = QVBoxLayout()
        self.log_group.setLayout(self.log_layout)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_layout.addWidget(self.log_area)
        self.left_layout.addWidget(self.log_group)

        self.worker = None
        self.thread = None

    def start_training(self):
        """Slot to start the training process in a background thread."""
        self.plot_canvas.clear()
        self.log_area.clear()
        self.append_log("Starting training process...")

        # Disable start button and enable stop button
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # 1. Gather hyperparameters from the GUI
        lr = self.lr_input.value()
        compressed_dim = self.compressed_dim_input.value()
        num_layers = self.num_layers_input.value()
        nhead = self.nhead_input.value()
        hidden_dim = self.hidden_dim_input.value()
        epochs = self.epochs_input.value()

        # Create ModelConfig. vocab_size will be set in the worker.
        model_config = ModelConfig(
            vocab_size=0, # Placeholder
            compressed_dim=compressed_dim,
            num_layers=num_layers,
            nhead=nhead,
            hidden_dim=hidden_dim,
            traditional_embedding_dim=128 # Not used by LearnedEncodingModel but required by dataclass
        )

        # 2. Set up the worker and thread
        self.thread = QThread()
        self.worker = TrainingWorker(model_config, epochs, lr)
        self.worker.moveToThread(self.thread)

        # 3. Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_training_finished)
        self.worker.progress.connect(self.update_plot)
        self.worker.log.connect(self.append_log)

        # 4. Start the thread
        self.thread.start()

    def stop_training(self):
        """Slot to stop the training process."""
        if self.worker:
            self.worker.stop()
        self.append_log("Stop signal sent to worker...")
        self.stop_button.setEnabled(False)

    def on_training_finished(self):
        """Slot to handle cleanup after training is finished."""
        self.append_log("Worker has finished.")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self.thread = None
        self.worker = None

    def update_plot(self, epoch, train_loss, val_loss, val_ppl):
        """Slot to update the plot with new data from the worker."""
        self.plot_canvas.plot(epoch, train_loss, val_loss)

    def append_log(self, message: str):
        """Append a message to the log area."""
        self.log_area.append(message)
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    def closeEvent(self, event):
        """Ensure the worker thread is stopped when the window is closed."""
        self.stop_training()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = HyperparameterOptimizerGUI()
    main_window.show()
    sys.exit(app.exec())
