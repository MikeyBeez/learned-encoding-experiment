#!/usr/bin/env python3
"""
End-to-End Training and Comparison GUI

This application provides an interface to run a full end-to-end training
comparison between the LearnedEncodingModel and the TraditionalModel,
visualizing their performance side-by-side.
"""

import sys
import time
from dataclasses import asdict

import torch
import torch.optim as optim
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFormLayout, QSpinBox, QTextEdit, QMainWindow, QGroupBox,
                             QTableWidget, QTableWidgetItem, QHeaderView)

# Matplotlib integration
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import components from the experiment script
from real_world_experiment import (
    ModelConfig,
    LearnedEncodingModel,
    TraditionalModel,
    SimpleAutoencoder,
    get_dataloaders,
    train_autoencoder,
    run_language_model_training
)

class ComparisonWorker(QObject):
    """
    Worker to run the full comparison training in a background thread.
    """
    log = pyqtSignal(str)
    model_progress = pyqtSignal(str, int, float)  # model_name, epoch, val_loss
    model_finished = pyqtSignal(str, dict)       # model_name, results_dict
    finished = pyqtSignal()

    def __init__(self, epochs: int, compression_ratio: float):
        super().__init__()
        self.epochs = epochs
        self.compression_ratio = compression_ratio
        self.is_running = True

    def run(self):
        """Main work method."""
        try:
            self.log.emit("--- Starting Full Comparison Run ---")

            # --- Setup ---
            self.log.emit("Loading data...")
            train_loader, val_loader, _, vocab_size = get_dataloaders(batch_size=16, seq_len=35)
            self.log.emit(f"Data loaded. Vocab size: {vocab_size}")

            traditional_dim = 128
            compressed_dim = int(traditional_dim / self.compression_ratio)

            config = ModelConfig(
                vocab_size=vocab_size,
                traditional_embedding_dim=traditional_dim,
                compressed_dim=compressed_dim,
                num_layers=2,
                hidden_dim=128,
                nhead=4
            )
            self.log.emit(f"Using compression ratio {self.compression_ratio}:1 ({traditional_dim}D -> {compressed_dim}D)")

            # --- 1. Train Learned Encoding Model ---
            if not self.is_running: return
            self.log.emit("\n--- Training LearnedEncodingModel ---")
            learned_model = LearnedEncodingModel(config)

            # We need a custom training loop to emit signals
            optimizer = optim.Adam(learned_model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            start_time = time.time()
            best_val_loss = float('inf')

            for epoch in range(self.epochs):
                if not self.is_running: break
                train_loss = self._train_epoch(learned_model, train_loader, optimizer, criterion, config.vocab_size)
                val_loss, _ = self._evaluate(learned_model, val_loader, criterion, config.vocab_size)
                self.log.emit(f"[Learned] Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
                self.model_progress.emit("Learned", epoch + 1, val_loss)
                if val_loss < best_val_loss: best_val_loss = val_loss

            learned_results = {
                'loss': best_val_loss,
                'perplexity': torch.exp(torch.tensor(best_val_loss)).item(),
                'time': time.time() - start_time
            }
            self.model_finished.emit("Learned", learned_results)
            self.log.emit("--- LearnedEncodingModel Finished ---")

            # --- 2. Train Traditional Model ---
            if not self.is_running: return
            self.log.emit("\n--- Training TraditionalModel ---")

            start_time = time.time()

            # Pre-train autoencoder
            self.log.emit("[Traditional] Pre-training autoencoder...")
            autoencoder = SimpleAutoencoder(config.vocab_size, config.traditional_embedding_dim, config.compressed_dim)
            trained_ae = train_autoencoder(autoencoder, train_loader, epochs=max(1, self.epochs // 2))

            # Train main model
            self.log.emit("[Traditional] Training main model...")
            traditional_model = TraditionalModel(config, trained_ae)
            optimizer = optim.Adam(traditional_model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            best_val_loss = float('inf')
            for epoch in range(self.epochs):
                if not self.is_running: break
                train_loss = self._train_epoch(traditional_model, train_loader, optimizer, criterion, config.vocab_size)
                val_loss, _ = self._evaluate(traditional_model, val_loader, criterion, config.vocab_size)
                self.log.emit(f"[Traditional] Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
                self.model_progress.emit("Traditional", epoch + 1, val_loss)
                if val_loss < best_val_loss: best_val_loss = val_loss

            traditional_results = {
                'loss': best_val_loss,
                'perplexity': torch.exp(torch.tensor(best_val_loss)).item(),
                'time': time.time() - start_time
            }
            self.model_finished.emit("Traditional", traditional_results)
            self.log.emit("--- TraditionalModel Finished ---")

        except Exception as e:
            self.log.emit(f"An error occurred: {e}")
        finally:
            self.log.emit("\n--- Comparison Finished ---")
            self.finished.emit()

    def stop(self):
        self.is_running = False

    # Duplicating these methods here to avoid modifying the original script.
    # A better solution might be to refactor real_world_experiment.py to be more modular.
    def _train_epoch(self, model, dataloader, optimizer, criterion, vocab_size):
        model.train()
        total_loss = 0
        for data, targets in dataloader:
            if not self.is_running: break
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def _evaluate(self, model, dataloader, criterion, vocab_size):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, targets in dataloader:
                if not self.is_running: break
                output = model(data)
                loss = criterion(output.view(-1, vocab_size), targets.view(-1))
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        return avg_loss, torch.exp(torch.tensor(avg_loss))


class ComparisonPlotCanvas(FigureCanvas):
    """A Matplotlib canvas for side-by-side comparison plotting."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(parent)
        self.clear()

    def update_plot(self, model_name, epoch, val_loss):
        """Update one of the model's lines."""
        if model_name == "Learned":
            self.learned_epochs.append(epoch)
            self.learned_losses.append(val_loss)
        elif model_name == "Traditional":
            self.trad_epochs.append(epoch)
            self.trad_losses.append(val_loss)

        self.axes.clear()
        if self.learned_epochs:
            self.axes.plot(self.learned_epochs, self.learned_losses, 'r-', label='Learned Model Loss')
        if self.trad_epochs:
            self.axes.plot(self.trad_epochs, self.trad_losses, 'b--', label='Traditional Model Loss')
        self.axes.set_title("Validation Loss Comparison")
        self.axes.set_xlabel("Epoch")
        self.axes.set_ylabel("Validation Loss")
        self.axes.legend()
        self.draw()

    def clear(self):
        self.learned_epochs, self.learned_losses = [], []
        self.trad_epochs, self.trad_losses = [], []
        self.axes.clear()
        self.axes.legend()
        self.draw()


class ModelComparisonGUI(QMainWindow):
    """The main window for the Model Comparison GUI."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Learned vs. Traditional Model Comparison")
        self.setGeometry(100, 100, 900, 600)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Top section: Controls and Results Table
        self.top_layout = QHBoxLayout()
        self.main_layout.addLayout(self.top_layout)

        # --- Controls ---
        self.controls_group = QGroupBox("Controls")
        self.form_layout = QFormLayout()
        self.controls_group.setLayout(self.form_layout)

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 50)
        self.epochs_input.setValue(5)
        self.form_layout.addRow("Epochs:", self.epochs_input)

        self.ratio_input = QSpinBox()
        self.ratio_input.setRange(2, 64)
        self.ratio_input.setValue(8)
        self.form_layout.addRow("Compression Ratio (N:1):", self.ratio_input)

        self.start_button = QPushButton("Start Comparison")
        self.start_button.clicked.connect(self.start_comparison)
        self.form_layout.addRow(self.start_button)

        self.top_layout.addWidget(self.controls_group, 1)

        # --- Results Table ---
        self.results_table = QTableWidget()
        self.results_table.setRowCount(3)
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Learned Model", "Traditional Model"])
        self.results_table.setVerticalHeaderLabels(["Final Val Loss", "Final Perplexity", "Training Time (s)"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.top_layout.addWidget(self.results_table, 2)

        # Bottom section: Plot and Logs
        self.bottom_layout = QHBoxLayout()
        self.main_layout.addLayout(self.bottom_layout)

        self.plot_canvas = ComparisonPlotCanvas(self)
        self.bottom_layout.addWidget(self.plot_canvas, 2)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.bottom_layout.addWidget(self.log_area, 1)

        self.thread = None
        self.worker = None

    def start_comparison(self):
        self.start_button.setEnabled(False)
        self.plot_canvas.clear()
        self.log_area.clear()
        self.results_table.clearContents()
        self.append_log("Starting comparison...")

        epochs = self.epochs_input.value()
        ratio = self.ratio_input.value()

        self.thread = QThread()
        self.worker = ComparisonWorker(epochs, ratio)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_comparison_finished)
        self.worker.log.connect(self.append_log)
        self.worker.model_progress.connect(self.plot_canvas.update_plot)
        self.worker.model_finished.connect(self.update_table)

        self.thread.start()

    def on_comparison_finished(self):
        self.start_button.setEnabled(True)
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self.thread = None
        self.worker = None

    def update_table(self, model_name, results):
        col = 0 if model_name == "Learned" else 1
        self.results_table.setItem(0, col, QTableWidgetItem(f"{results['loss']:.4f}"))
        self.results_table.setItem(1, col, QTableWidgetItem(f"{results['perplexity']:.2f}"))
        self.results_table.setItem(2, col, QTableWidgetItem(f"{results['time']:.2f}"))

    def append_log(self, message):
        self.log_area.append(message)
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    def closeEvent(self, event):
        if self.worker: self.worker.stop()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = ModelComparisonGUI()
    main_window.show()
    sys.exit(app.exec())
