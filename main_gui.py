#!/usr/bin/env python3
"""
Main GUI for the Learned Encoding Unified Experimentation Framework

This application provides a unified interface for running experiments,
visualizing results, and comparing models.
"""

import sys
import time
from dataclasses import asdict

import torch
import torch.optim as optim
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFormLayout, QSpinBox, QDoubleSpinBox, QTextEdit,
                             QMainWindow, QGroupBox, QTabWidget, QTableWidget,
                             QTableWidgetItem, QHeaderView)

# Matplotlib integration for plotting
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import necessary components from the main experiment script
from real_world_experiment import (
    ModelConfig,
    LearnedEncodingModel,
    TraditionalModel,
    SimpleAutoencoder,
    get_dataloaders,
    train_autoencoder,
    train_epoch,
    evaluate
)

# --- Worker for Optimizer Tab ---
class TrainingWorker(QObject):
    progress = pyqtSignal(int, float, float, float)
    finished = pyqtSignal(object)  # Emits the trained model object
    log = pyqtSignal(str)

    def __init__(self, model_config: ModelConfig, epochs: int, lr: float):
        super().__init__()
        self.model_config = model_config
        self.epochs = epochs
        self.lr = lr
        self.is_running = True
        self.model = None

    def run(self):
        try:
            self.log.emit("--- Starting Training ---")
            self.log.emit("Loading data...")
            train_loader, val_loader, _, vocab_size = get_dataloaders(batch_size=16, seq_len=35)
            self.log.emit(f"Data loaded. Vocab size: {vocab_size}")
            self.model_config.vocab_size = vocab_size

            self.log.emit(f"Initializing model with config: {asdict(self.model_config)}")
            self.model = LearnedEncodingModel(self.model_config)
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            criterion = torch.nn.CrossEntropyLoss()
            self.log.emit("Model and optimizer created.")

            self.log.emit(f"Starting training for {self.epochs} epochs.")
            for epoch in range(self.epochs):
                if not self.is_running:
                    self.log.emit("Training stopped by user.")
                    break
                start_time = time.time()
                train_loss = train_epoch(self.model, train_loader, optimizer, criterion, self.model_config.vocab_size)
                val_loss, val_ppl = evaluate(self.model, val_loader, criterion, self.model_config.vocab_size)
                epoch_duration = time.time() - start_time
                self.log.emit(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val PPL: {val_ppl:7.3f} | Time: {epoch_duration:.2f}s")
                self.progress.emit(epoch + 1, train_loss, val_loss, val_ppl)

            if self.is_running:
                self.log.emit("--- Training Finished ---")

        except Exception as e:
            self.log.emit(f"An error occurred: {e}")
            self.model = None # Ensure model is not returned on error
        finally:
            # Emit the model if training finished successfully, otherwise emit None
            self.finished.emit(self.model if self.is_running else None)

    def stop(self):
        self.is_running = False

# --- Worker for Comparison Tab ---
class ComparisonWorker(QObject):
    log = pyqtSignal(str)
    model_progress = pyqtSignal(str, int, float)
    model_finished = pyqtSignal(str, dict)
    finished = pyqtSignal(object, object) # Emits (learned_model, traditional_model)

    def __init__(self, epochs: int, compression_ratio: float):
        super().__init__()
        self.epochs = epochs
        self.compression_ratio = compression_ratio
        self.is_running = True
        self.learned_model = None
        self.traditional_model = None

    def run(self):
        try:
            self.log.emit("--- Starting Full Comparison Run ---")
            self.log.emit("Loading data...")
            train_loader, val_loader, _, vocab_size = get_dataloaders(batch_size=16, seq_len=35)
            self.log.emit(f"Data loaded. Vocab size: {vocab_size}")

            traditional_dim = 128
            compressed_dim = int(traditional_dim / self.compression_ratio)
            config = ModelConfig(vocab_size=vocab_size, traditional_embedding_dim=traditional_dim,
                                 compressed_dim=compressed_dim, num_layers=2, hidden_dim=128, nhead=4)
            self.log.emit(f"Using compression ratio {self.compression_ratio}:1 ({traditional_dim}D -> {compressed_dim}D)")

            if not self.is_running: return
            self.log.emit("\n--- Training LearnedEncodingModel ---")
            self.learned_model = LearnedEncodingModel(config)
            optimizer = optim.Adam(self.learned_model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            start_time = time.time()
            best_val_loss = float('inf')
            for epoch in range(self.epochs):
                if not self.is_running: break
                train_loss = self._train_epoch(self.learned_model, train_loader, optimizer, criterion, config.vocab_size)
                val_loss, _ = self._evaluate(self.learned_model, val_loader, criterion, config.vocab_size)
                self.log.emit(f"[Learned] Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
                self.model_progress.emit("Learned", epoch + 1, val_loss)
                if val_loss < best_val_loss: best_val_loss = val_loss
            learned_results = {'loss': best_val_loss, 'perplexity': torch.exp(torch.tensor(best_val_loss)).item(), 'time': time.time() - start_time}
            self.model_finished.emit("Learned", learned_results)
            self.log.emit("--- LearnedEncodingModel Finished ---")

            if not self.is_running: return
            self.log.emit("\n--- Training TraditionalModel ---")
            start_time = time.time()
            self.log.emit("[Traditional] Pre-training autoencoder...")
            autoencoder = SimpleAutoencoder(config.vocab_size, config.traditional_embedding_dim, config.compressed_dim)
            trained_ae = train_autoencoder(autoencoder, train_loader, epochs=max(1, self.epochs // 2))
            self.log.emit("[Traditional] Training main model...")
            self.traditional_model = TraditionalModel(config, trained_ae)
            optimizer = optim.Adam(self.traditional_model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            best_val_loss = float('inf')
            for epoch in range(self.epochs):
                if not self.is_running: break
                train_loss = self._train_epoch(self.traditional_model, train_loader, optimizer, criterion, config.vocab_size)
                val_loss, _ = self._evaluate(self.traditional_model, val_loader, criterion, config.vocab_size)
                self.log.emit(f"[Traditional] Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
                self.model_progress.emit("Traditional", epoch + 1, val_loss)
                if val_loss < best_val_loss: best_val_loss = val_loss
            traditional_results = {'loss': best_val_loss, 'perplexity': torch.exp(torch.tensor(best_val_loss)).item(), 'time': time.time() - start_time}
            self.model_finished.emit("Traditional", traditional_results)
            self.log.emit("--- TraditionalModel Finished ---")
        except Exception as e:
            self.log.emit(f"An error occurred: {e}")
            self.learned_model = None
            self.traditional_model = None
        finally:
            self.log.emit("\n--- Comparison Finished ---")
            self.finished.emit(self.learned_model, self.traditional_model)

    def stop(self):
        self.is_running = False

    def _train_epoch(self, model, dataloader, optimizer, criterion, vocab_size):
        model.train(); total_loss = 0
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
        model.eval(); total_loss = 0
        with torch.no_grad():
            for data, targets in dataloader:
                if not self.is_running: break
                output = model(data)
                loss = criterion(output.view(-1, vocab_size), targets.view(-1))
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        return avg_loss, torch.exp(torch.tensor(avg_loss))

# --- Plot Canvases ---
class OptimizerPlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.clear()
    def plot(self, epoch, train_loss, val_loss):
        self.epochs.append(epoch); self.train_losses.append(train_loss); self.val_losses.append(val_loss)
        self.axes.clear()
        self.axes.plot(self.epochs, self.train_losses, 'r-', label='Training Loss')
        self.axes.plot(self.epochs, self.val_losses, 'b-', label='Validation Loss')
        self.axes.set_title("Training and Validation Loss"); self.axes.set_xlabel("Epoch"); self.axes.set_ylabel("Loss")
        self.axes.legend(); self.draw()
    def clear(self):
        self.train_losses, self.val_losses, self.epochs = [], [], []
        self.axes.clear(); self.draw()

class ComparisonPlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(parent)
        self.clear()
    def update_plot(self, model_name, epoch, val_loss):
        if model_name == "Learned": self.learned_epochs.append(epoch); self.learned_losses.append(val_loss)
        elif model_name == "Traditional": self.trad_epochs.append(epoch); self.trad_losses.append(val_loss)
        self.axes.clear()
        if self.learned_epochs: self.axes.plot(self.learned_epochs, self.learned_losses, 'r-', label='Learned Model Loss')
        if self.trad_epochs: self.axes.plot(self.trad_epochs, self.trad_losses, 'b--', label='Traditional Model Loss')
        self.axes.set_title("Validation Loss Comparison"); self.axes.set_xlabel("Epoch"); self.axes.set_ylabel("Validation Loss")
        self.axes.legend(); self.draw()
    def clear(self):
        self.learned_epochs, self.learned_losses = [], []; self.trad_epochs, self.trad_losses = [], []
        self.axes.clear(); self.axes.legend(); self.draw()

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser
from experiment_analysis import (
    generate_embedding_visualization,
    compare_embedding_spaces,
    generate_markdown_report
)

# --- Reusable Dialog for Displaying Analysis Results ---
class ResultDialog(QDialog):
    """A dialog to display a matplotlib figure and a text summary."""
    def __init__(self, figure: Figure, summary_text: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis Result")
        self.setGeometry(150, 150, 800, 600)

        layout = QVBoxLayout(self)

        # Add the matplotlib canvas
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)

        # Add the summary text if provided
        if summary_text:
            summary_box = QTextBrowser()
            summary_box.setMarkdown(summary_text)
            summary_box.setFixedHeight(150) # Give it a fixed height
            layout.addWidget(summary_box)

# --- Tab Widgets ---
class OptimizerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QHBoxLayout(self)
        self.left_layout = QVBoxLayout(); self.main_layout.addLayout(self.left_layout, 1)
        self.plot_canvas = OptimizerPlotCanvas(self); self.main_layout.addWidget(self.plot_canvas, 2)
        self.controls_group = QGroupBox("Hyperparameters"); self.form_layout = QFormLayout()
        self.controls_group.setLayout(self.form_layout); self.left_layout.addWidget(self.controls_group)
        self.lr_input = QDoubleSpinBox(); self.lr_input.setRange(0.00001, 0.1); self.lr_input.setSingleStep(0.0001); self.lr_input.setValue(0.001); self.lr_input.setDecimals(5)
        self.form_layout.addRow("Learning Rate:", self.lr_input)
        self.compressed_dim_input = QSpinBox(); self.compressed_dim_input.setRange(4, 1024); self.compressed_dim_input.setValue(16)
        self.form_layout.addRow("Compressed Dimension:", self.compressed_dim_input)
        self.num_layers_input = QSpinBox(); self.num_layers_input.setRange(1, 12); self.num_layers_input.setValue(2)
        self.form_layout.addRow("Num Transformer Layers:", self.num_layers_input)
        self.nhead_input = QSpinBox(); self.nhead_input.setRange(1, 16); self.nhead_input.setValue(4)
        self.form_layout.addRow("Num Attention Heads:", self.nhead_input)
        self.hidden_dim_input = QSpinBox(); self.hidden_dim_input.setRange(16, 2048); self.hidden_dim_input.setValue(64)
        self.form_layout.addRow("Hidden Dimension (FFN):", self.hidden_dim_input)
        self.epochs_input = QSpinBox(); self.epochs_input.setRange(1, 100); self.epochs_input.setValue(10)
        self.form_layout.addRow("Epochs:", self.epochs_input)
        self.button_layout = QHBoxLayout(); self.start_button = QPushButton("Start Training"); self.start_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Stop Training"); self.stop_button.clicked.connect(self.stop_training); self.stop_button.setEnabled(False)
        self.button_layout.addWidget(self.start_button); self.button_layout.addWidget(self.stop_button); self.left_layout.addLayout(self.button_layout)

        # --- Analysis Section (Progressive Disclosure) ---
        self.analysis_group = QGroupBox("Analysis")
        self.analysis_layout = QVBoxLayout()
        self.analysis_group.setLayout(self.analysis_layout)
        self.visualize_button = QPushButton("Visualize Embeddings (t-SNE)")
        self.visualize_button.clicked.connect(self.visualize_embeddings)
        self.visualize_button.setEnabled(False) # Disabled until training is complete
        self.analysis_layout.addWidget(self.visualize_button)
        self.left_layout.addWidget(self.analysis_group)

        self.log_group = QGroupBox("Logs"); self.log_layout = QVBoxLayout(); self.log_group.setLayout(self.log_layout)
        self.log_area = QTextEdit(); self.log_area.setReadOnly(True); self.log_layout.addWidget(self.log_area); self.left_layout.addWidget(self.log_group)
        self.worker = None; self.thread = None
        self.trained_model = None # To store the model after training
    def start_training(self):
        self.plot_canvas.clear(); self.log_area.clear(); self.append_log("Starting training process...")
        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.visualize_button.setEnabled(False) # Disable on new run
        self.trained_model = None

        model_config = ModelConfig(vocab_size=0, compressed_dim=self.compressed_dim_input.value(), num_layers=self.num_layers_input.value(),
                                   nhead=self.nhead_input.value(), hidden_dim=self.hidden_dim_input.value(), traditional_embedding_dim=128)
        self.thread = QThread(); self.worker = TrainingWorker(model_config, self.epochs_input.value(), self.lr_input.value())
        self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_training_finished); self.worker.progress.connect(self.update_plot); self.worker.log.connect(self.append_log)
        self.thread.start()
    def stop_training(self):
        if self.worker: self.worker.stop()
        self.append_log("Stop signal sent to worker..."); self.stop_button.setEnabled(False)

    def on_training_finished(self, model):
        self.append_log("Worker has finished.")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        if model:
            self.trained_model = model
            self.visualize_button.setEnabled(True)
            self.append_log("✅ Model training complete. Analysis is now available.")
        else:
            self.append_log("❌ Training failed or was stopped. No model available for analysis.")

        if self.thread: self.thread.quit(); self.thread.wait()
        self.thread = None; self.worker = None

    def visualize_embeddings(self):
        if not self.trained_model:
            self.append_log("Cannot visualize: No model has been trained successfully.")
            return

        self.append_log("Generating embedding visualization...")
        try:
            # Call the metamodel function
            fig = generate_embedding_visualization(self.trained_model)
            # Display in a dialog
            dialog = ResultDialog(fig, "", self) # No summary text for this one
            dialog.exec()
            self.append_log("Visualization dialog closed.")
        except Exception as e:
            self.append_log(f"Error during visualization: {e}")

    def update_plot(self, epoch, train_loss, val_loss, val_ppl): self.plot_canvas.plot(epoch, train_loss, val_loss)
    def append_log(self, message: str): self.log_area.append(message); self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())
    def stop_worker_thread(self):
        if self.worker: self.worker.stop()
        if self.thread and self.thread.isRunning(): self.thread.quit(); self.thread.wait()

class ComparisonTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.top_layout = QHBoxLayout(); self.main_layout.addLayout(self.top_layout)
        self.controls_group = QGroupBox("Controls"); self.form_layout = QFormLayout(); self.controls_group.setLayout(self.form_layout)
        self.epochs_input = QSpinBox(); self.epochs_input.setRange(1, 50); self.epochs_input.setValue(5)
        self.form_layout.addRow("Epochs:", self.epochs_input)
        self.ratio_input = QSpinBox(); self.ratio_input.setRange(2, 64); self.ratio_input.setValue(8)
        self.form_layout.addRow("Compression Ratio (N:1):", self.ratio_input)
        self.start_button = QPushButton("Start Comparison"); self.start_button.clicked.connect(self.start_comparison)
        self.form_layout.addRow(self.start_button); self.top_layout.addWidget(self.controls_group, 1)

        # --- Analysis Section ---
        self.compare_button = QPushButton("Compare Embedding Spaces (CCA)")
        self.compare_button.clicked.connect(self.compare_embeddings)
        self.compare_button.setEnabled(False)
        self.form_layout.addRow(self.compare_button)

        self.results_table = QTableWidget(); self.results_table.setRowCount(3); self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Learned Model", "Traditional Model"]); self.results_table.setVerticalHeaderLabels(["Final Val Loss", "Final Perplexity", "Training Time (s)"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); self.top_layout.addWidget(self.results_table, 2)
        self.bottom_layout = QHBoxLayout(); self.main_layout.addLayout(self.bottom_layout)
        self.plot_canvas = ComparisonPlotCanvas(self); self.bottom_layout.addWidget(self.plot_canvas, 2)
        self.log_area = QTextEdit(); self.log_area.setReadOnly(True); self.bottom_layout.addWidget(self.log_area, 1)
        self.thread = None; self.worker = None
        self.trained_learned_model = None
        self.trained_traditional_model = None

    def start_comparison(self):
        self.start_button.setEnabled(False); self.plot_canvas.clear(); self.log_area.clear(); self.results_table.clearContents()
        self.compare_button.setEnabled(False)
        self.trained_learned_model = None
        self.trained_traditional_model = None
        self.append_log("Starting comparison...")
        self.thread = QThread(); self.worker = ComparisonWorker(self.epochs_input.value(), self.ratio_input.value())
        self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_comparison_finished); self.worker.log.connect(self.append_log)
        self.worker.model_progress.connect(self.plot_canvas.update_plot); self.worker.model_finished.connect(self.update_table)
        self.thread.start()

    def on_comparison_finished(self, learned_model, traditional_model):
        self.start_button.setEnabled(True)
        if learned_model and traditional_model:
            self.trained_learned_model = learned_model
            self.trained_traditional_model = traditional_model
            self.compare_button.setEnabled(True)
            self.append_log("✅ Comparison finished. Embedding space analysis is now available.")
        else:
            self.append_log("❌ Comparison failed or was stopped. No models available for analysis.")

        if self.thread: self.thread.quit(); self.thread.wait()
        self.thread = None; self.worker = None

    def compare_embeddings(self):
        if not self.trained_learned_model or not self.trained_traditional_model:
            self.append_log("Cannot compare: Both models must be trained successfully.")
            return

        self.append_log("Generating embedding space comparison (CCA)...")
        try:
            # Call the metamodel function
            fig, summary = compare_embedding_spaces(self.trained_learned_model, self.trained_traditional_model)
            # Display in a dialog
            dialog = ResultDialog(fig, summary, self)
            dialog.exec()
            self.append_log("Comparison dialog closed.")
        except Exception as e:
            self.append_log(f"Error during comparison: {e}")

    def update_table(self, model_name, results):
        col = 0 if model_name == "Learned" else 1
        self.results_table.setItem(0, col, QTableWidgetItem(f"{results['loss']:.4f}"))
        self.results_table.setItem(1, col, QTableWidgetItem(f"{results['perplexity']:.2f}"))
        self.results_table.setItem(2, col, QTableWidgetItem(f"{results['time']:.2f}"))
    def append_log(self, message): self.log_area.append(message); self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())
    def stop_worker_thread(self):
        if self.worker: self.worker.stop()
        if self.thread and self.thread.isRunning(): self.thread.quit(); self.thread.wait()

class MainGUI(QMainWindow):
    """The main window for the unified GUI."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Learned Encoding: Unified Experimentation Framework")
        self.setGeometry(100, 100, 1000, 700)
        self.tab_widget = QTabWidget(); self.setCentralWidget(self.tab_widget)
        self.optimizer_tab = OptimizerTab(); self.tab_widget.addTab(self.optimizer_tab, "Hyperparameter Optimizer")
        self.comparison_tab = ComparisonTab(); self.tab_widget.addTab(self.comparison_tab, "Model Comparison")
    def closeEvent(self, event):
        """Ensure worker threads are stopped when the window is closed."""
        self.optimizer_tab.stop_worker_thread()
        self.comparison_tab.stop_worker_thread()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainGUI()
    main_window.show()
    sys.exit(app.exec())
