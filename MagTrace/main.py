import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QListWidget,
                             QLabel, QComboBox, QStackedWidget, QRadioButton,
                             QButtonGroup, QGridLayout, QLineEdit, QSlider,
                             QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.signal import savgol_filter
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QMovie



class QRangeSlider(QWidget):
    valueChanged = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.min_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_slider = QSlider(Qt.Orientation.Horizontal)

        layout.addWidget(self.min_slider)
        layout.addWidget(self.max_slider)

        self.min_slider.valueChanged.connect(self.update_range)
        self.max_slider.valueChanged.connect(self.update_range)

    def setRange(self, minimum, maximum):
        self.min_slider.setRange(minimum, maximum)
        self.max_slider.setRange(minimum, maximum)
        self.max_slider.setValue(maximum)

    def value(self):
        return (self.min_slider.value(), self.max_slider.value())

    def setValue(self, value):
        self.min_slider.setValue(value[0])
        self.max_slider.setValue(value[1])

    def update_range(self):
        if self.min_slider.value() > self.max_slider.value():
            self.min_slider.setValue(self.max_slider.value())
        self.valueChanged.emit(self.value())


class SharedDataManager:
    def __init__(self):
        self.cleaned_files = []

    def add_cleaned_file(self, file_path):
        if file_path not in self.cleaned_files:
            self.cleaned_files.append(file_path)

    def get_cleaned_files(self):
        return self.cleaned_files


class DataCleanerUI(QMainWindow):
    def __init__(self, shared_data_manager):
        super().__init__()
        self.shared_data_manager = shared_data_manager
        self.setWindowTitle("Data Cleaner")
        self.setGeometry(100, 100, 1400, 800)

        # Initialize data storage
        self.df = None
        self.selected_columns = []
        self.exclude_regions = []
        self.column_scales = {}
        self.column_offsets = {}

        # Create UI
        self.setup_ui()

    def update_scaling_controls(self):
        # Clear existing scaling controls
        for i in reversed(range(self.scaling_layout.count())):
            widget = self.scaling_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Add scaling controls for selected columns
        selected_items = self.column_list.selectedItems()
        for item in selected_items:
            col_name = item.text()
            scaling_group = QWidget()
            scaling_layout = QHBoxLayout(scaling_group)

            # Add column label
            scaling_layout.addWidget(QLabel(col_name))

            # Add "Scale:" label before scaling combo box
            scaling_layout.addWidget(QLabel("Scale:"))
            # Add scaling combo box
            scale_combo = QComboBox()
            scale_combo.addItems(['1x', '÷10', '÷100', '÷1000'])
            scale_combo.setCurrentText(self.column_scales.get(col_name, '1x'))
            scale_combo.currentTextChanged.connect(
                lambda text, col=col_name: self.update_column_scale(col, text))
            scaling_layout.addWidget(scale_combo)

            # Add "Offset:" label before offset input
            scaling_layout.addWidget(QLabel("Offset:"))
            # Add offset input
            offset_input = QLineEdit()
            offset_input.setPlaceholderText("Offset")
            offset_input.setFixedWidth(60)
            offset_input.setText(str(self.column_offsets.get(col_name, 0)))
            offset_input.editingFinished.connect(
                lambda col=col_name, input=offset_input: self.update_column_offset(col, input))
            scaling_layout.addWidget(offset_input)

            self.scaling_layout.addWidget(scaling_group)

        self.update_plot()

    def update_column_offset(self, column, input_field):
        try:
            value = float(input_field.text())
            self.column_offsets[column] = value
        except ValueError:
            self.column_offsets[column] = 0.0
        self.update_plot()

    def update_column_scale(self, column, scale):
        self.column_scales[column] = scale
        self.update_plot()

    def setup_ui(self):
        # Main widget setup
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left panel setup
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Add controls to left panel
        load_button = QPushButton("Load Data File")
        load_button.clicked.connect(self.load_file)
        left_layout.addWidget(load_button)

        # Column selection
        self.column_list = QListWidget()
        self.column_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.column_list.itemSelectionChanged.connect(self.update_scaling_controls)
        left_layout.addWidget(QLabel("Select Columns:"))
        left_layout.addWidget(self.column_list)

        # Scaling controls
        self.scaling_widget = QWidget()
        self.scaling_layout = QVBoxLayout(self.scaling_widget)
        left_layout.addWidget(QLabel("Column Scaling (Multiply/Divide) and Offset:"))
        left_layout.addWidget(self.scaling_widget)

        # Time range controls
        self.setup_time_controls(left_layout)

        # --- Exclude region time input controls ---
        self.exclude_start_input = QLineEdit()
        self.exclude_start_input.setPlaceholderText("Start")
        self.exclude_start_input.setFixedWidth(70)

        self.exclude_end_input = QLineEdit()
        self.exclude_end_input.setPlaceholderText("End")
        self.exclude_end_input.setFixedWidth(70)

        exclude_time_layout = QHBoxLayout()
        exclude_time_layout.addWidget(QLabel("Exclude Region:"))
        exclude_time_layout.addWidget(self.exclude_start_input)
        exclude_time_layout.addWidget(self.exclude_end_input)
        left_layout.addLayout(exclude_time_layout)

        # Add "Add Exclude Region" button
        exclude_button = QPushButton("Add Exclude Region")
        exclude_button.clicked.connect(self.add_exclude_region)
        left_layout.addWidget(exclude_button)

        # Exclude region list and remove button
        self.exclude_list = QListWidget()
        left_layout.addWidget(QLabel("Excluded Regions:"))
        left_layout.addWidget(self.exclude_list)
        remove_region_button = QPushButton("Remove Selected Region")
        remove_region_button.clicked.connect(self.remove_exclude_region)
        left_layout.addWidget(remove_region_button)

        # Save button
        save_button = QPushButton("Save Cleaned Data")
        save_button.clicked.connect(self.save_data)
        left_layout.addWidget(save_button)

        # Right panel setup
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        # Add panels to main layout
        layout.addWidget(left_panel, stretch=1)
        layout.addWidget(right_panel, stretch=2)

    def remove_exclude_region(self):
        selected_items = self.exclude_list.selectedItems()
        for item in selected_items:
            row = self.exclude_list.row(item)
            self.exclude_list.takeItem(row)
            del self.exclude_regions[row]
        self.update_plot()

    def add_exclude_region(self):
        try:
            if self.exclude_start_input.text() and self.exclude_end_input.text():
                start = float(self.exclude_start_input.text())
                end = float(self.exclude_end_input.text())
            else:
                start, end = self.time_slider.value()

            if start > end:
                start, end = end, start

            self.exclude_regions.append((start, end))
            self.exclude_list.addItem(f"{start:.1f} - {end:.1f}")
            self.update_plot()
        except ValueError:
            pass

    def setup_time_controls(self, parent_layout):
        time_control_widget = QWidget()
        time_control_layout = QHBoxLayout(time_control_widget)

        self.min_time_input = QLineEdit()
        self.max_time_input = QLineEdit()
        self.min_time_input.setFixedWidth(70)
        self.max_time_input.setFixedWidth(70)

        self.time_slider = QRangeSlider()

        time_control_layout.addWidget(QLabel("Min:"))
        time_control_layout.addWidget(self.min_time_input)
        time_control_layout.addWidget(QLabel("Max:"))
        time_control_layout.addWidget(self.max_time_input)

        parent_layout.addWidget(QLabel("Time Range:"))
        parent_layout.addWidget(self.time_slider)
        parent_layout.addWidget(time_control_widget)

        # Connect signals
        self.min_time_input.returnPressed.connect(self.update_time_from_input)
        self.max_time_input.returnPressed.connect(self.update_time_from_input)
        self.time_slider.valueChanged.connect(self.update_time_display)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Data File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            try:
                # Read the CSV file with custom logic for multi-level columns
                self.df = pd.read_csv(file_path, delimiter=';', skiprows=1, header=[0, 1])
                self.df = self.df.dropna(axis=1, how='all')
                self.df.columns = [self._flatten_col(col) for col in self.df.columns]
                self.df = self.df.apply(pd.to_numeric, errors='coerce')
                if 'Timestamp' in self.df.columns:
                    self.df['Timestamp'] = self.df['Timestamp'] / 1000 / 60
                # Update the column list
                self.column_list.clear()
                self.column_list.addItems(self.df.columns)
                # If there's a timestamp column, set up the time range
                if 'Timestamp' in self.df.columns:
                    min_time = int(self.df['Timestamp'].min())
                    max_time = int(self.df['Timestamp'].max())
                    self.time_slider.setRange(min_time, max_time)
                    self.min_time_input.setText(f"{min_time:.1f}")
                    self.max_time_input.setText(f"{max_time:.1f}")
                # Update the plot
                self.update_plot()
                # Add the file to shared data manager
                self.shared_data_manager.add_cleaned_file(file_path)
            except Exception as e:
                print(f"Error loading file: {str(e)}")

    def _flatten_col(self, col):
        if isinstance(col, tuple):
            first = str(col[0]).strip() if pd.notna(col[0]) else ""
            second = str(col[1]).strip() if pd.notna(col[1]) else ""
            if second and second != first:
                return f"{first}({second})"
            return first
        return str(col).strip()

    def update_plot(self):
        if self.df is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Get selected columns
        selected_items = self.column_list.selectedItems()
        selected_columns = [item.text() for item in selected_items]

        # Filter by time range if possible
        df_filtered = self.df
        if 'Timestamp' in self.df.columns:
            try:
                time_range = self.time_slider.value()
                mask = (self.df['Timestamp'] >= time_range[0]) & (self.df['Timestamp'] <= time_range[1])
                df_filtered = self.df[mask]
            except Exception:
                pass
            # Apply exclude regions
            for region in self.exclude_regions:
                mask = ~((df_filtered['Timestamp'] >= region[0]) & (df_filtered['Timestamp'] <= region[1]))
                df_filtered = df_filtered[mask]

        # Plot selected columns
        for col in selected_columns:
            if col in df_filtered.columns:
                # Get scale text and factor
                scale_text = self.column_scales.get(col, '1x')
                scale_factor = {
                    '÷10': 0.1,
                    '÷100': 0.01,
                    '÷1000': 0.001
                }.get(scale_text, 1.0)
                offset = self.column_offsets.get(col, 0.0)
                y_data = (df_filtered[col] + offset) * scale_factor  # offset first, then scaling
                if 'Timestamp' in df_filtered.columns:
                    ax.plot(df_filtered['Timestamp'], y_data, label=f"{col} ({scale_text})")
                else:
                    ax.plot(df_filtered.index, y_data, label=f"{col} ({scale_text})")

        if 'Timestamp' in df_filtered.columns:
            ax.set_xlabel('Timestamp')
        else:
            ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        # ax.grid(True)  # Removed as per instructions
        ax.legend()
        # Increase font sizes
        ax.tick_params(axis='both', labelsize=12)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        self.canvas.draw()

    def update_time_from_input(self):
        try:
            min_time = float(self.min_time_input.text())
            max_time = float(self.max_time_input.text())

            if min_time > max_time:
                min_time, max_time = max_time, min_time

            slider_min = self.time_slider.min_slider.minimum()
            slider_max = self.time_slider.max_slider.maximum()

            # Ensure values are within slider range
            min_time = max(slider_min, min(slider_max, min_time))
            max_time = max(slider_min, min(slider_max, max_time))

            self.time_slider.setValue((int(min_time), int(max_time)))
            self.update_plot()
        except ValueError:
            # Restore previous values if input is invalid
            self.update_time_display(self.time_slider.value())

    def update_time_display(self, values):
        """Update the time input displays when the slider changes"""
        self.min_time_input.setText(f"{values[0]:.1f}")
        self.max_time_input.setText(f"{values[1]:.1f}")
        self.update_plot()

    def save_data(self):
        if self.df is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Cleaned Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            # Apply time range filter
            time_range = self.time_slider.value()
            mask = (self.df['Timestamp'] >= time_range[0]) & \
                   (self.df['Timestamp'] <= time_range[1])
            df_filtered = self.df[mask].copy()

            # Apply exclude regions
            for region in self.exclude_regions:
                mask = ~((df_filtered['Timestamp'] >= region[0]) & (df_filtered['Timestamp'] <= region[1]))
                df_filtered = df_filtered[mask]

            # Apply scaling and offset
            for column, scale in self.column_scales.items():
                if column in df_filtered.columns:
                    scale_factor = {
                        '÷10': 0.1,
                        '÷100': 0.01,
                        '÷1000': 0.001
                    }.get(scale, 1.0)
                    offset = self.column_offsets.get(column, 0.0)
                    # Apply offset first, then scaling
                    df_filtered[column] = (df_filtered[column] + offset) * scale_factor
                    # Update column name to reflect scaling
                    if scale != '1x':
                        new_column = f"{column}_{scale[1:]}"  # Remove the '÷' symbol
                        df_filtered.rename(columns={column: new_column}, inplace=True)

            # Save the filtered and scaled data
            df_filtered.to_csv(file_path, index=False)

            # Add the saved file to shared data manager
            self.shared_data_manager.add_cleaned_file(file_path)

    # Add the rest of the DataCleanerUI methods here...
    # (update_plot, etc.)


class PlotterUI(QMainWindow):
    def __init__(self, shared_data_manager):
        super().__init__()
        self.shared_data_manager = shared_data_manager
        self.setWindowTitle("IV Plotter")
        self.setGeometry(100, 100, 1400, 800)

        self.df = None
        # Base and plot timestamp unit combo boxes
        self.base_unit_combo = QComboBox()
        self.base_unit_combo.addItems(["ms", "s", "min", "h", "day"])
        self.base_unit_combo.setCurrentText("min")

        self.plot_unit_combo = QComboBox()
        self.plot_unit_combo.addItems(["ms", "s", "min", "h", "day"])
        self.plot_unit_combo.setCurrentText("min")

        self.setup_ui()

    def setup_ui(self):
        # Setup the main UI components
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left panel setup
        left_panel = QWidget()
        left_panel.setMinimumWidth(300)
        left_layout = QVBoxLayout(left_panel)

        # File selection
        self.setup_file_selection(left_layout)

        # Axis controls
        self.setup_axis_controls(left_layout)

        # Plot controls
        self.setup_plot_controls(left_layout)

        # Right panel with plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.setup_plot_area(right_layout)

        # Add panels to main layout, left panel in scroll area
        from PyQt5.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(left_panel)
        layout.addWidget(scroll_area, stretch=1)
        layout.addWidget(right_panel, stretch=2)

    def setup_file_selection(self, layout):
        self.load_button = QPushButton("Load Cleaned File")
        self.load_button.clicked.connect(self.load_file)
        layout.addWidget(self.load_button)
        # Add button to open arbitrary CSV file
        self.open_file_button = QPushButton("Open CSV (Any File)")
        self.open_file_button.clicked.connect(self.open_external_file)
        layout.addWidget(self.open_file_button)
        self.file_list = QListWidget()
        layout.addWidget(QLabel("Available Files:"))
        layout.addWidget(self.file_list)

    def setup_axis_controls(self, layout):
        self.x_axis_combo = QComboBox()
        self.y1_axis_combo = QComboBox()
        self.y2_axis_combo = QComboBox()
        self.y2_axis_combo.setEnabled(False)
        self.y3_axis_combo = QComboBox()
        self.y3_axis_combo.setEnabled(False)
        self.y4_axis_combo = QComboBox()
        self.y4_axis_combo.setEnabled(False)

        self.x_label_input = QLineEdit()
        self.x_label_input.setPlaceholderText("X-axis Label")

        self.y1_label_input = QLineEdit()
        self.y1_label_input.setPlaceholderText("Left Y-axis Label")

        self.y2_label_input = QLineEdit()
        self.y2_label_input.setPlaceholderText("Right Y-axis Label")
        self.y2_label_input.setEnabled(False)

        self.y3_label_input = QLineEdit()
        self.y3_label_input.setPlaceholderText("Y3 Axis Label")
        self.y3_label_input.setEnabled(False)

        self.y4_label_input = QLineEdit()
        self.y4_label_input.setPlaceholderText("Y4 Axis Label")
        self.y4_label_input.setEnabled(False)

        self.enable_y2_checkbox = QCheckBox("Enable Second Y-axis")
        self.enable_y2_checkbox.stateChanged.connect(self.toggle_second_y_axis)
        self.enable_y3_checkbox = QCheckBox("Enable Third Y-axis")
        self.enable_y3_checkbox.stateChanged.connect(self.toggle_third_y_axis)
        self.enable_y4_checkbox = QCheckBox("Enable Fourth Y-axis")
        self.enable_y4_checkbox.stateChanged.connect(self.toggle_fourth_y_axis)

        # --- Add Y-axis min/max QLineEdit widgets ---
        self.y1_min_input = QLineEdit()
        self.y1_max_input = QLineEdit()
        self.y2_min_input = QLineEdit()
        self.y2_max_input = QLineEdit()
        self.y3_min_input = QLineEdit()
        self.y3_max_input = QLineEdit()
        self.y4_min_input = QLineEdit()
        self.y4_max_input = QLineEdit()

        # --- Resistance plotting controls ---
        self.enable_resistance_checkbox = QCheckBox("Calculate and Plot Resistance")
        self.enable_resistance_checkbox.stateChanged.connect(self.toggle_resistance_controls)
        layout.addWidget(self.enable_resistance_checkbox)

        self.resistance_widget = QWidget()
        self.resistance_layout = QVBoxLayout(self.resistance_widget)

        self.voltage_combo = QComboBox()
        self.current_combo = QComboBox()
        self.voltage_scale_combo = QComboBox()
        # Update voltage scale combo box items for resistance calculation
        self.voltage_scale_combo.clear()
        self.voltage_scale_combo.addItems(['mV', 'mV×100', 'mV×1000'])

        self.resistance_layout.addWidget(QLabel("Voltage Column:"))
        self.resistance_layout.addWidget(self.voltage_combo)
        self.resistance_layout.addWidget(QLabel("Current Column:"))
        self.resistance_layout.addWidget(self.current_combo)
        self.resistance_layout.addWidget(QLabel("Voltage Scale:"))
        self.resistance_layout.addWidget(self.voltage_scale_combo)

        layout.addWidget(self.resistance_widget)
        self.resistance_widget.setVisible(False)

        self.x_min_input = QLineEdit()
        self.x_max_input = QLineEdit()
        self.x_min_input.setFixedWidth(70)
        self.x_max_input.setFixedWidth(70)
        self.x_min_input.setPlaceholderText("X min")
        self.x_max_input.setPlaceholderText("X max")

        # Add absolute value checkboxes for Y1-Y4
        self.y1_abs_checkbox = QCheckBox("Use |Y1|")
        self.y2_abs_checkbox = QCheckBox("Use |Y2|")
        self.y2_abs_checkbox.setEnabled(False)
        self.y3_abs_checkbox = QCheckBox("Use |Y3|")
        self.y3_abs_checkbox.setEnabled(False)
        self.y4_abs_checkbox = QCheckBox("Use |Y4|")
        self.y4_abs_checkbox.setEnabled(False)

        # Add derivative checkboxes for Y1-Y4
        self.y1_deriv_checkbox = QCheckBox("Use d/dx Y1")
        self.y2_deriv_checkbox = QCheckBox("Use d/dx Y2")
        self.y2_deriv_checkbox.setEnabled(False)
        self.y3_deriv_checkbox = QCheckBox("Use d/dx Y3")
        self.y3_deriv_checkbox.setEnabled(False)
        self.y4_deriv_checkbox = QCheckBox("Use d/dx Y4")
        self.y4_deriv_checkbox.setEnabled(False)

        layout.addWidget(QLabel("X Axis:"))
        layout.addWidget(self.x_axis_combo)
        layout.addWidget(self.x_label_input)

        layout.addWidget(QLabel("Y1 Axis:"))
        layout.addWidget(self.y1_axis_combo)
        layout.addWidget(self.y1_label_input)
        layout.addWidget(self.y1_abs_checkbox)
        layout.addWidget(self.y1_deriv_checkbox)
        # Add Y1 min/max inputs in a horizontal layout
        y1_range_layout = QHBoxLayout()
        y1_range_layout.addWidget(QLabel("Y1 Min:"))
        y1_range_layout.addWidget(self.y1_min_input)
        y1_range_layout.addWidget(QLabel("Max:"))
        y1_range_layout.addWidget(self.y1_max_input)
        layout.addLayout(y1_range_layout)
        layout.addWidget(self.enable_y2_checkbox)

        layout.addWidget(QLabel("Y2 Axis:"))
        layout.addWidget(self.y2_axis_combo)
        layout.addWidget(self.y2_label_input)
        layout.addWidget(self.y2_abs_checkbox)
        layout.addWidget(self.y2_deriv_checkbox)
        # Add Y2 min/max inputs in a horizontal layout
        y2_range_layout = QHBoxLayout()
        y2_range_layout.addWidget(QLabel("Y2 Min:"))
        y2_range_layout.addWidget(self.y2_min_input)
        y2_range_layout.addWidget(QLabel("Max:"))
        y2_range_layout.addWidget(self.y2_max_input)
        layout.addLayout(y2_range_layout)
        layout.addWidget(self.enable_y3_checkbox)

        layout.addWidget(QLabel("Y3 Axis:"))
        layout.addWidget(self.y3_axis_combo)
        layout.addWidget(self.y3_label_input)
        layout.addWidget(self.y3_abs_checkbox)
        layout.addWidget(self.y3_deriv_checkbox)
        # Add Y3 min/max inputs in a horizontal layout
        y3_range_layout = QHBoxLayout()
        y3_range_layout.addWidget(QLabel("Y3 Min:"))
        y3_range_layout.addWidget(self.y3_min_input)
        y3_range_layout.addWidget(QLabel("Max:"))
        y3_range_layout.addWidget(self.y3_max_input)
        layout.addLayout(y3_range_layout)
        layout.addWidget(self.enable_y4_checkbox)

        layout.addWidget(QLabel("Y4 Axis:"))
        layout.addWidget(self.y4_axis_combo)
        layout.addWidget(self.y4_label_input)
        layout.addWidget(self.y4_abs_checkbox)
        layout.addWidget(self.y4_deriv_checkbox)
        # Add Y4 min/max inputs in a horizontal layout
        y4_range_layout = QHBoxLayout()
        y4_range_layout.addWidget(QLabel("Y4 Min:"))
        y4_range_layout.addWidget(self.y4_min_input)
        y4_range_layout.addWidget(QLabel("Max:"))
        y4_range_layout.addWidget(self.y4_max_input)
        layout.addLayout(y4_range_layout)
        layout.addWidget(QLabel("X range:"))
        layout.addWidget(self.x_min_input)
        layout.addWidget(self.x_max_input)
        # Add legend toggle checkbox
        self.show_legend_checkbox = QCheckBox("Show Legend")
        self.show_legend_checkbox.setChecked(True)
        layout.addWidget(self.show_legend_checkbox)

        # Add custom legend entry inputs below the legend toggle checkbox
        self.y1_legend_input = QLineEdit()
        self.y1_legend_input.setPlaceholderText("Y1 Legend")

        self.y2_legend_input = QLineEdit()
        self.y2_legend_input.setPlaceholderText("Y2 Legend")
        self.y2_legend_input.setEnabled(False)

        self.y3_legend_input = QLineEdit()
        self.y3_legend_input.setPlaceholderText("Y3 Legend")
        self.y3_legend_input.setEnabled(False)

        self.y4_legend_input = QLineEdit()
        self.y4_legend_input.setPlaceholderText("Y4 Legend")
        self.y4_legend_input.setEnabled(False)

        layout.addWidget(self.y1_legend_input)
        layout.addWidget(self.y2_legend_input)
        layout.addWidget(self.y3_legend_input)
        layout.addWidget(self.y4_legend_input)

        # Add QLabel to display R at 100A under the options panel
        self.r100a_label = QLabel("")
        layout.addWidget(self.r100a_label)

    def toggle_third_y_axis(self, state):
        enabled = state == Qt.Checked
        self.y3_axis_combo.setEnabled(enabled)
        self.y3_label_input.setEnabled(enabled)
        self.y3_abs_checkbox.setEnabled(enabled)
        self.y3_legend_input.setEnabled(enabled)
        self.y3_deriv_checkbox.setEnabled(enabled)

    def toggle_fourth_y_axis(self, state):
        enabled = state == Qt.Checked
        self.y4_axis_combo.setEnabled(enabled)
        self.y4_label_input.setEnabled(enabled)
        self.y4_abs_checkbox.setEnabled(enabled)
        self.y4_legend_input.setEnabled(enabled)
        self.y4_deriv_checkbox.setEnabled(enabled)

    def toggle_resistance_controls(self, state):
        enabled = state == Qt.Checked
        self.resistance_widget.setVisible(enabled)
        # Disable all extra Y axes if resistance is enabled
        if enabled:
            self.y2_axis_combo.setEnabled(False)
            self.y2_label_input.setEnabled(False)
            self.y2_abs_checkbox.setEnabled(False)
            self.y2_legend_input.setEnabled(False)
            self.y2_deriv_checkbox.setEnabled(False)
            self.enable_y2_checkbox.setEnabled(False)
            self.y3_axis_combo.setEnabled(False)
            self.y3_label_input.setEnabled(False)
            self.y3_abs_checkbox.setEnabled(False)
            self.y3_legend_input.setEnabled(False)
            self.y3_deriv_checkbox.setEnabled(False)
            self.enable_y3_checkbox.setEnabled(False)
            self.y4_axis_combo.setEnabled(False)
            self.y4_label_input.setEnabled(False)
            self.y4_abs_checkbox.setEnabled(False)
            self.y4_legend_input.setEnabled(False)
            self.y4_deriv_checkbox.setEnabled(False)
            self.enable_y4_checkbox.setEnabled(False)
        else:
            self.enable_y2_checkbox.setEnabled(True)
            self.enable_y3_checkbox.setEnabled(True)
            self.enable_y4_checkbox.setEnabled(True)
            self.toggle_second_y_axis(self.enable_y2_checkbox.checkState())
            self.toggle_third_y_axis(self.enable_y3_checkbox.checkState())
            self.toggle_fourth_y_axis(self.enable_y4_checkbox.checkState())

    def toggle_second_y_axis(self, state):
        enabled = state == Qt.Checked
        self.y2_axis_combo.setEnabled(enabled)
        self.y2_label_input.setEnabled(enabled)
        self.y2_abs_checkbox.setEnabled(enabled)
        self.y2_legend_input.setEnabled(enabled)
        self.y2_deriv_checkbox.setEnabled(enabled)

    def setup_plot_controls(self, layout):
        # Timestamp unit controls
        layout.addWidget(QLabel("Base Timestamp Unit:"))
        layout.addWidget(self.base_unit_combo)
        layout.addWidget(QLabel("Plot Timestamp Unit:"))
        layout.addWidget(self.plot_unit_combo)
        plot_button = QPushButton("Plot")
        plot_button.clicked.connect(self.plot_selected)
        layout.addWidget(plot_button)

    def setup_plot_area(self, layout):
        self.figure = Figure(figsize=(8, 6), dpi=100)  # 800x600 pixels, 4:3 aspect ratio
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def update_file_list(self):
        self.file_list.clear()
        for f in self.shared_data_manager.get_cleaned_files():
            self.file_list.addItem(f)

    def load_file(self):
        if self.file_list.currentItem():
            file_path = self.file_list.currentItem().text()
            try:
                self.df = pd.read_csv(file_path)
                self.x_axis_combo.clear()
                self.y1_axis_combo.clear()
                self.y2_axis_combo.clear()
                self.y3_axis_combo.clear()
                self.y4_axis_combo.clear()
                self.x_axis_combo.addItems(self.df.columns)
                self.y1_axis_combo.addItems(self.df.columns)
                self.y2_axis_combo.addItems(self.df.columns)
                self.y3_axis_combo.addItems(self.df.columns)
                self.y4_axis_combo.addItems(self.df.columns)
                # Update resistance combos
                self.voltage_combo.clear()
                self.current_combo.clear()
                self.voltage_combo.addItems(self.df.columns)
                self.current_combo.addItems(self.df.columns)
            except Exception as e:
                print(f"Failed to load file: {e}")

    def plot_selected(self):
        if self.df is not None:
            if self.enable_resistance_checkbox.isChecked():
                self.y2_axis_combo.setEnabled(False)
                self.y2_label_input.setEnabled(False)
                self.y2_abs_checkbox.setEnabled(False)
                self.y2_legend_input.setEnabled(False)
                self.y2_deriv_checkbox.setEnabled(False)
                self.enable_y2_checkbox.setEnabled(False)
                self.y3_axis_combo.setEnabled(False)
                self.y3_label_input.setEnabled(False)
                self.y3_abs_checkbox.setEnabled(False)
                self.y3_legend_input.setEnabled(False)
                self.y3_deriv_checkbox.setEnabled(False)
                self.enable_y3_checkbox.setEnabled(False)
                self.y4_axis_combo.setEnabled(False)
                self.y4_label_input.setEnabled(False)
                self.y4_abs_checkbox.setEnabled(False)
                self.y4_legend_input.setEnabled(False)
                self.y4_deriv_checkbox.setEnabled(False)
                self.enable_y4_checkbox.setEnabled(False)
            else:
                self.enable_y2_checkbox.setEnabled(True)
                self.enable_y3_checkbox.setEnabled(True)
                self.enable_y4_checkbox.setEnabled(True)
                self.toggle_second_y_axis(self.enable_y2_checkbox.checkState())
                self.toggle_third_y_axis(self.enable_y3_checkbox.checkState())
                self.toggle_fourth_y_axis(self.enable_y4_checkbox.checkState())

            x_col = self.x_axis_combo.currentText()
            y1_col = self.y1_axis_combo.currentText()
            y2_col = self.y2_axis_combo.currentText() if self.y2_axis_combo.isEnabled() else None
            y3_col = self.y3_axis_combo.currentText() if self.y3_axis_combo.isEnabled() else None
            y4_col = self.y4_axis_combo.currentText() if self.y4_axis_combo.isEnabled() else None
            x_min = float(self.x_min_input.text()) if self.x_min_input.text() else None
            x_max = float(self.x_max_input.text()) if self.x_max_input.text() else None

            # --- Determine how many extra Y axes are in use (Y2, Y3, Y4) ---
            y_axes_used = sum(bool(c) for c in [y2_col, y3_col, y4_col])

            # --- Fixed canvas and figure size for consistent aspect ratio ---
            self.canvas.setFixedSize(800, 600)
            self.figure.set_size_inches(8, 6)

            self.figure.clear()
            self.figure.set_facecolor('white')
            base_fontsize = max(int(self.figure.get_size_inches()[0] * self.figure.dpi / 100), 10)
            ax1 = self.figure.add_subplot(111)
            ax2 = None
            ax3 = None
            ax4 = None
            ax1.set_facecolor('#f9f9f9')

            # Timestamp scaling logic for X axis if first column is timestamp
            df = self.df
            if x_col == df.columns[0]:
                # Use base/plot unit scaling
                base_unit = self.base_unit_combo.currentText()
                plot_unit = self.plot_unit_combo.currentText()
                unit_to_min = {"ms": 1/60000, "s": 1/60, "min": 1, "h": 60, "day": 1440}
                timestamps = df.iloc[:, 0].values
                x_data = timestamps * unit_to_min[base_unit] / unit_to_min[plot_unit]
                x_data = pd.Series(x_data, index=df.index)
            else:
                x_data = df[x_col]
            n_points = len(x_data)
            marker_size = min(max(3, 300 / n_points), 10)
            line_width = 2.0 if n_points < 200 else 1.5

            mask = pd.Series(True, index=df.index)
            if x_min is not None:
                mask &= x_data >= x_min
            if x_max is not None:
                mask &= x_data <= x_max

            x_data = x_data[mask]

            # Plot Y1
            if y1_col:
                y1_data = self.df[y1_col][mask]
                if self.y1_abs_checkbox.isChecked():
                    y1_data = y1_data.abs()
                if self.y1_deriv_checkbox.isChecked():
                    y1_data = y1_data.diff() / x_data.diff()
                ax1.plot(
                    x_data,
                    y1_data,
                    label=self.y1_legend_input.text() or y1_col,
                    color='tab:blue',
                    linewidth=line_width,
                    alpha=0.9,
                    marker='o',
                    markersize=marker_size,
                )
                ax1.set_ylabel(self.y1_label_input.text() or y1_col)
                # --- Color styling for Y1 axis ---
                ax1.spines["left"].set_color('tab:blue')
                ax1.tick_params(axis='y', colors='tab:blue', labelsize=base_fontsize)
                ax1.yaxis.label.set_color('tab:blue')
                ax1.yaxis.label.set_size(base_fontsize + 2)
                ax1.tick_params(axis='x', labelsize=base_fontsize)
                ax1.xaxis.label.set_size(base_fontsize + 2)
                # Apply Y1 min/max if set
                try:
                    y1_min = float(self.y1_min_input.text())
                    y1_max = float(self.y1_max_input.text())
                    ax1.set_ylim(y1_min, y1_max)
                except ValueError:
                    pass

            legend_axes = [ax1]
            axes_labels = [ax1.get_ylabel()]
            if self.enable_resistance_checkbox.isChecked():
                v_col = self.voltage_combo.currentText()
                i_col = self.current_combo.currentText()
                scale_text = self.voltage_scale_combo.currentText()
                scale_factor = {
                    'mV': 1.0e-3,
                    'mV×100': 1.0e-3 / 100,
                    'mV×1000': 1.0e-3 / 1000
                }.get(scale_text, 1.0e-3)

                voltage = self.df[v_col][mask] * scale_factor
                current = self.df[i_col][mask]
                resistance = (voltage / current) * 1e6  # Convert to microohms
                ax2 = ax1.twinx()
                ax2.set_facecolor('#f9f9f9')
                ax2.plot(
                    x_data,
                    resistance,
                    label="Resistance",
                    color='tab:red',
                    linewidth=line_width,
                    alpha=0.9,
                    marker='o',
                    markersize=marker_size,
                )
                ax2.set_ylabel("Resistance (µΩ)")
                ax2.ticklabel_format(style='plain', axis='y')
                # --- Color styling for resistance axis (use tab:orange for Y2) ---
                ax2.spines["right"].set_color('tab:orange')
                ax2.tick_params(axis='y', colors='tab:orange', labelsize=base_fontsize)
                ax2.yaxis.label.set_color('tab:orange')
                ax2.yaxis.label.set_size(base_fontsize + 2)
                # Default Y2 label placement for resistance (vertical)
                ax2.yaxis.label.set_rotation(90)
                ax2.yaxis.labelpad = 15
                ax2.yaxis.label.set_position((1.12, 0.5))
                ax2.tick_params(axis='x', labelsize=base_fontsize)
                ax2.xaxis.label.set_size(base_fontsize + 2)
                # Apply Y2 min/max if set
                try:
                    y2_min = float(self.y2_min_input.text())
                    y2_max = float(self.y2_max_input.text())
                    ax2.set_ylim(y2_min, y2_max)
                except ValueError:
                    pass
                legend_axes.append(ax2)
                axes_labels.append(ax2.get_ylabel())
                # Display resistance at 100 A in UI
                if not current.empty:
                    closest = current.sub(100).abs().idxmin()
                    r_at_100a = resistance.loc[closest]
                    self.r100a_label.setText(f"Resistance at 100A: {r_at_100a:.2f} µΩ")
                else:
                    self.r100a_label.setText("")
            else:
                self.r100a_label.setText("")
                # Plot Y2 if enabled
                if y2_col:
                    ax2 = ax1.twinx()
                    ax2.set_facecolor('#f9f9f9')
                    y2_data = self.df[y2_col][mask]
                    if self.y2_abs_checkbox.isChecked():
                        y2_data = y2_data.abs()
                    if self.y2_deriv_checkbox.isChecked():
                        y2_data = y2_data.diff() / x_data.diff()
                    ax2.plot(
                        x_data,
                        y2_data,
                        label=self.y2_legend_input.text() or y2_col,
                        color='tab:orange',
                        linewidth=line_width,
                        alpha=0.9,
                        marker='o',
                        markersize=marker_size,
                    )
                    ax2.set_ylabel(self.y2_label_input.text() or y2_col)
                    # --- Color styling for Y2 axis ---
                    ax2.spines["right"].set_color('tab:orange')
                    ax2.tick_params(axis='y', colors='tab:orange', labelsize=base_fontsize)
                    ax2.yaxis.label.set_color('tab:orange')
                    ax2.yaxis.label.set_size(base_fontsize + 2)
                    # --- Y2 label positioning logic based on number of axes ---
                    if y_axes_used >= 3:
                        ax2.yaxis.label.set_rotation(45)
                        ax2.yaxis.labelpad = 10
                        ax2.yaxis.label.set_verticalalignment('bottom')
                        ax2.yaxis.label.set_horizontalalignment('left')
                        ax2.yaxis.label.set_position((1.12, 1.05))
                    else:
                        ax2.yaxis.label.set_rotation(90)
                        ax2.yaxis.labelpad = 15
                        ax2.yaxis.label.set_position((1.12, 0.5))
                    ax2.tick_params(axis='x', labelsize=base_fontsize)
                    ax2.xaxis.label.set_size(base_fontsize + 2)
                    # Apply Y2 min/max if set
                    try:
                        y2_min = float(self.y2_min_input.text())
                        y2_max = float(self.y2_max_input.text())
                        ax2.set_ylim(y2_min, y2_max)
                    except ValueError:
                        pass
                    legend_axes.append(ax2)
                    axes_labels.append(ax2.get_ylabel())
                # Plot Y3 if enabled
                if y3_col:
                    # Y3 always uses twinx and positions outward
                    ax3 = ax1.twinx()
                    ax3.set_facecolor('#f9f9f9')
                    ax3.spines["right"].set_position(("outward", 60))
                    y3_data = self.df[y3_col][mask]
                    if self.y3_abs_checkbox.isChecked():
                        y3_data = y3_data.abs()
                    if self.y3_deriv_checkbox.isChecked():
                        y3_data = y3_data.diff() / x_data.diff()
                    ax3.plot(
                        x_data,
                        y3_data,
                        label=self.y3_legend_input.text() or y3_col,
                        color='tab:green',
                        linewidth=line_width,
                        alpha=0.9,
                        marker='o',
                        markersize=marker_size,
                    )
                    ax3.set_ylabel(self.y3_label_input.text() or y3_col)
                    # --- Color styling for Y3 axis ---
                    ax3.spines["right"].set_color('tab:green')
                    ax3.tick_params(axis='y', colors='tab:green', labelsize=base_fontsize)
                    ax3.yaxis.label.set_color('tab:green')
                    ax3.yaxis.label.set_size(base_fontsize + 2)
                    # --- Y3 label positioning logic based on number of axes ---
                    ax3.yaxis.set_label_position("right")
                    ax3.yaxis.set_offset_position("right")
                    if y_axes_used >= 3:
                        ax3.yaxis.label.set_rotation(45)
                        ax3.yaxis.labelpad = 10
                        ax3.yaxis.label.set_verticalalignment('bottom')
                        ax3.yaxis.label.set_horizontalalignment('left')
                        ax3.yaxis.label.set_position((1.22, 1.05))
                    else:
                        ax3.yaxis.label.set_rotation(45)
                        ax3.yaxis.labelpad = 10
                        ax3.yaxis.label.set_verticalalignment('bottom')
                        ax3.yaxis.label.set_horizontalalignment('left')
                        ax3.yaxis.label.set_position((1.22, 1.05))
                    # Apply Y3 min/max if set
                    try:
                        y3_min = float(self.y3_min_input.text())
                        y3_max = float(self.y3_max_input.text())
                        ax3.set_ylim(y3_min, y3_max)
                    except ValueError:
                        pass
                    legend_axes.append(ax3)
                    axes_labels.append(ax3.get_ylabel())
                # Plot Y4 if enabled
                if y4_col:
                    # Y4 always uses twinx and positions further outward
                    ax4 = ax1.twinx()
                    ax4.set_facecolor('#f9f9f9')
                    ax4.spines["right"].set_position(("outward", 120))
                    y4_data = self.df[y4_col][mask]
                    if self.y4_abs_checkbox.isChecked():
                        y4_data = y4_data.abs()
                    if self.y4_deriv_checkbox.isChecked():
                        y4_data = y4_data.diff() / x_data.diff()
                    ax4.plot(
                        x_data,
                        y4_data,
                        label=self.y4_legend_input.text() or y4_col,
                        color='tab:purple',
                        linewidth=line_width,
                        alpha=0.9,
                        marker='o',
                        markersize=marker_size,
                    )
                    ax4.set_ylabel(self.y4_label_input.text() or y4_col)
                    # --- Color styling for Y4 axis ---
                    ax4.spines["right"].set_color('tab:purple')
                    ax4.tick_params(axis='y', colors='tab:purple', labelsize=base_fontsize)
                    ax4.yaxis.label.set_color('tab:purple')
                    ax4.yaxis.label.set_size(base_fontsize + 2)
                    # --- Y4 label positioning logic based on number of axes ---
                    ax4.yaxis.set_label_position("right")
                    ax4.yaxis.set_offset_position("right")
                    if y_axes_used >= 3:
                        ax4.yaxis.label.set_rotation(45)
                        ax4.yaxis.labelpad = 10
                        ax4.yaxis.label.set_verticalalignment('bottom')
                        ax4.yaxis.label.set_horizontalalignment('left')
                        ax4.yaxis.label.set_position((1.33, 1.05))
                    else:
                        ax4.yaxis.label.set_rotation(-45)
                        ax4.yaxis.labelpad = 10
                        ax4.yaxis.label.set_verticalalignment('top')
                        ax4.yaxis.label.set_horizontalalignment('left')
                        ax4.yaxis.label.set_position((1.33, -0.1))
                    # Apply Y4 min/max if set
                    try:
                        y4_min = float(self.y4_min_input.text())
                        y4_max = float(self.y4_max_input.text())
                        ax4.set_ylim(y4_min, y4_max)
                    except ValueError:
                        pass
                    legend_axes.append(ax4)
                    axes_labels.append(ax4.get_ylabel())

            # Set X-axis label using input
            ax1.set_xlabel(self.x_label_input.text() or x_col)
            ax1.tick_params(axis='x', labelsize=base_fontsize)
            ax1.xaxis.label.set_size(base_fontsize + 2)

            # Adjust subplot for multiple y-axes
            self.figure.subplots_adjust(left=0.15, right=0.75, bottom=0.15, top=0.9)

            # Add combined legend at best location if enabled
            if self.show_legend_checkbox.isChecked():
                lines, labels = [], []
                for ax in legend_axes:
                    l, lb = ax.get_legend_handles_labels()
                    lines += l
                    labels += lb
                ax1.legend(lines, labels, loc='best', fontsize=base_fontsize)

            # Enforce tight layout for clean export
            self.figure.tight_layout()
            self.canvas.draw()

    def open_external_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.x_axis_combo.clear()
                self.y1_axis_combo.clear()
                self.y2_axis_combo.clear()
                self.y3_axis_combo.clear()
                self.y4_axis_combo.clear()
                self.x_axis_combo.addItems(self.df.columns)
                self.y1_axis_combo.addItems(self.df.columns)
                self.y2_axis_combo.addItems(self.df.columns)
                self.y3_axis_combo.addItems(self.df.columns)
                self.y4_axis_combo.addItems(self.df.columns)
                # Update resistance combos
                self.voltage_combo.clear()
                self.current_combo.clear()
                self.voltage_combo.addItems(self.df.columns)
                self.current_combo.addItems(self.df.columns)
            except Exception as e:
                print(f"Failed to open file: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.loading_label = QLabel(self)
        self.loading_movie = QMovie("spinning_magnet.gif")
        self.loading_label.setMovie(self.loading_movie)
        self.loading_label.setVisible(True)
        self.loading_movie.start()

        self.setWindowTitle("Magnet Data Analysis Tool")
        self.setGeometry(100, 100, 1400, 800)

        self.shared_data_manager = SharedDataManager()
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Mode selection buttons
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)

        cleaner_button = QPushButton("🧹 Data Cleaner")
        plotter_button = QPushButton("📈 Plotter")
        combiner_button = QPushButton("🔗 Combine Files")
        live_button = QPushButton("⚡ Live Viewer")

        button_layout.addWidget(cleaner_button)
        button_layout.addWidget(plotter_button)
        button_layout.addWidget(combiner_button)
        button_layout.addWidget(live_button)
        layout.addWidget(button_widget)

        # Stacked widget for interfaces
        self.stacked_widget = QStackedWidget()
        self.cleaner = DataCleanerUI(self.shared_data_manager)
        self.plotter = PlotterUI(self.shared_data_manager)
        self.combiner = FileCombinerUI(self.shared_data_manager)
        self.live_viewer = LiveViewerUI()

        self.stacked_widget.addWidget(self.cleaner)
        self.stacked_widget.addWidget(self.plotter)
        self.stacked_widget.addWidget(self.combiner)
        self.stacked_widget.addWidget(self.live_viewer)

        layout.addWidget(self.stacked_widget)

        # Connect buttons
        cleaner_button.clicked.connect(self.switch_to_cleaner)
        plotter_button.clicked.connect(self.switch_to_plotter)
        combiner_button.clicked.connect(self.switch_to_combiner)
        live_button.clicked.connect(self.switch_to_live_viewer)

    def switch_to_live_viewer(self):
        self.stacked_widget.setCurrentIndex(3)

    def switch_to_cleaner(self):
        self.stacked_widget.setCurrentIndex(0)

    def switch_to_plotter(self):
        self.stacked_widget.setCurrentIndex(1)
        self.plotter.update_file_list()

    def switch_to_combiner(self):
        self.stacked_widget.setCurrentIndex(2)
        self.combiner.update_file_list()


# --- File Combiner UI ---
class FileCombinerUI(QMainWindow):
    def __init__(self, shared_data_manager):
        super().__init__()
        self.shared_data_manager = shared_data_manager
        self.setWindowTitle("File Combiner")
        self.setGeometry(100, 100, 1400, 800)
        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.MultiSelection)
        # Remove drag-and-drop reordering; use selection order instead
        # self.file_list.setDragDropMode(QListWidget.InternalMove)
        # self.file_list.setDefaultDropAction(Qt.MoveAction)
        layout.addWidget(QLabel("Click cleaned files in the desired order to combine.\n"
                                "The order will be shown by numbers."))
        layout.addWidget(self.file_list)

        # Remove drag-and-drop reordering signal
        # self.file_list.model().rowsMoved.connect(lambda *args: self.refresh_numbered_file_list())

        combine_button = QPushButton("Combine Selected Files")
        combine_button.clicked.connect(self.combine_files)
        layout.addWidget(combine_button)

        self.update_file_list()

    def update_file_list(self):
        self.file_list.clear()
        for file in self.shared_data_manager.get_cleaned_files():
            self.file_list.addItem(file)
        self.refresh_numbered_file_list()

    def combine_files(self):
        # Always refresh numbered file list to reflect selection order
        self.refresh_numbered_file_list()
        # Get selected paths in the order user clicked them
        selected_paths = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.isSelected():
                selected_paths.append(item.text().split(": ", 1)[-1])

        if not selected_paths:
            return

        dfs = []
        offset = 0
        for file_path in selected_paths:
            df = pd.read_csv(file_path)
            if 'Timestamp' in df.columns:
                df['Timestamp'] = df['Timestamp'] + offset
                offset = df['Timestamp'].iloc[-1] + 0.01  # ensure next starts slightly after
            dfs.append(df)

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Combined Data", "", "CSV Files (*.csv);;All Files (*)")
            if save_path:
                combined_df.to_csv(save_path, index=False)
                self.shared_data_manager.add_cleaned_file(save_path)
            # Update each list item text after combining to refresh ordering
            self.update_file_list()
            self.refresh_numbered_file_list()

    def refresh_numbered_file_list(self):
        # Number only selected items in the order they were selected
        selected = [self.file_list.item(i) for i in range(self.file_list.count()) if
                    self.file_list.item(i).isSelected()]
        for i, item in enumerate(selected):
            text = item.text().split(": ", 1)[-1]
            item.setText(f"{i + 1}: {text}")


import threading
import time


class LiveViewerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Viewer")
        self.setGeometry(100, 100, 1400, 800)

        self.df = pd.DataFrame()
        self.running = False
        self.file_path = ""
        self.update_interval = 1000  # milliseconds

        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        control_layout = QHBoxLayout()

        self.load_button = QPushButton("Select File")
        self.load_button.clicked.connect(self.select_file)
        control_layout.addWidget(self.load_button)

        self.interval_input = QLineEdit("1000")
        self.interval_input.setFixedWidth(80)
        control_layout.addWidget(QLabel("Update ms:"))
        control_layout.addWidget(self.interval_input)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_plotting)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_plotting)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        layout.addLayout(control_layout)

        # --- Replace QListWidget with three QComboBox for X, Y1, Y2 ---
        selector_layout = QHBoxLayout()
        self.x_combo = QComboBox()
        self.y1_combo = QComboBox()
        self.y2_combo = QComboBox()
        selector_layout.addWidget(QLabel("X:"))
        selector_layout.addWidget(self.x_combo)
        selector_layout.addWidget(QLabel("Y1:"))
        selector_layout.addWidget(self.y1_combo)
        selector_layout.addWidget(QLabel("Y2:"))
        selector_layout.addWidget(self.y2_combo)
        layout.addLayout(selector_layout)

        self.figure = Figure(figsize=(6, 4), dpi=300)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "All Files (*)")
        if file_path:
            self.file_path = file_path
            try:
                # Use the same logic as DataCleanerUI.load_file for multi-level headers
                self.df = pd.read_csv(self.file_path, delimiter=';', skiprows=1, header=[0, 1], low_memory=False)
                self.df = self.df.dropna(axis=1, how='all')
                self.df.columns = [self._flatten_col(col) for col in self.df.columns]
                self.df = self.df.apply(pd.to_numeric, errors='coerce')
                # Populate combo boxes for X, Y1, Y2
                self.x_combo.clear()
                self.y1_combo.clear()
                self.y2_combo.clear()
                self.x_combo.addItems(self.df.columns)
                self.y1_combo.addItems(self.df.columns)
                self.y2_combo.addItems(self.df.columns)
                # Optionally set default selection for X to 'Timestamp'
                if 'Timestamp' in self.df.columns:
                    self.x_combo.setCurrentText('Timestamp')
            except Exception as e:
                print(f"Failed to load file: {e}")

    def _flatten_col(self, col):
        if isinstance(col, tuple):
            first = str(col[0]).strip() if pd.notna(col[0]) else ""
            second = str(col[1]).strip() if pd.notna(col[1]) else ""
            if second and second != first:
                return f"{first}({second})"
            return first
        return str(col).strip()

    def start_plotting(self):
        if not self.file_path:
            return
        try:
            self.update_interval = int(self.interval_input.text())
        except ValueError:
            self.update_interval = 1000

        # Fallback: Load the file once when live plotting starts
        try:
            self.df = pd.read_csv(self.file_path)
            self.plot_live_data()
        except Exception as e:
            print(f"Initial load error: {e}")

        self.running = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.thread = threading.Thread(target=self.live_plot_loop, daemon=True)
        self.thread.start()

    def stop_plotting(self):
        self.running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def live_plot_loop(self):
        while self.running:
            try:
                self.df = pd.read_csv(self.file_path)
                self.plot_live_data()
            except Exception as e:
                print(f"Live plotting error: {e}")
            time.sleep(self.update_interval / 1000.0)

    def plot_live_data(self):
        # Always attempt to plot; only check for valid columns
        x_col = self.x_combo.currentText()
        y1_col = self.y1_combo.currentText()
        y2_col = self.y2_combo.currentText()
        if not x_col or not y1_col or x_col not in self.df.columns or y1_col not in self.df.columns:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        x = self.df[x_col]
        # Plot Y1
        if y1_col in self.df.columns:
            ax.plot(x, self.df[y1_col], label=y1_col, color='tab:blue')
            ax.set_ylabel(y1_col, color='tab:blue')
            ax.tick_params(axis='y', labelcolor='tab:blue')
        # Plot Y2 if selected and different from Y1 and present in columns
        if y2_col and y2_col != y1_col and y2_col in self.df.columns:
            ax2 = ax.twinx()
            ax2.plot(x, self.df[y2_col], label=y2_col, color='tab:red')
            ax2.set_ylabel(y2_col, color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
        ax.set_xlabel(x_col)
        # ax.grid(True)  # Removed as per instructions
        ax.legend(loc='upper left')
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())