from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QPointF, QRectF
import numpy as np


class NativePlotWidget(QWidget):
    """Native PyQt5 plot widget for real-time EEG display"""
    
    def __init__(self, buffer_length, sample_rate):
        super().__init__()
        self.buffer_length = buffer_length
        self.sample_rate = sample_rate
        self.data = np.zeros(buffer_length)
        self.data2 = np.zeros(buffer_length)
        self.setMinimumHeight(200)
        
        # Plot settings
        self.margin_left = 70  # More space for Y-axis labels
        self.margin_right = 30
        self.margin_top = 30
        self.margin_bottom = 40
        self.y_min = -100
        self.y_max = 100
        
    def update_data(self, new_data, new_data2=None):
        """Update the plot data and trigger repaint"""
        self.data = new_data
        if new_data2 is not None:
            self.data2 = new_data2
        
        # Auto-scale Y axis with 0 always visible
        if len(new_data) > 0:
            # Consider both datasets for scaling
            data_min = np.min(new_data)
            data_max = np.max(new_data)
            
            if new_data2 is not None and len(new_data2) > 0:
                data_min = min(data_min, np.min(new_data2))
                data_max = max(data_max, np.max(new_data2))
            
            # Ensure 0 is always included
            abs_max = max(abs(data_min), abs(data_max))
            # Round to nearest 50 for clean ticks
            abs_max = max(50, ((abs_max // 50) + 1) * 50)
            
            self.y_min = -abs_max
            self.y_max = abs_max
        self.update()  # Triggers paintEvent
        
    def paintEvent(self, event):
        """Custom painting using QPainter"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        plot_width = width - self.margin_left - self.margin_right
        plot_height = height - self.margin_top - self.margin_bottom
        
        # Draw axes
        painter.setPen(QPen(QColor('#333333'), 1))
        # X-axis
        painter.drawLine(self.margin_left, height - self.margin_bottom, 
                        width - self.margin_right, height - self.margin_bottom)
        # Y-axis
        painter.drawLine(self.margin_left, self.margin_top, 
                        self.margin_left, height - self.margin_bottom)
        
        # Calculate Y-axis tick positions (ensure 0 is included)
        tick_interval = 50 if self.y_max <= 150 else 100
        y_ticks = []
        current_tick = 0
        while current_tick <= self.y_max:
            y_ticks.append(current_tick)
            if current_tick != 0:
                y_ticks.append(-current_tick)
            current_tick += tick_interval
        y_ticks.sort()
        
        # Draw horizontal grid lines and Y labels
        painter.setPen(QPen(QColor('#cccccc'), 1, Qt.DotLine))
        for tick_val in y_ticks:
            y_norm = (tick_val - self.y_min) / (self.y_max - self.y_min)
            y = int(height - self.margin_bottom - y_norm * plot_height)
            painter.drawLine(self.margin_left, y, width - self.margin_right, y)
            
        # Draw vertical grid lines  
        for i in range(7):  # 0, 5, 10, 15, 20, 25, 30 seconds
            x = int(self.margin_left + i * plot_width / 6)
            painter.drawLine(x, self.margin_top, x, height - self.margin_bottom)
            
        # Draw axis labels
        painter.setPen(QPen(QColor('#333333'), 1))
        font = QFont('Arial', 9)
        painter.setFont(font)
        
        # X-axis labels (time in seconds)
        for i in range(7):
            time_val = i * 5  # 0, 5, 10, 15, 20, 25, 30 seconds
            x = int(self.margin_left + i * plot_width / 6)
            painter.drawText(QRectF(x - 20, height - self.margin_bottom + 5, 40, 20),
                           Qt.AlignCenter, f'{time_val}')
                           
        # Y-axis labels with proper ticks including 0
        for tick_val in y_ticks:
            y_norm = (tick_val - self.y_min) / (self.y_max - self.y_min)
            y = int(height - self.margin_bottom - y_norm * plot_height)
            painter.drawText(QRectF(5, y - 10, self.margin_left - 15, 20),
                           Qt.AlignRight | Qt.AlignVCenter, f'{tick_val}')
        
        # Draw the second signal line (grey)
        if len(self.data2) > 1:
            painter.setPen(QPen(QColor('#808080'), 1.5))
            
            # Downsample for performance (draw every 10th point)
            step = max(1, len(self.data2) // 768)  # Max 768 points to draw
            
            # Create path
            prev_point = None
            for i in range(0, len(self.data2), step):
                # Map data to pixel coordinates
                x = self.margin_left + (i / len(self.data2)) * plot_width
                # Normalize y value to plot range
                if self.y_max != self.y_min:
                    y_norm = (self.data2[i] - self.y_min) / (self.y_max - self.y_min)
                else:
                    y_norm = 0.5
                y = height - self.margin_bottom - y_norm * plot_height
                
                current_point = QPointF(x, y)
                if prev_point is not None:
                    painter.drawLine(prev_point, current_point)
                prev_point = current_point
        
        # Draw the main signal line (green)
        painter.setPen(QPen(QColor('#4CAF50'), 1.5))
        
        if len(self.data) > 1:
            # Downsample for performance (draw every 10th point)
            step = max(1, len(self.data) // 768)  # Max 768 points to draw
            
            # Create path
            prev_point = None
            for i in range(0, len(self.data), step):
                # Map data to pixel coordinates
                x = self.margin_left + (i / len(self.data)) * plot_width
                # Normalize y value to plot range
                if self.y_max != self.y_min:
                    y_norm = (self.data[i] - self.y_min) / (self.y_max - self.y_min)
                else:
                    y_norm = 0.5
                y = height - self.margin_bottom - y_norm * plot_height
                
                current_point = QPointF(x, y)
                if prev_point is not None:
                    painter.drawLine(prev_point, current_point)
                prev_point = current_point
                
        # Draw legend in upper right corner
        legend_x = width - 120
        legend_y = 40
        
        # Draw legend background
        painter.fillRect(QRectF(legend_x - 5, legend_y - 5, 110, 45), QColor(255, 255, 255, 200))
        painter.setPen(QPen(QColor('#cccccc'), 1))
        painter.drawRect(QRectF(legend_x - 5, legend_y - 5, 110, 45))
        
        # Draw legend items
        painter.setPen(QPen(QColor('#4CAF50'), 2))
        painter.drawLine(legend_x, legend_y + 5, legend_x + 20, legend_y + 5)
        painter.setPen(QPen(QColor('#333333'), 1))
        painter.drawText(QRectF(legend_x + 25, legend_y - 2, 80, 15),
                        Qt.AlignLeft | Qt.AlignVCenter, '0.1 - 45 Hz')
        
        painter.setPen(QPen(QColor('#808080'), 2))
        painter.drawLine(legend_x, legend_y + 20, legend_x + 20, legend_y + 20)
        painter.setPen(QPen(QColor('#333333'), 1))
        painter.drawText(QRectF(legend_x + 25, legend_y + 13, 80, 15),
                        Qt.AlignLeft | Qt.AlignVCenter, '0.5 - 4 Hz')
        
        # Draw axis titles (not bold)
        painter.setPen(QPen(QColor('#333333'), 1))
        # Keep same font, not bold
        painter.drawText(QRectF(0, height - 25, width, 20),
                        Qt.AlignCenter, 'Time (seconds)')
        
        # Y-axis label (rotated) with more space from tick labels
        painter.save()
        painter.translate(12, height / 2)
        painter.rotate(-90)
        painter.drawText(QRectF(-50, -10, 100, 20),
                        Qt.AlignCenter, 'Amplitude (μV)')
        painter.restore()