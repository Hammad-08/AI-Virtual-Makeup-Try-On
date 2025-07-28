from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QGridLayout, QFrame, QGraphicsOpacityEffect, QSlider
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPixmap, QFont, QPainter, QPainterPath, QColor
import sys
import json
import traceback
from camera import CameraThread, ProcessingThread
import onnxruntime as ort

class ModernButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
    def enterEvent(self, event):
        self.animate_hover(True)
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self.animate_hover(False)
        super().leaveEvent(event)
        
    def animate_hover(self, hover):
        pass

class AnimatedProductCard(QFrame):
    def __init__(self, pixmap, title, category, parent=None):
        super().__init__(parent)
        self.pixmap = pixmap
        self.title = title
        self.category = category
        self.parent_window = parent
        self.setup_ui()
        self.setup_animations()
        
    def setup_ui(self):
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFixedSize(200, 220)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Modern card styling
        self.setStyleSheet("""
            AnimatedProductCard {
                background-color: #f8f9fa;
                border: 1px solid #e8e8e8;
                border-radius: 12px;
                padding: 16px;
            }
            AnimatedProductCard:hover {
                border: 2px solid #4A90E2;
                box-shadow: 0 8px 25px rgba(74, 144, 226, 0.15);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Product image
        self.image_label = QLabel()
        self.image_label.setPixmap(self.pixmap.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Product title
        self.title_label = QLabel(self.title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        self.title_label.setStyleSheet("color: #333333; margin-top: 8px;")
        
        layout.addWidget(self.image_label)
        layout.addWidget(self.title_label)
        
    def setup_animations(self):
        # Opacity animation
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(300)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
    def mousePressEvent(self, event):
        if self.parent_window:
            self.parent_window.apply_product(self.title, self.category, self.pixmap)
        super().mousePressEvent(event)
        
    def enterEvent(self, event):
        self.animate_hover(True)
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self.animate_hover(False)
        super().leaveEvent(event)
        
    def animate_hover(self, hover):
        pass

class ModernTabButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(50)
        self.setMinimumWidth(120)
        
        # Default styling
        self.setStyleSheet("""
            ModernTabButton {
                background-color: transparent;
                border: none;
                border-bottom: 3px solid transparent;
                color: #666666;
                padding: 12px 24px;
                text-align: center;
            }
            ModernTabButton:hover {
                color: #4A90E2;
                background-color: rgba(74, 144, 226, 0.05);
            }
            ModernTabButton:checked {
                color: black;
                border-bottom: 3px solid #4A90E2;
                background-color: transparent;
            }
        """)

class CameraWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            CameraWidget {
                background-color: #000000;
                border: 2px solid #e8e8e8;
                border-radius: 12px;
                color: white;
                font-family: 'Segoe UI';
                font-size: 16px;
                font-weight: 500;
            }
        """)
        self.setText("Initializing Camera...")

    def update_frame(self, q_image):
        try:
            if q_image is not None and not q_image.isNull():
                pixmap = QPixmap.fromImage(q_image)
                if not pixmap.isNull():
                    # Create rounded pixmap
                    rounded_pixmap = self.create_rounded_pixmap(pixmap)
                    self.setPixmap(rounded_pixmap)
                    self.setText("")
                else:
                    self.setText("Invalid Frame")
            else:
                self.setText("No Frame")
        except Exception as e:
            print(f"Error updating camera widget: {e}")
            self.setText("Display Error")
            
    def create_rounded_pixmap(self, pixmap):
        # Scale pixmap to fit widget
        scaled_pixmap = pixmap.scaled(
            self.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Create rounded corners
        rounded_pixmap = QPixmap(scaled_pixmap.size())
        rounded_pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(rounded_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        path = QPainterPath()
        path.addRoundedRect(0, 0, scaled_pixmap.width(), scaled_pixmap.height(), 12, 12)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, scaled_pixmap)
        painter.end()
        
        return rounded_pixmap

class ProductLoaderThread(QThread):
    images_loaded = Signal(dict)

    def __init__(self, products_data):
        super().__init__()
        self.products_data = products_data

    def run(self):
        try:
            categories = {}
            for category, category_data in self.products_data.items():
                products = []
                for product in category_data["products"]:
                    image_path = product["image"]
                    try:
                        pixmap = QPixmap(image_path)
                        if pixmap.isNull():
                            # Create colored placeholder based on color_code
                            pixmap = QPixmap(120, 120)
                            color_code = product.get("color_code", "#808080")
                            color = QColor(color_code)
                            pixmap.fill(color)
                        else:
                            pixmap = pixmap.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                        # Create colored placeholder
                        pixmap = QPixmap(120, 120)
                        color_code = product.get("color_code", "#808080")
                        color = QColor(color_code)
                        pixmap.fill(color)
                    
                    product_title = product["title"]
                    products.append((pixmap, product_title, category))
                categories[category] = products
            self.images_loaded.emit(categories)
        except Exception as e:
            print(f"Product loader error: {e}")
            traceback.print_exc()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Virtual Makeup Try-On")
        
        # Hide the maximize button and disable resizing
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        # Set window to maximized and disable resizing
        self.showMaximized()
        self.setFixedSize(self.size())
        
        # Set application font
        font = QFont("Segoe UI", 9)
        self.setFont(font)
        
        # Modern window styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
        """)

        # Track applied products by category
        self.applied_products_by_category = {}
        self.current_category = "Foundation"

        # Initialize threads as None
        self.camera_thread = None
        self.processing_thread = None
        self.product_loader_thread = None

        with open("products.json", "r") as file:
            self.products_data = json.load(file)
            print("Products loaded successfully")

        # Setup UI first
        self.setup_ui()
        
        # Start threads after UI is ready
        QTimer.singleShot(100, self.start_threads)

    def setup_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(24)
        main_layout.setContentsMargins(24, 5, 24, 24)

        # Left section - Products
        left_section = self.create_left_section()
        main_layout.addWidget(left_section, 2)

        # Right section - Camera and Applied Products
        right_section = self.create_right_section()
        main_layout.addWidget(right_section, 1)

    def create_left_section(self):
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(20)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Category tabs
        self.create_category_tabs(left_layout)
        
        # Products grid
        self.create_products_area(left_layout)
        
        return left_widget

    def create_category_tabs(self, parent_layout):
        # Tab container
        tab_container = QWidget()
        tab_container.setFixedHeight(60)
        # border-radius: 12px;
        tab_container.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                
            }
        """)
        
        tab_layout = QHBoxLayout(tab_container)
        tab_layout.setSpacing(0)
        tab_layout.setContentsMargins(8, 0, 8, 5)
        
        # Create tab buttons
        self.tab_buttons = {}
        categories = ["Foundation", "Eyebrow", "Eyeliner", "Eyeshadow", "Lip Stick"]
        
        for i, category in enumerate(categories):
            btn = ModernTabButton(category)
            btn.clicked.connect(lambda checked, cat=category: self.switch_category(cat))
            
            if category == "Foundation":
                btn.setChecked(True)
                
            self.tab_buttons[category] = btn
            tab_layout.addWidget(btn)
            
        parent_layout.addWidget(tab_container)

    def create_products_area(self, parent_layout):
        # Products scroll area
        self.products_scroll = QScrollArea()
        self.products_scroll.setWidgetResizable(True)
        self.products_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.products_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.products_scroll.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: 1px solid #e8e8e8;
                border-radius: 12px;
            }
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #c0c0c0;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #a0a0a0;
            }
        """)

        # Products container
        self.products_widget = QWidget()
        self.products_layout = QGridLayout(self.products_widget)
        self.products_layout.setSpacing(20)
        self.products_layout.setContentsMargins(20, 20, 20, 20)
        
        self.products_scroll.setWidget(self.products_widget)
        parent_layout.addWidget(self.products_scroll)

        # Load products
        self.product_loader_thread = ProductLoaderThread(self.products_data)
        self.product_loader_thread.images_loaded.connect(self.populate_products)
        self.product_loader_thread.start()

    def create_right_section(self):
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(20)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Camera widget
        self.camera_widget = CameraWidget()
        self.camera_widget.setFixedSize(480, 360)
        right_layout.addWidget(self.camera_widget)

        # Applied products section
        applied_section = self.create_applied_products_section()
        right_layout.addWidget(applied_section)
        
        # Clear all button
        self.clear_all_button = ModernButton("Clear All")
        self.clear_all_button.setFixedHeight(45)
        self.clear_all_button.setStyleSheet("""
            ModernButton {
                background-color: #FF6B6B;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
            }
            ModernButton:hover {
                background-color: #FF5252;
            }
            ModernButton:pressed {
                background-color: #E53E3E;
            }
            ModernButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.clear_all_button.clicked.connect(self.clear_applied_products)
        right_layout.addWidget(self.clear_all_button)

        return right_widget

    def create_applied_products_section(self):
        applied_container = QWidget()
        applied_container.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-radius: 12px;
            }
        """)

        container_layout = QVBoxLayout(applied_container)
        container_layout.setSpacing(10)
        container_layout.setContentsMargins(16, 16, 16, 16)

        # Title (moved to top-left with better margin)
        title_label = QLabel("Applied Products")
        title_label.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #222222; margin-bottom: 4px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        container_layout.addWidget(title_label)

        # Scroll area
        self.applied_scroll = QScrollArea()
        self.applied_scroll.setWidgetResizable(True)
        self.applied_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.applied_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.applied_scroll.setFixedHeight(300)
        self.applied_scroll.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }

            QScrollBar:horizontal {
                background: #e0e0e0;
                height: 8px;
                margin: 0px 16px 0px 16px;
                border-radius: 4px;
            }

            QScrollBar::handle:horizontal {
                background: #999999;
                min-width: 30px;
                border-radius: 4px;
            }

            QScrollBar::handle:horizontal:hover {
                background: #7a7a7a;
            }

            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                background: none;
                border: none;
                width: 0;
            }

            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
        """)



        self.applied_products_widget = QWidget()
        self.applied_products_layout = QHBoxLayout(self.applied_products_widget)
        self.applied_products_layout.setSpacing(12)
        self.applied_products_layout.setContentsMargins(0, 0, 0, 0)
        self.applied_products_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.applied_scroll.setWidget(self.applied_products_widget)
        container_layout.addWidget(self.applied_scroll)

        self.update_clear_all_state()
        return applied_container


    def switch_category(self, category):
        # Update button states
        for cat, btn in self.tab_buttons.items():
            btn.setChecked(cat == category)
        
        self.current_category = category
        self.update_products_display()

    def populate_products(self, categories):
        self.categories_data = categories
        self.update_products_display()

    def update_products_display(self):
        # Clear current products
        for i in reversed(range(self.products_layout.count())):
            child = self.products_layout.itemAt(i).widget()
            if child:
                child.setParent(None)

        # Add products for current category
        if hasattr(self, 'categories_data') and self.current_category in self.categories_data:
            products = self.categories_data[self.current_category]
            
            for i, (pixmap, title, category) in enumerate(products):
                product_card = AnimatedProductCard(pixmap, title, category, self)
                
                row = i // 4  # 4 products per row
                col = i % 4
                self.products_layout.addWidget(product_card, row, col)

    def apply_product(self, title, category, pixmap):
        try:
            print(f"Applying product: {title} from {category}")
        
            # Remove existing product from the same category
            if category in self.applied_products_by_category:
                self.remove_applied_product_by_category(category)

            # Find product data
            product_data = next(
                product
                for product in self.products_data[category]["products"]
                if product["title"] == title
            )

            color_code = product_data["color_code"]
            face_elements = product_data["face_elements"]
            default_intensity = 0.3  # Default 30% intensity

            # Notify the processing thread
            if self.processing_thread:
                self.processing_thread.update_makeup(face_elements, color_code, default_intensity)

            # Create applied product widget
            applied_product = self.create_applied_product_widget(pixmap, title, category)
            self.applied_products_layout.addWidget(applied_product)

            # Track the product with intensity
            self.applied_products_by_category[category] = {
                "widget": applied_product,
                "title": title,
                "intensity": default_intensity
            }

            self.update_clear_all_state()
        
        except Exception as e:
            print(f"Error applying product: {e}")
            traceback.print_exc()

    def create_applied_product_widget(self, pixmap, title, category):
        applied_widget = QFrame()
        applied_widget.setObjectName("appliedCard")
        applied_widget.setFixedSize(200, 220)
        applied_widget.setStyleSheet("""
    #appliedCard {
        background-color: #f8f9fa;
       border: 1px solid #e0e0e0;
        border-radius: 12px;
    }
    #appliedCard:hover {
        border: 2px solid #4A90E2;
        background-color: #f8f9fa;
    }
""")


        layout = QVBoxLayout(applied_widget)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # ✕ Remove button
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        remove_btn = QPushButton("✕")
        remove_btn.setFixedSize(20, 20)
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6B6B;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF4C4C;
            }
        """)
        remove_btn.clicked.connect(lambda: self.remove_applied_product_by_category(category))
        top_layout.addWidget(remove_btn)
        layout.addLayout(top_layout)

        # Product image
        image_label = QLabel()
        image_label.setPixmap(pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(image_label)

        # Title
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
        title_label.setStyleSheet("color: #333333;")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        if category in ["Foundation", "Eyeshadow"]:
            # Show intensity slider only for Foundation
            intensity_slider = QSlider(Qt.Orientation.Horizontal)
            intensity_slider.setMinimum(10)
            intensity_slider.setMaximum(100)
            intensity_slider.setValue(30)
            intensity_slider.setFixedHeight(24)
            intensity_slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    background: #e0e0e0;
                    height: 6px;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background: #999999;
                    width: 14px;
                    height: 14px;
                    margin: -5px 0;
                    border-radius: 7px;
                }
                QSlider::sub-page:horizontal {
                    background: #999999;
                    border-radius: 3px;
                }
            """)
            layout.addWidget(intensity_slider)

            intensity_value_label = QLabel("30%")
            intensity_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            intensity_value_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            intensity_value_label.setStyleSheet("color: #555555;")
            layout.addWidget(intensity_value_label)

            # Intensity logic
            def update_intensity(value):
                if value > 50:
                    intensity_slider.setValue(50)  # Restrict slider back to 50
                    return
                intensity_value_label.setText(f"{value}%")
                if self.processing_thread:
                    product_data = next(
                        p for p in self.products_data[category]["products"]
                        if p["title"] == title
                    )
                    self.processing_thread.update_product_intensity(
                        product_data["face_elements"],
                        product_data["color_code"],
                        value / 100.0
                    )


            intensity_slider.valueChanged.connect(update_intensity)

            applied_widget.intensity_slider = intensity_slider

        # Track info regardless of category
        applied_widget.category = category
        applied_widget.title = title

        return applied_widget

    
    def remove_applied_product_by_category(self, category):
        if category in self.applied_products_by_category:
            # Get product info
            product_info = self.applied_products_by_category[category]
            
            # Find product data for face elements
            product_data = next(
                p for p in self.products_data[category]["products"]
                if p["title"] == product_info["title"]
            )
            
            # Notify processing thread
            if self.processing_thread:
                self.processing_thread.remove_makeup(product_data["face_elements"])

            # Remove from UI
            widget = product_info["widget"]
            self.applied_products_layout.removeWidget(widget)
            widget.deleteLater()

            # Remove from tracking
            del self.applied_products_by_category[category]
            
            self.update_clear_all_state()

    def clear_applied_products(self):
        try:
            # Clear all applied products
            while self.applied_products_layout.count():
                child = self.applied_products_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            # Clear tracking
            self.applied_products_by_category.clear()

            # Notify processing thread
            if self.processing_thread:
                self.processing_thread.clear_all_makeup()

            self.update_clear_all_state()
            
        except Exception as e:
            print(f"Error clearing products: {e}")

    def update_clear_all_state(self):
        if hasattr(self, 'clear_all_button'):
            self.clear_all_button.setEnabled(bool(self.applied_products_by_category))

    def start_threads(self):
        try:
            # Initialize threads
            self.camera_thread = CameraThread()
            self.processing_thread = ProcessingThread()

            # Connect signals
            self.camera_thread.frame_captured.connect(self.processing_thread.update_frame)
            self.processing_thread.frame_ready.connect(self.camera_widget.update_frame)
            
            # Start threads
            print("Starting camera thread...")
            self.camera_thread.start()
            print("Starting processing thread...")
            self.processing_thread.start()
            
        except Exception as e:
            print(f"Error starting threads: {e}")
            traceback.print_exc()

    def closeEvent(self, event):
        try:
            print("Closing application...")
            
            # Stop threads properly
            if self.camera_thread:
                self.camera_thread.stop()
            if self.processing_thread:
                self.processing_thread.stop()
            if self.product_loader_thread:
                self.product_loader_thread.quit()
            
            # Force quit after a short delay
            QTimer.singleShot(1000, QApplication.instance().quit)
            
            event.accept()
        except Exception as e:
            print(f"Error during close: {e}")
            event.accept()
            
            
def check_gpu_provider():
    providers = ort.get_available_providers()
    print("ONNX Execution Providers:", providers)
    if "CUDAExecutionProvider" in providers:
        print("✅ GPU (CUDA) is available and will be used for ONNX inference.")
    else:
        print("❌ GPU not available. Using CPU for ONNX inference.")


def main():
    try:
        check_gpu_provider()  # ✅ Check GPU availability
        app = QApplication(sys.argv)
        
        # Set application style
        app.setStyle('Fusion')
        
        window = MainWindow()
        window.show()
        print("Modern Virtual Makeup Application started successfully")
        sys.exit(app.exec())
    except Exception as e:
        print(f"Application error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
    