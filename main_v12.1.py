import sys
import os
import time
import struct
import numpy as np
import serial
import serial.tools.list_ports
import csv  
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar 
import matplotlib.gridspec as gridspec # Para el layout de 3 gráficos
import matplotlib.patheffects as pe # Para contornos de texto legibles en el mapa
from matplotlib.colors import LightSource # Para el renderizado de sombras (Hillshade)

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                             QGroupBox, QCheckBox, QSlider, QFrame, QSpinBox, QGridLayout, 
                             QTabWidget, QFileDialog, QComboBox, QLineEdit)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QRect
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor

# Intentamos importar las librerías oficiales de Prophesee (Metavision)
TRY_METAVISION = True
try:
    from metavision_core.event_io import EventsIterator
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
    HAS_METAVISION = True
except ImportError:
    HAS_METAVISION = False

# --- LIBRERÍAS GEOESPACIALES ---
try:
    import rasterio
    from rasterio.windows import from_bounds
    from shapely.geometry import LineString
    from pyproj import Transformer, Geod
    HAS_GEOSPATIAL = True
except ImportError:
    HAS_GEOSPATIAL = False
    print("ADVERTENCIA: Librerías geoespaciales no encontradas (rasterio, shapely, pyproj).")

# --- CLASE DE VIDEO CON ROI ---
class ROI_VideoLabel(QLabel):
    """Etiqueta inteligente que permite dibujar un ROI con el mouse"""
    roi_changed = pyqtSignal(tuple) 

    def __init__(self):
        super().__init__()
        self.selection_rect = None
        self.start_point = None
        self.is_selecting = False
        self.roi_active = False    
        
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #000; border: 2px solid #555;")
        self.setMinimumSize(320, 240) 
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_selecting = True
            self.start_point = event.pos()
            self.selection_rect = QRect(self.start_point, self.start_point)
            self.update() 

    def mouseMoveEvent(self, event):
        if self.is_selecting and self.start_point:
            self.selection_rect = QRect(self.start_point, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_selecting = False
            if self.selection_rect:
                self.calculate_sensor_roi()

    def calculate_sensor_roi(self):
        if not self.pixmap(): return
        
        disp_w = self.width()
        disp_h = self.height()
        sensor_w = 640
        sensor_h = 480
        
        if disp_w == 0 or disp_h == 0: return

        scale_x = sensor_w / disp_w
        scale_y = sensor_h / disp_h
        
        r = self.selection_rect
        x = int(r.x() * scale_x)
        y = int(r.y() * scale_y)
        w = int(r.width() * scale_x)
        h = int(r.height() * scale_y)
        
        x = max(0, min(x, sensor_w-1))
        y = max(0, min(y, sensor_h-1))
        w = max(1, min(w, sensor_w - x))
        h = max(1, min(h, sensor_h - y))
        
        self.roi_changed.emit((x, y, w, h))

    def paintEvent(self, event):
        super().paintEvent(event) 
        
        if self.selection_rect:
            painter = QPainter(self)
            color = QColor(255, 0, 0) if (self.is_selecting or self.roi_active) else QColor(100, 100, 100)
            pen = QPen(color, 2, Qt.PenStyle.SolidLine) 
            painter.setPen(pen)
            painter.drawRect(self.selection_rect)
    
    def set_visual_roi_from_coords(self, x, y, w, h):
        if not self.pixmap(): return
        
        disp_w = self.width()
        disp_h = self.height()
        sensor_w = 640
        sensor_h = 480
        
        if disp_w == 0 or disp_h == 0: return

        scale_x = disp_w / sensor_w
        scale_y = disp_h / sensor_h
        
        vis_x = int(x * scale_x)
        vis_y = int(y * scale_y)
        vis_w = int(w * scale_x)
        vis_h = int(h * scale_y)
        
        self.selection_rect = QRect(vis_x, vis_y, vis_w, vis_h)
        self.update() 


class CameraWorker(QThread):
    image_signal = pyqtSignal(QImage)
    stats_signal = pyqtSignal(str, float)
    
    def __init__(self, use_simulation=False):
        super().__init__()
        self.running = True
        self.recording = False
        
        self.req_start_rec = None
        self.req_stop_rec = False
        
        self.use_simulation = use_simulation
        self.width = 640
        self.height = 480
        self.show_raw_data = False
        
        self.accumulation_time_us = 10000 
        self.target_fps = 60              

        self.roi_enabled = False
        self.roi_coords = (0, 0, 640, 480)
        
        self.biases_facility = None
        self.current_biases = {
            'bias_diff': 299,
            'bias_diff_on': 350,  
            'bias_diff_off': 150, 
            'bias_refr': 1500     
        }
        self.sim_sensitivity = 1.0 
        self.sync_mode = "STANDALONE" 
        self.serial_number = ""

    def start_recording(self, filename):
        self.req_start_rec = filename
        return True

    def stop_recording(self):
        self.req_stop_rec = True

    def update_bias(self, name, value):
        self.current_biases[name] = value
        if self.biases_facility:
            try: self.biases_facility.set(name, value)
            except: pass
        if self.use_simulation and name == 'bias_diff_on':
            self.sim_sensitivity = 1000.0 / (value + 1)

    def update_timing_params(self, accum_us, fps):
        self.accumulation_time_us = max(200, accum_us)
        self.target_fps = max(1, fps)

    def run(self):
        if self.use_simulation or not HAS_METAVISION:
            self._run_simulation()
        else:
            self._run_camera()

    def _run_camera(self):
        while self.running:
            try:
                current_delta_t = self.accumulation_time_us
                cam_path = self.serial_number if self.serial_number != "" else ""
                mv_iterator = EventsIterator(input_path=cam_path, delta_t=current_delta_t)
                
                i_events_stream = None
                
                try:
                    if hasattr(mv_iterator, 'reader'): device = mv_iterator.reader.device
                    elif hasattr(mv_iterator, '_reader'): device = mv_iterator._reader.device
                    else: device = None
                    
                    if device:
                        self.biases_facility = device.get_i_ll_biases()
                        for name, val in self.current_biases.items():
                            try: self.biases_facility.set(name, val)
                            except: pass
                            
                        try:
                            sync_facility = device.get_i_camera_synchronization()
                            if sync_facility:
                                if self.sync_mode == "MASTER": sync_facility.set_mode_master()
                                elif self.sync_mode == "SLAVE": sync_facility.set_mode_slave()
                                else: sync_facility.set_mode_standalone()
                        except: pass
                        
                        try:
                            i_events_stream = device.get_i_events_stream()
                        except: pass
                        
                except Exception as e_hal:
                    print(f"Advertencia Hardware: {e_hal}")

                visual_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                last_visual_time = time.time()
                
                for evs in mv_iterator:
                    if not self.running: break
                    
                    if self.req_start_rec:
                        if i_events_stream:
                            i_events_stream.log_raw_data(self.req_start_rec)
                            self.recording = True
                        self.req_start_rec = None
                        
                    if self.req_stop_rec:
                        if i_events_stream:
                            i_events_stream.stop_log_raw_data()
                        self.recording = False
                        self.req_stop_rec = False
                    
                    if self.accumulation_time_us != current_delta_t:
                        break 

                    events_to_process = evs
                    if self.roi_enabled:
                        rx, ry, rw, rh = self.roi_coords
                        mask = (evs['x'] >= rx) & (evs['x'] < rx + rw) & \
                               (evs['y'] >= ry) & (evs['y'] < ry + rh)
                        events_to_process = evs[mask]

                    on_events = events_to_process[events_to_process['p'] == 1]
                    off_events = events_to_process[events_to_process['p'] == 0]
                    
                    if len(on_events) > 0:
                        visual_buffer[on_events['y'], on_events['x']] = [0, 255, 0] 
                    if len(off_events) > 0:
                        visual_buffer[off_events['y'], off_events['x']] = [255, 0, 0]

                    now = time.time()
                    visual_period = 1.0 / self.target_fps
                    
                    if (now - last_visual_time) >= visual_period:
                        h, w, ch = visual_buffer.shape
                        qt_img = QImage(visual_buffer.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                        self.image_signal.emit(qt_img)
                        visual_buffer.fill(0) 
                        last_visual_time = now
                        
                        debug_text = ""
                        if self.show_raw_data:
                            debug_text += f"[FÍSICA] Delta T: {current_delta_t}µs\n"
                            if self.roi_enabled: debug_text += f"[ROI VISUAL] {self.roi_coords}\n"
                            if self.recording: debug_text += "[REC] Guardando sensor completo (EVT3)\n"
                            if len(events_to_process) > 0:
                                sample = events_to_process[:5]
                                debug_text += "--- DATA SAMPLE ---\n"
                                for e in sample:
                                    debug_text += f"T:{e['t']} X:{e['x']} Y:{e['y']} P:{e['p']}\n"
                        
                        event_rate = len(evs) / (current_delta_t)
                        self.stats_signal.emit(debug_text, event_rate)

            except Exception as e:
                self.stats_signal.emit(f"Error Driver: {str(e)}", 0)
                time.sleep(1) 
            finally:
                if i_events_stream and self.recording:
                    i_events_stream.stop_log_raw_data()
                    self.recording = False

    def _run_simulation(self):
        print("Modo simulación no tiene soporte para guardado Metavision.")
        while self.running:
            time.sleep(0.1)

    def stop(self):
        self.stop_recording()
        self.running = False
        self.wait()



class IMUWorker(QThread):
    log_signal = pyqtSignal(str)
    stats_signal = pyqtSignal(float, float, int) 
    coord_signal = pyqtSignal(float, float)
    pps_signal = pyqtSignal(int)  # <-- NUEVA SEÑAL PARA EL MONITOR PPS
    
    def __init__(self, port, baudrate):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = True
        self.recording = False
        self.armed = False     # <-- NUEVO ESTADO: Armado y esperando Sello Raíz
        self.bin_file = None
        self.raw_mode = False
        self.pps_count = 0  
        
        self.PACKET_SIZE = 18
        self.HEADER_IMU = 0xBBAA 
        self.HEADER_GPS = 0xBBCC 
        self.HEADER_COORD = 0xBBDD
        self.HEADER_ATM = 0xFFEE # <-- NUEVO: ENCABEZADO DEL MARCADOR ABSOLUTO

    def start_recording(self, filename, raw_mode):
        self.raw_mode = raw_mode
        try:
            self.bin_file = open(filename, "wb", buffering=0)
            if self.raw_mode:
                self.recording = True
            else:
                self.armed = True # Quedamos a la espera del satélite
                self.log_signal.emit("⏳ [ARMADO] Esperando Marcador de Tiempo Absoluto del GPS...")
            return True
        except Exception as e:
            self.log_signal.emit(f"[ERROR] No se pudo crear archivo: {e}\n")
            return False

    def stop_recording(self):
        self.recording = False
        self.armed = False
        if self.bin_file:
            try: self.bin_file.close()
            except: pass
            self.bin_file = None

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=0.02)
            ser.dtr = False; ser.rts = False; time.sleep(0.05)
            ser.dtr = True; ser.rts = True; time.sleep(0.1)
            ser.reset_input_buffer()
            self.log_signal.emit(f"--- CONECTADO A {self.port} @ {self.baudrate} bps ---\n")
            
            last_check = time.time()
            packet_counter_hz = 0
            local_bytes = 0
            sync_errors = 0
            buffer = bytearray()
            
            while self.running:
                data_processed = False
                if ser.in_waiting:
                    chunk = ser.read(ser.in_waiting)
                    if chunk:
                        if self.recording and self.raw_mode:
                            self.bin_file.write(chunk)
                            local_bytes += len(chunk)
                            data_processed = True
                            continue 

                        buffer.extend(chunk)
                        local_bytes += len(chunk)
                        data_processed = True

                    if not self.raw_mode:
                        while len(buffer) >= self.PACKET_SIZE:
                            header_check = struct.unpack("<H", buffer[:2])[0]
                            
                            # Si es paquete estándar (IMU o PPS)
                            if header_check == self.HEADER_IMU or header_check == self.HEADER_GPS:
                                data_packet = buffer[:self.PACKET_SIZE]
                                if self.recording and self.bin_file:
                                    self.bin_file.write(data_packet)
                                
                                if header_check == self.HEADER_IMU:
                                    packet_counter_hz += 1
                                elif header_check == self.HEADER_GPS:
                                    if not hasattr(self, 'pps_count'): self.pps_count = 0
                                    self.pps_count += 1
                                    try:
                                        self.pps_signal.emit(self.pps_count) # Protegido contra caídas de GUI
                                    except Exception: pass
                                    
                                del buffer[:self.PACKET_SIZE]
                                if sync_errors > 0: sync_errors = 0
                            
                            # --- PARSER DE COORDENADAS Y SELLO RAÍZ (ATM) ---
                            elif header_check == self.HEADER_COORD:
                                try:
                                    # Desempaquetamos Header(2) + Lat(4) + Lon(4) + Y(2) + M(1) + D(1) + H(1) + M(1) + S(1) + V(1) = 18 bytes
                                    p = struct.unpack_from("<H f f H B B B B B B", buffer[:18])
                                    latitud, longitud = p[1], p[2]
                                    year, month, day, hour, minute, second, valid = p[3], p[4], p[5], p[6], p[7], p[8], p[9]
                                    
                                    try: self.coord_signal.emit(latitud, longitud)
                                    except Exception: pass

                                    # LÓGICA DEL SELLO RAÍZ (ATM)
                                    if self.armed and valid == 1:
                                        import datetime
                                        dt = datetime.datetime(year, month, day, hour, minute, second, tzinfo=datetime.timezone.utc)
                                        unix_timestamp = dt.timestamp()
                                        
                                        # Creamos paquete ATM: Header 0xFFEE (2b) + Timestamp (8b double) + 8 bytes relleno = 18 bytes exactos
                                        atm_packet = struct.pack("<H d 8x", self.HEADER_ATM, unix_timestamp)
                                        self.bin_file.write(atm_packet)
                                        
                                        self.armed = False
                                        self.recording = True # ¡Abre la compuerta del torrente de datos!
                                        self.log_signal.emit(f"🎯 [ATM FIJADO] Época: {hour:02d}:{minute:02d}:{second:02d} UTC. ¡Grabando!")
                                    
                                    # Si ya estamos grabando normal, guardamos el paquete de tracking espacial
                                    elif self.recording and self.bin_file:
                                        self.bin_file.write(buffer[:self.PACKET_SIZE])
                                    
                                    del buffer[:self.PACKET_SIZE]
                                    if sync_errors > 0: sync_errors = 0
                                except Exception as e:
                                    del buffer[0]; sync_errors += 1
                                    
                            else:
                                del buffer[0]
                                sync_errors += 1
                                
                now = time.time()
                if now - last_check >= 1.0: 
                    hz = packet_counter_hz / (now - last_check)
                    bps = local_bytes / (now - last_check)
                    self.stats_signal.emit(hz, bps, sync_errors)
                    packet_counter_hz = 0
                    local_bytes = 0
                    last_check = now
                
                if not data_processed:
                    time.sleep(0.001) 
                    
        except Exception as e:
            self.log_signal.emit(f"ERROR SERIAL: {str(e)}\n")
        finally:
            self.stop_recording()
            try: ser.close()
            except: pass
            self.log_signal.emit("--- DESCONECTADO ---\n")

    def stop(self):
        self.running = False
        self.wait()

class ReconWorker(QThread):
    """Integrador Vectorial (Euler) para Reconstrucción en Vivo (Camino A)"""
    image_signal = pyqtSignal(QImage)
    stats_signal = pyqtSignal(str, float)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.width = 640
        self.height = 480
        self.accumulation_time_us = 16666 # ~60 FPS (Delta T)
        
        # Parámetros físicos (modificables desde la GUI)
        self.tau = 0.1  # Constante de memoria en segundos
        self.c_on = 0.15
        self.c_off = 0.15
        self.serial_number = ""

    def run(self):
        try:
            mv_iterator = EventsIterator(input_path=self.serial_number, delta_t=self.accumulation_time_us)
            
            # Matriz de estado de energía logarítmica (El "Pizarrón")
            log_i = np.zeros((self.height, self.width), dtype=np.float32)

            for evs in mv_iterator:
                if not self.running: break
                
                # 1. DECAIMIENTO (Fuga de memoria exponencial)
                dt_sec = self.accumulation_time_us * 1e-6
                log_i *= np.exp(-dt_sec / self.tau)

                if len(evs) > 0:
                    # 2. ACUMULACIÓN CUÁNTICA (Salto de eventos)
                    on_mask = evs['p'] == 1
                    off_mask = evs['p'] == 0
                    
                    # np.add.at es el secreto de NumPy para sumar rápido en coordenadas específicas
                    np.add.at(log_i, (evs['y'][on_mask], evs['x'][on_mask]), self.c_on)
                    np.add.at(log_i, (evs['y'][off_mask], evs['x'][off_mask]), -self.c_off)

                # 3. RECUPERACIÓN LINEAL (Intensidad = e^log_i)
                intensity = np.exp(log_i)

                # 4. AUTO-CONTRASTE ESTADÍSTICO (Percentiles 1 y 99 para ignorar ruido extremo)
                p_low, p_high = np.percentile(intensity, (1, 99))
                if p_high > p_low:
                    norm_img = np.clip((intensity - p_low) / (p_high - p_low) * 255.0, 0, 255).astype(np.uint8)
                else:
                    norm_img = np.zeros((self.height, self.width), dtype=np.uint8)

                # Convertimos a QImage RGB para la interfaz
                rgb_img = np.stack((norm_img,)*3, axis=-1)
                qt_img = QImage(rgb_img.data, self.width, self.height, 3*self.width, QImage.Format.Format_RGB888).copy()
                
                self.image_signal.emit(qt_img)

                # Enviamos datos para análisis
                rate = len(evs) / self.accumulation_time_us
                stats_txt = f"[FÍSICA DEL RECONSTRUCTOR]\nDelta T: 16.6ms\nTau (Memoria): {self.tau} s\nSalto C_on: {self.c_on:.2f}\nSalto C_off: {self.c_off:.2f}"
                self.stats_signal.emit(stats_txt, rate)

        except Exception as e:
            self.stats_signal.emit(f"Error Recon: {str(e)}", 0)

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visor SilkyEvCam - Obukhov Lab Control")
        self.setGeometry(50, 50, 1300, 700)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QTabWidget::pane { border: 1px solid #333; background-color: #121212; }
            QTabBar::tab { background-color: #1e1e1e; color: #888; padding: 10px 20px; border: 1px solid #333; border-bottom: none; margin-right: 2px; }
            QTabBar::tab:selected { background-color: #2b2b2b; color: #00aaaa; border-bottom: 2px solid #00aaaa; }
            QWidget { background-color: #121212; color: #ffffff; }
            QLabel { color: #ccc; font-size: 12px; background: transparent; }
            QGroupBox { border: 1px solid #333; margin-top: 15px; font-weight: bold; color: #00aaaa; font-size: 13px; background-color: #1a1a1a; border-radius: 5px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 10px; }
            QTextEdit { background-color: #0a0a0a; color: #00ff00; font-family: 'Consolas', monospace; font-size: 11px; border: 1px solid #333; }
            QPushButton { background-color: #252525; color: white; border: 1px solid #444; border-radius: 4px; padding: 6px; font-weight: bold; }
            QPushButton:hover { background-color: #353535; border: 1px solid #00aaaa; }
            QPushButton:disabled { background-color: #151515; color: #444; border: 1px solid #222; }
            QSlider::groove:horizontal { border: 1px solid #333; height: 6px; background: #000; margin: 2px 0; }
            QSlider::handle:horizontal { background: #00aaaa; border: 1px solid #00aaaa; width: 16px; margin: -5px 0; border-radius: 8px; }
            QSpinBox { background-color: #222; color: #00ff00; border: 1px solid #444; padding: 3px; font-weight: bold; }
            QCheckBox { color: #ddd; spacing: 5px; }
            QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #444; background: #222; }
            QCheckBox::indicator:checked { background: #00aaaa; border: 1px solid #00aaaa; }
        """)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.setStyleSheet("QTabBar::tab { padding: 10px 20px; font-weight: bold; font-size: 14px; }")
        
        self.tab_live = QWidget()
        live_layout = QHBoxLayout(self.tab_live)
        self.panel_biases = self.create_bias_panel()
        self.panel_visor = self.create_visor_panel()
        self.panel_controls = self.create_controls_panel()
        live_layout.addWidget(self.panel_biases, 15)   
        live_layout.addLayout(self.panel_visor, 60)    
        live_layout.addWidget(self.panel_controls, 25) 
        self.tabs.addTab(self.tab_live, "🔴 Laboratorio (Adquisición)")

        self.tab_recon = self.create_recon_tab()
        self.tabs.addTab(self.tab_recon, "👁️ Enfoque (Reconstrucción)")

        self.tab_imu = self.create_imu_tab()
        self.tabs.addTab(self.tab_imu, "🧭 Adquisición IMU / GPS")

        self.tab_analysis = self.create_analysis_tab()
        self.tabs.addTab(self.tab_analysis, "📊 Conversor y Gráficos IMU")

        self.tab_topo = self.create_topo_tab()
        self.tabs.addTab(self.tab_topo, "⛰️ Topografía y Enlace")

        self.worker_1 = None
        self.worker_2 = None
        self.imu_worker = None

        # Variables internas para guardar la última coordenada recibida del ESP32
        self.latest_lat = None
        self.latest_lon = None

    def create_bias_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        container.setFixedWidth(280)
        grp = QGroupBox("Configuración Sensor (Biases)")
        l_grp = QVBoxLayout()
        l_grp.setSpacing(15) 
        l_grp.addWidget(QLabel("⚠️ Ajustes Físicos (Voltaje)"))
        self.sl_diff, self.sb_diff = self.add_bias_row(l_grp, "bias_diff (Ref Sagrada)", 250, 350, 299, "Nivel base de referencia. ¡Cuidado!")
        self.sl_diff_on, self.sb_diff_on = self.add_bias_row(l_grp, "bias_diff_on (Sensib. ON)", 300, 600, 374, "Menor valor = Más ruido verde")
        self.sl_diff_off, self.sb_diff_off = self.add_bias_row(l_grp, "bias_diff_off (Sensib. OFF)", 100, 290, 221, "Menor valor = Más ruido rojo")
        self.sl_refr, self.sb_refr = self.add_bias_row(l_grp, "bias_refr (Refractario)", 1300, 1800, 1500, "Tiempo muerto tras evento (µs)")
        l_grp.addStretch()
        grp.setLayout(l_grp)
        layout.addWidget(grp)
        return container

    def add_bias_row(self, layout, name, min_val, max_val, default_val, tooltip):
        lbl = QLabel(f"<b>{name}</b>")
        lbl.setToolTip(tooltip)
        layout.addWidget(lbl)
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default_val)
        spin.setFixedWidth(100)
        spin.setStyleSheet("background-color: #333; color: #0f0; font-weight: bold;")
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        clean_name = name.split(" ")[0]
        slider.valueChanged.connect(lambda v: self.on_bias_change(clean_name, v))
        row_layout.addWidget(slider)
        row_layout.addWidget(spin)
        layout.addWidget(row_widget)
        return slider, spin

    def create_visor_panel(self):
        main_layout = QVBoxLayout()
        row_cam1 = QHBoxLayout()
        ctrl_cam1 = QGroupBox("CÁMARA 1 (Arriba)")
        l_ctrl1 = QVBoxLayout()
        self.txt_serial_1 = QLineEdit(); self.txt_serial_1.setPlaceholderText("N° Serie (Vacío=Auto)")
        self.cmb_sync_1 = QComboBox(); self.cmb_sync_1.addItems(["Standalone", "MASTER", "SLAVE"])
        self.cmb_sync_1.setStyleSheet("color: #ffaa00;")
        self.btn_start_1 = QPushButton("▶ INICIAR CAM 1")
        self.btn_start_1.clicked.connect(self.start_camera_1)
        self.btn_start_1.setStyleSheet("background-color: #2da44e;")
        self.lbl_status_1 = QLabel("Estado: Detenida")
        self.lbl_perf_1 = QLabel("Tasa: 0.00 MEv/s")
        l_ctrl1.addWidget(QLabel("Puerto / N° Serie:"))
        l_ctrl1.addWidget(self.txt_serial_1)
        l_ctrl1.addWidget(QLabel("Modo Sync (Hirose):"))
        l_ctrl1.addWidget(self.cmb_sync_1)
        l_ctrl1.addWidget(self.btn_start_1)
        l_ctrl1.addWidget(self.lbl_status_1)
        l_ctrl1.addWidget(self.lbl_perf_1)
        l_ctrl1.addStretch()
        ctrl_cam1.setLayout(l_ctrl1)
        ctrl_cam1.setFixedWidth(200)
        self.lbl_video_1 = ROI_VideoLabel()
        self.lbl_video_1.roi_changed.connect(self.on_roi_changed)
        row_cam1.addWidget(ctrl_cam1)
        row_cam1.addWidget(self.lbl_video_1)

        row_cam2 = QHBoxLayout()
        ctrl_cam2 = QGroupBox("CÁMARA 2 (Abajo)")
        l_ctrl2 = QVBoxLayout()
        self.txt_serial_2 = QLineEdit(); self.txt_serial_2.setPlaceholderText("N° Serie (Vacío=Auto)")
        self.cmb_sync_2 = QComboBox(); self.cmb_sync_2.addItems(["Standalone", "MASTER", "SLAVE"])
        self.cmb_sync_2.setCurrentIndex(2) 
        self.cmb_sync_2.setStyleSheet("color: #00aaaa;")
        self.btn_start_2 = QPushButton("▶ INICIAR CAM 2")
        self.btn_start_2.clicked.connect(self.start_camera_2)
        self.btn_start_2.setStyleSheet("background-color: #2da44e;")
        self.lbl_status_2 = QLabel("Estado: Detenida")
        self.lbl_perf_2 = QLabel("Tasa: 0.00 MEv/s")
        l_ctrl2.addWidget(QLabel("Puerto / N° Serie:"))
        l_ctrl2.addWidget(self.txt_serial_2)
        l_ctrl2.addWidget(QLabel("Modo Sync (Hirose):"))
        l_ctrl2.addWidget(self.cmb_sync_2)
        l_ctrl2.addWidget(self.btn_start_2)
        l_ctrl2.addWidget(self.lbl_status_2)
        l_ctrl2.addWidget(self.lbl_perf_2)
        l_ctrl2.addStretch()
        ctrl_cam2.setLayout(l_ctrl2)
        ctrl_cam2.setFixedWidth(200)
        self.lbl_video_2 = ROI_VideoLabel()
        self.lbl_video_2.roi_changed.connect(self.on_roi_changed)
        row_cam2.addWidget(ctrl_cam2)
        row_cam2.addWidget(self.lbl_video_2)

        main_layout.addLayout(row_cam1)
        main_layout.addLayout(row_cam2)
        return main_layout

    def create_controls_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        container.setFixedWidth(280)
        
        grp_status = QGroupBox("Estado Global y ROI")
        l_status = QVBoxLayout()
        self.lbl_status = QLabel("● Sistema Detenido")
        l_status.addWidget(self.lbl_status)
        grid_roi = QGridLayout()
        grid_roi.setContentsMargins(0, 5, 0, 0)
        self.sb_roi_x = self.create_roi_spinbox(0, 639, "X")
        self.sb_roi_y = self.create_roi_spinbox(0, 479, "Y")
        self.sb_roi_w = self.create_roi_spinbox(1, 640, "W")
        self.sb_roi_h = self.create_roi_spinbox(1, 480, "H")
        grid_roi.addWidget(QLabel("X:"), 0, 0); grid_roi.addWidget(self.sb_roi_x, 0, 1)
        grid_roi.addWidget(QLabel("W:"), 0, 2); grid_roi.addWidget(self.sb_roi_w, 0, 3)
        grid_roi.addWidget(QLabel("Y:"), 1, 0); grid_roi.addWidget(self.sb_roi_y, 1, 1)
        grid_roi.addWidget(QLabel("H:"), 1, 2); grid_roi.addWidget(self.sb_roi_h, 1, 3)
        l_status.addLayout(grid_roi)
        grp_status.setLayout(l_status)
        
        grp_timing = QGroupBox("Tiempos (Global)")
        l_timing = QVBoxLayout()
        l_timing.setSpacing(10)
        l_timing.addWidget(QLabel("Acumulación (µs) [Física]"))
        row_accum = QHBoxLayout()
        self.sl_accum = QSlider(Qt.Orientation.Horizontal)
        self.sl_accum.setRange(200, 50000); self.sl_accum.setValue(10000)
        self.sb_accum = QSpinBox()
        self.sb_accum.setRange(200, 100000); self.sb_accum.setValue(10000); self.sb_accum.setFixedWidth(100)
        self.sl_accum.valueChanged.connect(self.sb_accum.setValue)
        self.sb_accum.valueChanged.connect(self.sl_accum.setValue)
        self.sl_accum.valueChanged.connect(self.on_timing_change)
        row_accum.addWidget(self.sl_accum); row_accum.addWidget(self.sb_accum)
        l_timing.addLayout(row_accum)
        l_timing.addWidget(QLabel("FPS Visualizador [Pantalla]"))
        row_fps = QHBoxLayout()
        self.sl_fps = QSlider(Qt.Orientation.Horizontal)
        self.sl_fps.setRange(10, 165); self.sl_fps.setValue(60)
        self.sb_fps = QSpinBox()
        self.sb_fps.setRange(1, 200); self.sb_fps.setValue(60); self.sb_fps.setFixedWidth(100)
        self.sl_fps.valueChanged.connect(self.sb_fps.setValue)
        self.sb_fps.valueChanged.connect(self.sl_fps.setValue)
        self.sl_fps.valueChanged.connect(self.on_timing_change)
        row_fps.addWidget(self.sl_fps); row_fps.addWidget(self.sb_fps)
        l_timing.addLayout(row_fps)
        grp_timing.setLayout(l_timing)

        grp_ctrl = QGroupBox("Acciones (Aplica a ambas)")
        l_ctrl = QVBoxLayout()
        self.btn_stop = QPushButton("■ DETENER TODAS")
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_stop.setEnabled(False)
        self.btn_rec = QPushButton("● GRABAR RAW OFICIAL")
        self.btn_rec.clicked.connect(self.toggle_recording)
        self.btn_rec.setEnabled(False)
        self.btn_rec.setStyleSheet("color: #ff5555; font-weight: bold; border: 1px solid #555;")
        self.chk_roi = QCheckBox("🎯 Activar ROI Global")
        self.chk_roi.toggled.connect(self.toggle_roi)
        self.chk_roi.setStyleSheet("color: #ff5555; font-weight: bold;")
        self.chk_raw = QCheckBox("Mostrar Log")
        self.chk_raw.toggled.connect(self.toggle_raw_data)
        l_ctrl.addWidget(self.btn_stop)
        l_ctrl.addWidget(self.btn_rec)
        l_ctrl.addWidget(self.chk_roi)
        l_ctrl.addWidget(self.chk_raw)
        grp_ctrl.setLayout(l_ctrl)
        
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        
        layout.addWidget(grp_status)
        layout.addWidget(grp_timing)
        layout.addWidget(grp_ctrl)
        layout.addWidget(self.txt_log)
        return container

    def create_roi_spinbox(self, min_val, max_val, tooltip):
        sb = QSpinBox()
        sb.setRange(min_val, max_val)
        sb.setFixedWidth(80)
        sb.setStyleSheet("background-color: #333; color: #fff;")
        sb.setToolTip(f"Coordenada {tooltip} del sensor")
        sb.valueChanged.connect(self.on_manual_roi_input)
        return sb

    def on_bias_change(self, name, value):
        if self.worker_1: self.worker_1.update_bias(name, value)
        if self.worker_2: self.worker_2.update_bias(name, value)

    def toggle_roi(self, checked):
        self.lbl_video_1.roi_active = checked
        self.lbl_video_2.roi_active = checked
        self.lbl_video_1.update() 
        self.lbl_video_2.update() 
        for worker, visor in [(self.worker_1, self.lbl_video_1), (self.worker_2, self.lbl_video_2)]:
            if worker:
                worker.roi_enabled = checked
                if not visor.selection_rect: worker.roi_coords = (0, 0, 640, 480)
                else: visor.calculate_sensor_roi()

    def on_roi_changed(self, roi_tuple):
        x, y, w, h = roi_tuple
        self.sb_roi_x.blockSignals(True)
        self.sb_roi_y.blockSignals(True)
        self.sb_roi_w.blockSignals(True)
        self.sb_roi_h.blockSignals(True)
        self.sb_roi_x.setValue(x)
        self.sb_roi_y.setValue(y)
        self.sb_roi_w.setValue(w)
        self.sb_roi_h.setValue(h)
        self.sb_roi_x.blockSignals(False)
        self.sb_roi_y.blockSignals(False)
        self.sb_roi_w.blockSignals(False)
        self.sb_roi_h.blockSignals(False)
        self.lbl_status.setText("ROI Global Actualizado por Mouse")
        if self.worker_1: self.worker_1.roi_coords = roi_tuple
        if self.worker_2: self.worker_2.roi_coords = roi_tuple

    def on_manual_roi_input(self):
        x = self.sb_roi_x.value()
        y = self.sb_roi_y.value()
        w = self.sb_roi_w.value()
        h = self.sb_roi_h.value()
        if self.worker_1: self.worker_1.roi_coords = (x, y, w, h)
        if self.worker_2: self.worker_2.roi_coords = (x, y, w, h)
        self.lbl_video_1.set_visual_roi_from_coords(x, y, w, h)
        self.lbl_video_2.set_visual_roi_from_coords(x, y, w, h)
        self.lbl_status.setText(f"ROI Manual Global: {x},{y},{w},{h}")

    def update_image(self, cam_id, qt_img):
        pixmap = QPixmap.fromImage(qt_img)
        if cam_id == 1: self.lbl_video_1.setPixmap(pixmap.scaled(self.lbl_video_1.size(), Qt.AspectRatioMode.KeepAspectRatio))
        elif cam_id == 2: self.lbl_video_2.setPixmap(pixmap.scaled(self.lbl_video_2.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def update_stats(self, cam_id, debug_text, rate):
        if cam_id == 1: self.lbl_perf_1.setText(f"Tasa: {rate:.2f} MEv/s")
        elif cam_id == 2: self.lbl_perf_2.setText(f"Tasa: {rate:.2f} MEv/s")
        if debug_text: self.txt_log.setPlainText(f"--- CAM {cam_id} ---\n{debug_text}")

    def start_camera_1(self): self._start_camera(1)
    def start_camera_2(self): self._start_camera(2)

    def _start_camera(self, cam_id):
        self.btn_stop.setEnabled(True)
        self.btn_stop.setStyleSheet("background-color: #cf222e;")
        self.btn_rec.setEnabled(True)
        self.lbl_status.setText("● Adquisición Activa")
        self.lbl_status.setStyleSheet("color: #0f0; font-weight: bold;")
        
        worker = CameraWorker(use_simulation=not HAS_METAVISION)
        if cam_id == 1:
            worker.serial_number = self.txt_serial_1.text().strip()
            worker.sync_mode = ["STANDALONE", "MASTER", "SLAVE"][self.cmb_sync_1.currentIndex()]
            worker.image_signal.connect(lambda img, cid=cam_id: self.update_image(cid, img))
            worker.stats_signal.connect(lambda txt, rate, cid=cam_id: self.update_stats(cid, txt, rate))
            self.lbl_status_1.setText("Estado: Corriendo")
            self.lbl_status_1.setStyleSheet("color: #0f0;")
            self.btn_start_1.setEnabled(False)
            self.worker_1 = worker
            visor_activo = self.lbl_video_1
        else:
            worker.serial_number = self.txt_serial_2.text().strip()
            worker.sync_mode = ["STANDALONE", "MASTER", "SLAVE"][self.cmb_sync_2.currentIndex()]
            worker.image_signal.connect(lambda img, cid=cam_id: self.update_image(cid, img))
            worker.stats_signal.connect(lambda txt, rate, cid=cam_id: self.update_stats(cid, txt, rate))
            self.lbl_status_2.setText("Estado: Corriendo")
            self.lbl_status_2.setStyleSheet("color: #0f0;")
            self.btn_start_2.setEnabled(False)
            self.worker_2 = worker
            visor_activo = self.lbl_video_2

        worker.current_biases['bias_diff'] = self.sl_diff.value()
        worker.current_biases['bias_diff_on'] = self.sl_diff_on.value()
        worker.current_biases['bias_diff_off'] = self.sl_diff_off.value()
        worker.current_biases['bias_refr'] = self.sl_refr.value()
        worker.update_timing_params(self.sl_accum.value(), self.sl_fps.value())
        worker.show_raw_data = self.chk_raw.isChecked()
        worker.roi_enabled = self.chk_roi.isChecked()
        
        if visor_activo.selection_rect: visor_activo.calculate_sensor_roi()
        else: worker.roi_coords = (0,0,640,480)
        
        worker.start()

    def stop_camera(self):
        if self.worker_1: self.worker_1.stop(); self.worker_1 = None
        if self.worker_2: self.worker_2.stop(); self.worker_2 = None
        
        self.btn_start_1.setEnabled(True); self.lbl_status_1.setText("Estado: Detenida"); self.lbl_status_1.setStyleSheet("color: #ccc;")
        self.btn_start_2.setEnabled(True); self.lbl_status_2.setText("Estado: Detenida"); self.lbl_status_2.setStyleSheet("color: #ccc;")
        self.btn_stop.setEnabled(False); self.btn_stop.setStyleSheet("background-color: #555;")
        self.btn_rec.setEnabled(False); self.lbl_status.setText("● Sistema Detenido"); self.lbl_status.setStyleSheet("color: #ccc;")

    def toggle_recording(self):
        is_recording = (self.worker_1 and getattr(self.worker_1, 'recording', False)) or (self.worker_2 and getattr(self.worker_2, 'recording', False))
        
        if not is_recording:
            timestamp = time.strftime("%Y%m%d_%H%M") # Formato pedido
            
            # Creación dinámica de carpetas
            cam_dir = os.path.join("Grabaciones EvCam-IMU-GPS", "Grabaciones_RAW_EvCam")
            os.makedirs(cam_dir, exist_ok=True)

            self.lbl_status.setText("● GRABANDO (Formato EVT3 Oficial)")
            self.btn_rec.setText("■ DETENER GRABACIÓN")
            self.btn_rec.setStyleSheet("background-color: #ff0000; color: white;")
            
            if self.worker_1: self.worker_1.start_recording(os.path.join(cam_dir, f"EvCam1_{timestamp}.raw"))
            if self.worker_2: self.worker_2.start_recording(os.path.join(cam_dir, f"EvCam2_{timestamp}.raw"))
        else:
            if self.worker_1: self.worker_1.stop_recording()
            if self.worker_2: self.worker_2.stop_recording()
            self.btn_rec.setText("● GRABAR RAW OFICIAL")
            self.btn_rec.setStyleSheet("color: #ff5555; font-weight: bold; border: 1px solid #555;")
            self.lbl_status.setText("● Grabación Finalizada (Compatible Metavision)")

    def toggle_raw_data(self, checked):
        if self.worker_1: self.worker_1.show_raw_data = checked
        if self.worker_2: self.worker_2.show_raw_data = checked

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()
    
    def on_timing_change(self):
        accum_us = self.sl_accum.value()
        fps = self.sl_fps.value()
        if self.worker_1: self.worker_1.update_timing_params(accum_us, fps)
        if self.worker_2: self.worker_2.update_timing_params(accum_us, fps)
        self.lbl_status.setText(f"Tiempos: {accum_us}µs / {fps} FPS")

    # ========================================================
    # MÉTODOS DE LA PESTAÑA 1.5 (ENFOQUE / RECONSTRUCCIÓN)
    # ========================================================
    def create_recon_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Visor
        visor_layout = QVBoxLayout()
        self.lbl_recon_video = QLabel("Reconstrucción en vivo (Presione Iniciar)")
        self.lbl_recon_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_recon_video.setStyleSheet("background-color: #000; border: 2px solid #555; color: #fff; font-size: 16px;")
        self.lbl_recon_video.setMinimumSize(640, 480)
        visor_layout.addWidget(self.lbl_recon_video)
        
        # Controles laterales
        ctrl_container = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_container)
        ctrl_container.setFixedWidth(320)
        
        grp_cam = QGroupBox("Cámara")
        l_cam = QVBoxLayout()
        self.txt_serial_recon = QLineEdit()
        self.txt_serial_recon.setPlaceholderText("N° Serie (Vacío=Auto)")
        self.btn_recon_start = QPushButton("▶ INICIAR ENFOQUE")
        self.btn_recon_start.setStyleSheet("background-color: #2da44e;")
        self.btn_recon_start.clicked.connect(self.toggle_recon)
        l_cam.addWidget(QLabel("Puerto / N° Serie:"))
        l_cam.addWidget(self.txt_serial_recon)
        l_cam.addWidget(self.btn_recon_start)
        grp_cam.setLayout(l_cam)
        
        grp_math = QGroupBox("Física del Integrador")
        l_math = QVBoxLayout()
        
        # Slider Tau (0.001s a 1.000s) mapeado de 1 a 1000
        l_math.addWidget(QLabel("Tau (Memoria / Estela) [s]:"))
        self.sl_tau = QSlider(Qt.Orientation.Horizontal)
        self.sl_tau.setRange(1, 1000); self.sl_tau.setValue(100) # 0.1s
        self.lbl_tau_val = QLabel("0.100 s")
        row_tau = QHBoxLayout(); row_tau.addWidget(self.sl_tau); row_tau.addWidget(self.lbl_tau_val)
        l_math.addLayout(row_tau)
        
        # Slider C_on (0.01 a 1.0) mapeado de 1 a 100
        l_math.addWidget(QLabel("Salto C_on (Luz):"))
        self.sl_con = QSlider(Qt.Orientation.Horizontal)
        self.sl_con.setRange(1, 100); self.sl_con.setValue(15) # 0.15
        self.lbl_con_val = QLabel("0.15")
        row_con = QHBoxLayout(); row_con.addWidget(self.sl_con); row_con.addWidget(self.lbl_con_val)
        l_math.addLayout(row_con)
        
        # Slider C_off (0.01 a 1.0) mapeado de 1 a 100
        l_math.addWidget(QLabel("Salto C_off (Sombra):"))
        self.sl_coff = QSlider(Qt.Orientation.Horizontal)
        self.sl_coff.setRange(1, 100); self.sl_coff.setValue(15) # 0.15
        self.lbl_coff_val = QLabel("0.15")
        row_coff = QHBoxLayout(); row_coff.addWidget(self.sl_coff); row_coff.addWidget(self.lbl_coff_val)
        l_math.addLayout(row_coff)
        
        self.sl_tau.valueChanged.connect(self.update_recon_params)
        self.sl_con.valueChanged.connect(self.update_recon_params)
        self.sl_coff.valueChanged.connect(self.update_recon_params)
        grp_math.setLayout(l_math)
        
        self.txt_recon_log = QTextEdit()
        self.txt_recon_log.setReadOnly(True)
        
        ctrl_layout.addWidget(grp_cam)
        ctrl_layout.addWidget(grp_math)
        ctrl_layout.addWidget(QLabel("Monitor de Integración:"))
        ctrl_layout.addWidget(self.txt_recon_log)
        
        layout.addLayout(visor_layout, 70)
        layout.addWidget(ctrl_container, 30)
        
        self.recon_worker = None
        return tab

    def update_recon_params(self):
        tau = self.sl_tau.value() / 1000.0
        con = self.sl_con.value() / 100.0
        coff = self.sl_coff.value() / 100.0
        
        self.lbl_tau_val.setText(f"{tau:.3f} s")
        self.lbl_con_val.setText(f"{con:.2f}")
        self.lbl_coff_val.setText(f"{coff:.2f}")
        
        if self.recon_worker:
            self.recon_worker.tau = tau
            self.recon_worker.c_on = con
            self.recon_worker.c_off = coff

    def toggle_recon(self):
        if self.recon_worker is None:
            self.recon_worker = ReconWorker()
            self.recon_worker.serial_number = self.txt_serial_recon.text().strip()
            self.update_recon_params() # Inyecta los valores actuales
            
            self.recon_worker.image_signal.connect(self.update_recon_image)
            self.recon_worker.stats_signal.connect(self.update_recon_stats)
            
            self.recon_worker.start()
            self.btn_recon_start.setText("■ DETENER ENFOQUE")
            self.btn_recon_start.setStyleSheet("background-color: #cf222e;")
        else:
            self.recon_worker.stop()
            self.recon_worker = None
            self.btn_recon_start.setText("▶ INICIAR ENFOQUE")
            self.btn_recon_start.setStyleSheet("background-color: #2da44e;")

    def update_recon_image(self, qt_img):
        pixmap = QPixmap.fromImage(qt_img)
        self.lbl_recon_video.setPixmap(pixmap.scaled(self.lbl_recon_video.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def update_recon_stats(self, text, rate):
        self.txt_recon_log.setPlainText(text + f"\nTasa: {rate:.2f} MEv/s")

    # ========================================================
    # MÉTODOS DE LA PESTAÑA TOPOGRAFÍA Y ENLACE
    # ========================================================
    
    def create_topo_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # --- PANEL IZQUIERDO (CONTROLES) ---
        ctrl_container = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_container)
        ctrl_container.setFixedWidth(350)

        grp_file = QGroupBox("1. Modelo de Elevación Digital (DEM)")
        l_file = QVBoxLayout()
        self.btn_load_tif = QPushButton("📂 CARGAR ARCHIVO .TIF")
        self.btn_load_tif.clicked.connect(self.load_tif_file)
        self.btn_load_tif.setStyleSheet("background-color: #d1861a; font-weight: bold; padding: 8px;")
        self.lbl_tif_name = QLabel("Ningún archivo cargado")
        self.lbl_tif_name.setStyleSheet("color: #aaa; font-style: italic;")
        self.lbl_tif_name.setWordWrap(True)
        l_file.addWidget(self.btn_load_tif)
        l_file.addWidget(self.lbl_tif_name)
        grp_file.setLayout(l_file)

        grp_coords = QGroupBox("2. Coordenadas del Enlace Geodésico")
        l_coords = QVBoxLayout()

        # Estación 1 (Cámara)
        l_coords.addWidget(QLabel("📍 Estación 1 (Origen):"))
        self.txt_name_a = QLineEdit("Estación Cámara")
        self.txt_name_a.setStyleSheet("background-color: #222; color: #0f0;")
        l_a = QHBoxLayout()
        self.txt_lat_a = QLineEdit("-33.000000")
        self.txt_lon_a = QLineEdit("-71.000000")
        l_a.addWidget(QLabel("Lat:")); l_a.addWidget(self.txt_lat_a)
        l_a.addWidget(QLabel("Lon:")); l_a.addWidget(self.txt_lon_a)
        l_coords.addWidget(self.txt_name_a)
        l_coords.addLayout(l_a)

        l_coords.addSpacing(10)

        # Estación 2 (Target)
        l_coords.addWidget(QLabel("🎯 Estación 2 (Destino):"))
        self.txt_name_b = QLineEdit("Estación Target")
        self.txt_name_b.setStyleSheet("background-color: #222; color: #00aaaa;")
        l_b = QHBoxLayout()
        self.txt_lat_b = QLineEdit("-33.100000")
        self.txt_lon_b = QLineEdit("-71.100000")
        l_b.addWidget(QLabel("Lat:")); l_b.addWidget(self.txt_lat_b)
        l_b.addWidget(QLabel("Lon:")); l_b.addWidget(self.txt_lon_b)
        l_coords.addWidget(self.txt_name_b)
        l_coords.addLayout(l_b)

        self.btn_calc_topo = QPushButton("Generar Perfil y Mapa de Sombras")
        self.btn_calc_topo.clicked.connect(self.process_topography)
        self.btn_calc_topo.setStyleSheet("background-color: #2da44e; font-weight: bold; padding: 10px; margin-top: 5px;")
        l_coords.addWidget(self.btn_calc_topo)
        grp_coords.setLayout(l_coords)

        self.txt_topo_log = QTextEdit()
        self.txt_topo_log.setReadOnly(True)
        self.txt_topo_log.setStyleSheet("background-color: #050505; color: #ddd; font-family: Consolas;")

        ctrl_layout.addWidget(grp_file)
        ctrl_layout.addWidget(grp_coords)
        ctrl_layout.addWidget(QLabel("Registro Topográfico:"))
        ctrl_layout.addWidget(self.txt_topo_log)

        # --- PANEL DERECHO (VISUALIZACIÓN MATPLOTLIB) ---
        visor_layout = QVBoxLayout()
        self.fig_topo, (self.ax_prof, self.ax_map) = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={'height_ratios': [1, 1.8]})
        self.fig_topo.patch.set_facecolor('#121212') 
        self.canvas_topo = FigureCanvas(self.fig_topo)
        self.toolbar_topo = NavigationToolbar(self.canvas_topo, tab)

        visor_layout.addWidget(self.toolbar_topo)
        visor_layout.addWidget(self.canvas_topo)

        layout.addWidget(ctrl_container, 25)
        layout.addLayout(visor_layout, 75)

        self.dem_path = None
        return tab

    def load_tif_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Abrir Modelo de Elevación Digital", "", "GeoTIFF (*.tif *.tiff);;All Files (*)")
        if file_name:
            self.dem_path = file_name
            self.lbl_tif_name.setText(file_name.split("/")[-1])
            self.lbl_tif_name.setStyleSheet("color: #0f0; font-weight: bold;")
            self.txt_topo_log.append(f"✓ DEM Cargado en memoria.")

    def process_topography(self):
        if not HAS_GEOSPATIAL:
            self.txt_topo_log.append("❌ Error Crítico: Librerías geoespaciales no detectadas.")
            return
        if not self.dem_path:
            self.txt_topo_log.append("⚠ Advertencia: Debes cargar un archivo .tif primero.")
            return

        try:
            self.txt_topo_log.append("\nCalculando matrices espaciales...")
            QApplication.processEvents()

            lat1, lon1 = float(self.txt_lat_a.text()), float(self.txt_lon_a.text())
            lat2, lon2 = float(self.txt_lat_b.text()), float(self.txt_lon_b.text())
            name1, name2 = self.txt_name_a.text(), self.txt_name_b.text()

            # 1. Distancia Geodésica Real
            geod = Geod(ellps="WGS84")
            _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
            dist_km = dist_m / 1000.0

            # 2. Extracción Numérica con Rasterio
            with rasterio.open(self.dem_path) as src:
                transformer_to_px = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x1, y1 = transformer_to_px.transform(lon1, lat1)
                x2, y2 = transformer_to_px.transform(lon2, lat2)

                # -- A) Perfil Topográfico 1D --
                line = LineString([(x1, y1), (x2, y2)])
                dists = np.linspace(0, line.length, 500)
                points = [line.interpolate(d) for d in dists]
                coords = [(p.x, p.y) for p in points]
                samples = list(src.sample(coords))
                elevs = np.array([s[0] if s[0] != src.nodata else np.nan for s in samples])

                # Limpieza de NaNs (Interpolación lineal)
                nans = np.isnan(elevs)
                if np.any(nans):
                    x_idx = lambda z: z.nonzero()[0]
                    elevs[nans] = np.interp(x_idx(nans), x_idx(~nans), elevs[~nans])

                real_profile_dist = np.linspace(0, dist_km, 500)

                # -- B) Recorte Espacial 2D para Hillshade --
                res_x = abs(src.transform.a)
                res_y = abs(src.transform.e)

                margin_x = max(abs(x1 - x2) * 0.4, 150 * res_x)
                margin_y = max(abs(y1 - y2) * 0.4, 150 * res_y)

                min_x, max_x = min(x1, x2) - margin_x, max(x1, x2) + margin_x
                min_y, max_y = min(y1, y2) - margin_y, max(y1, y2) + margin_y

                window = from_bounds(min_x, min_y, max_x, max_y, transform=src.transform)
                dem_window = src.read(1, window=window)
                win_transform = src.window_transform(window)

                # Coordenadas de los bordes para ubicar el imshow
                extent = [
                    win_transform.c,
                    win_transform.c + dem_window.shape[1] * win_transform.a,
                    win_transform.f + dem_window.shape[0] * win_transform.e,
                    win_transform.f
                ]

            # 3. Renderizado Matplotlib
            self.ax_prof.clear()
            self.ax_map.clear()

            # --- Gráfico 1: Perfil de Elevación ---
            self.ax_prof.set_facecolor('#1a1a1a')
            self.ax_prof.fill_between(real_profile_dist, 0, elevs, color='#4E342E', alpha=0.9)
            self.ax_prof.plot(real_profile_dist, elevs, color='#A1887F', lw=1)
            # Línea de vista (Ideal)
            self.ax_prof.plot([0, dist_km], [elevs[0], elevs[-1]], color='#00ffaa', lw=1.5, linestyle='--', label='Línea de Vista (LOS)')

            self.ax_prof.set_title(f"Perfil de Elevación - Distancia: {dist_km:.2f} km", color='white')
            self.ax_prof.set_ylabel("Altitud (m)", color='white')
            self.ax_prof.tick_params(colors='gray')
            
            # Etiquetas de puntos en el perfil
            box_style = dict(boxstyle="round,pad=0.3", fc="#222", ec="#555", alpha=0.9)
            self.ax_prof.annotate(f"{name1}\n{elevs[0]:.0f}m", xy=(0, elevs[0]), xytext=(0, 20), textcoords='offset points', ha='center', color='#0f0', bbox=box_style, fontweight='bold')
            self.ax_prof.annotate(f"{name2}\n{elevs[-1]:.0f}m", xy=(dist_km, elevs[-1]), xytext=(0, 20), textcoords='offset points', ha='center', color='#00aaaa', bbox=box_style, fontweight='bold')
            
            self.ax_prof.set_ylim(max(0, np.min(elevs) - 100), np.max(elevs) + (np.max(elevs)-np.min(elevs))*0.5)
            self.ax_prof.grid(True, alpha=0.2, color='gray')
            self.ax_prof.legend(facecolor='#1a1a1a', edgecolor='none', labelcolor='white')

            # --- Gráfico 2: Hillshade (Mapa de sombras) ---
            ls = LightSource(azdeg=315, altdeg=45)
            dem_plot = dem_window.copy().astype(float)
            if src.nodata is not None:
                dem_plot[dem_plot == src.nodata] = np.nan
            
            # Algoritmo de renderizado de sombras basado en la matriz
            shade = ls.hillshade(dem_plot, vert_exag=1.5, dx=abs(win_transform.a), dy=abs(win_transform.e))

            self.ax_map.set_facecolor('#1a1a1a')
            self.ax_map.imshow(shade, cmap='gray', extent=extent, origin='upper')

            # Línea trazada sobre el mapa
            self.ax_map.plot([x1, x2], [y1, y2], color='#ff3333', lw=2.5)
            self.ax_map.plot(x1, y1, marker='o', color='#0f0', markersize=8, markeredgecolor='black')
            self.ax_map.plot(x2, y2, marker='o', color='#00aaaa', markersize=8, markeredgecolor='black')

            # Textos con contorno (PathEffects) para que destaquen sobre la topografía
            txt_eff = [pe.withStroke(linewidth=3, foreground="black")]
            self.ax_map.text(x1, y1, f"  {name1}", color='#0f0', fontweight='bold', fontsize=11, ha='left', va='center', path_effects=txt_eff)
            self.ax_map.text(x2, y2, f"  {name2}", color='#00aaaa', fontweight='bold', fontsize=11, ha='right', va='center', path_effects=txt_eff)

            self.ax_map.set_title("Proyección Topográfica A-B (SRTM)", color='white')
            self.ax_map.set_xlabel("Coordenada X (CRS)", color='gray')
            self.ax_map.set_ylabel("Coordenada Y (CRS)", color='gray')
            self.ax_map.tick_params(colors='gray')

            self.fig_topo.tight_layout()
            self.canvas_topo.draw()

            self.txt_topo_log.append(f"✓ Renderizado completado exitosamente.")

        except Exception as e:
            self.txt_topo_log.append(f"❌ Error al procesar DEM: {str(e)}")

    # ========================================================
    # MÉTODOS DE LA PESTAÑA IMU & GPS (ESP32)
    # ========================================================
    def create_imu_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        grp_conn = QGroupBox("1. Conexión de Hardware (ESP32)")
        l_conn = QHBoxLayout()
        l_conn.addWidget(QLabel("Puerto COM:"))
        self.cmb_ports = QComboBox()
        self.cmb_ports.setFixedWidth(200)
        self.cmb_ports.setStyleSheet("background-color: #333; color: #fff;")
        self.btn_refresh_ports = QPushButton("⟳")
        self.btn_refresh_ports.setFixedWidth(40)
        self.btn_refresh_ports.clicked.connect(self.refresh_imu_ports)
        l_conn.addWidget(self.cmb_ports)
        l_conn.addWidget(self.btn_refresh_ports)
        l_conn.addWidget(QLabel("Baudios:"))
        self.cmb_baud = QComboBox()
        self.cmb_baud.addItems(["921600", "115200", "2000000"])
        self.cmb_baud.setFixedWidth(100)
        self.cmb_baud.setStyleSheet("background-color: #333; color: #fff;")
        l_conn.addWidget(self.cmb_baud)
        self.btn_connect_imu = QPushButton("CONECTAR ESP32")
        self.btn_connect_imu.setStyleSheet("background-color: #00aaaa; color: black;")
        self.btn_connect_imu.clicked.connect(self.toggle_imu_connection)
        l_conn.addWidget(self.btn_connect_imu)
        l_conn.addStretch()
        grp_conn.setLayout(l_conn)
        
        grp_stats = QGroupBox("2. Rendimiento (Tiempo Real)")
        l_stats = QHBoxLayout()
        self.lbl_imu_hz = QLabel("0.0 Hz")
        self.lbl_imu_hz.setStyleSheet("font-family: Consolas; font-size: 40px; color: #00ff00; font-weight: bold;")
        stats_col = QVBoxLayout()
        self.lbl_imu_bps = QLabel("Bytes/s: 0")
        self.lbl_imu_bps.setStyleSheet("font-family: Consolas; font-size: 14px; color: #ccc;")
        self.lbl_imu_err = QLabel("Errores Sincronización: 0")
        self.lbl_imu_err.setStyleSheet("font-family: Consolas; font-size: 14px; color: #ff5555;")
        
        # --- NUEVO: MONITOR DE PULSOS PPS EN LA GUI ---
        self.lbl_imu_pps = QLabel("⏱️ Pulsos Atómicos (PPS): 0")
        self.lbl_imu_pps.setStyleSheet("font-family: Consolas; font-size: 15px; color: #ffcc00; font-weight: bold;")
        
        stats_col.addWidget(self.lbl_imu_bps)
        stats_col.addWidget(self.lbl_imu_err)
        stats_col.addWidget(self.lbl_imu_pps) # <-- SE AGREGA AL LAYOUT
        l_stats.addWidget(self.lbl_imu_hz)
        l_stats.addSpacing(30)
        l_stats.addLayout(stats_col)
        l_stats.addStretch()
        grp_stats.setLayout(l_stats)
        
        # --- NUEVO: SECCIÓN DE POSICIÓN ESPACIAL ---
        grp_gps = QGroupBox("3. Referencia Espacial (GPS)")
        l_gps = QHBoxLayout()
        self.btn_get_coord = QPushButton("📍 FIJAR COORDENADA ACTUAL")
        self.btn_get_coord.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_get_coord.clicked.connect(self.fix_coordinate)
        self.lbl_gps_coord = QLabel("Latitud: --.----  |  Longitud: --.----")
        self.lbl_gps_coord.setStyleSheet("font-family: Consolas; font-size: 16px; color: #ffd700; font-weight: bold; padding-left: 15px;")
        l_gps.addWidget(self.btn_get_coord)
        l_gps.addWidget(self.lbl_gps_coord)
        l_gps.addStretch()
        grp_gps.setLayout(l_gps)
        # -------------------------------------------

        grp_rec = QGroupBox("4. Adquisición de Datos (Máxima Velocidad)")
        l_rec = QHBoxLayout()
        self.chk_imu_raw = QCheckBox("Modo RAW (Volcado directo sin procesar)")
        self.chk_imu_raw.setStyleSheet("color: #ffaa00;")
        self.btn_rec_imu = QPushButton("● GRABAR BINARIO (.bin)")
        self.btn_rec_imu.setStyleSheet("background-color: #444; color: #888;")
        self.btn_rec_imu.setEnabled(False)
        self.btn_rec_imu.clicked.connect(self.toggle_imu_recording)
        l_rec.addWidget(self.chk_imu_raw)
        l_rec.addStretch()
        l_rec.addWidget(self.btn_rec_imu)
        grp_rec.setLayout(l_rec)
        
        self.txt_imu_console = QTextEdit()
        self.txt_imu_console.setReadOnly(True)
        self.txt_imu_console.setStyleSheet("background-color: #050505; color: #0f0; font-family: Consolas; font-size: 13px;")
        self.txt_imu_console.append("Módulo IMU/GPS inicializado. Listo para conectar.")
        
        layout.addWidget(grp_conn)
        layout.addWidget(grp_stats)
        layout.addWidget(grp_gps) # Agregamos el nuevo GroupBox
        layout.addWidget(grp_rec)
        layout.addWidget(QLabel("Registro del Sistema:"))
        layout.addWidget(self.txt_imu_console, stretch=1)
        
        self.refresh_imu_ports() 
        return tab

    def fix_coordinate(self):
        """Bloquea en pantalla la última coordenada recibida silenciosamente desde el ESP32"""
        if self.latest_lat is not None and self.latest_lon is not None:
            self.lbl_gps_coord.setText(f"Latitud: {self.latest_lat:.6f}  |  Longitud: {self.latest_lon:.6f}")
            self.log_imu(f"📍 Marcador Espacial Fijado: Lat {self.latest_lat:.6f}, Lon {self.latest_lon:.6f}")
        else:
            self.lbl_gps_coord.setText("Esperando señal del ESP32...")
            self.log_imu("⚠ Aún no se han recibido paquetes 0xBBDD desde el hardware.")

    def store_incoming_coord(self, lat, lon):
        """Callback silencioso que guarda el dato cada vez que el ESP32 lo emite"""
        self.latest_lat = lat
        self.latest_lon = lon

    def update_realtime_pps(self, count):
        self.lbl_imu_pps.setText(f"⏱️ Pulsos Atómicos (PPS): {count}")
        # Efecto de parpadeo visual entre verde y amarillo
        if count % 2 == 0:
            self.lbl_imu_pps.setStyleSheet("font-family: Consolas; font-size: 15px; color: #00ffaa; font-weight: bold;")
        else:
            self.lbl_imu_pps.setStyleSheet("font-family: Consolas; font-size: 15px; color: #ffcc00; font-weight: bold;")

    def refresh_imu_ports(self):
        self.cmb_ports.setEditable(True)
        self.cmb_ports.clear()
        self.cmb_ports.addItem("Buscando...")
        QApplication.processEvents()
        time.sleep(0.1)
        self.cmb_ports.clear()
        try:
            import serial.tools.list_ports 
            puertos_brutos = serial.tools.list_ports.comports()
            raw_log = [f"{p.device} ({p.description})" for p in puertos_brutos]
            if raw_log: self.log_imu(f"🔍 USB crudos detectados por Windows: {raw_log}")
            ports = [p.device for p in puertos_brutos]
            if ports:
                self.cmb_ports.addItems(ports)
                self.log_imu(f"✓ Escaneo COM exitoso: {ports}")
            else:
                self.cmb_ports.addItem("COM3") 
                self.log_imu("⚠ Windows reporta 0 dispositivos. ESCRIBE TU PUERTO A MANO (ej: COM3 o COM4).")
        except Exception as e:
            self.cmb_ports.addItem("COM3")
            self.log_imu(f"❌ ERROR del Sistema Operativo: {str(e)}")

    def toggle_imu_connection(self):
        if self.imu_worker is None:
            port = self.cmb_ports.currentText()
            if port == "No se detectan placas" or port == "": return
            baud = int(self.cmb_baud.currentText())
            self.imu_worker = IMUWorker(port, baud)
            self.imu_worker.log_signal.connect(self.log_imu)
            self.imu_worker.stats_signal.connect(self.update_imu_stats)
            self.imu_worker.coord_signal.connect(self.store_incoming_coord) 
            self.imu_worker.pps_signal.connect(self.update_realtime_pps) # <-- CONECTA LA SEÑAL DEL PPS
            self.imu_worker.start()
            self.btn_connect_imu.setText("DESCONECTAR ESP32")
            self.btn_connect_imu.setStyleSheet("background-color: #cf222e; color: white;")
            self.btn_rec_imu.setEnabled(True)
            self.btn_rec_imu.setStyleSheet("background-color: #00aaaa; color: black;")
            self.cmb_ports.setEnabled(False)
            self.cmb_baud.setEnabled(False)
        else:
            self.imu_worker.stop()
            self.imu_worker = None
            self.btn_connect_imu.setText("CONECTAR ESP32")
            self.btn_connect_imu.setStyleSheet("background-color: #00aaaa; color: black;")
            self.btn_rec_imu.setEnabled(False)
            self.btn_rec_imu.setStyleSheet("background-color: #444; color: #888;")
            self.btn_rec_imu.setText("● GRABAR BINARIO (.bin)")
            self.cmb_ports.setEnabled(True)
            self.cmb_baud.setEnabled(True)
            self.lbl_imu_hz.setText("0.0 Hz")
            self.lbl_imu_pps.setText("⏱️ Pulsos Atómicos (PPS): 0")
            self.lbl_imu_pps.setStyleSheet("font-family: Consolas; font-size: 15px; color: #ffcc00; font-weight: bold;")

    def toggle_imu_recording(self):
        if not self.imu_worker: return
        if not self.imu_worker.recording and not getattr(self.imu_worker, 'armed', False):
            timestamp = time.strftime("%Y%m%d_%H%M") # Formato pedido
            
            # Creación dinámica de carpetas
            imu_dir = os.path.join("Grabaciones EvCam-IMU-GPS", "Grabaciones_acelerometro")
            os.makedirs(imu_dir, exist_ok=True)
            
            filename = os.path.join(imu_dir, f"Ace_{timestamp}.bin")
            raw_mode = self.chk_imu_raw.isChecked()
            
            if self.imu_worker.start_recording(filename, raw_mode):
                self.btn_rec_imu.setText("■ DETENER GRABACIÓN")
                self.btn_rec_imu.setStyleSheet("background-color: #ff0000; color: white;")
                self.log_imu(f"[INI] Preparando archivo {os.path.basename(filename)} ...")
        else:
            self.imu_worker.stop_recording()
            self.btn_rec_imu.setText("● GRABAR BINARIO (.bin)")
            self.btn_rec_imu.setStyleSheet("background-color: #00aaaa; color: black;")
            self.log_imu("[STOP] Grabación IMU finalizada.")

    def update_imu_stats(self, hz, bps, errors):
        self.lbl_imu_hz.setText(f"{hz:.1f} Hz")
        self.lbl_imu_bps.setText(f"Bytes/s: {int(bps)}")
        self.lbl_imu_err.setText(f"Errores Sincronización: {errors}")

    def log_imu(self, text):
        self.txt_imu_console.append(text.strip())

    # ========================================================
    # MÉTODOS DE LA PESTAÑA DE ANÁLISIS Y CONVERSIÓN (IMU)
    # ========================================================
    def create_analysis_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        left_panel = QFrame()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        lbl_title = QLabel("Procesamiento de Datos")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #77d9d4;")
        
        self.btn_load_bin = QPushButton("📂 ABRIR BINARIO (.BIN)")
        self.btn_load_bin.setStyleSheet("background-color: #00aaaa; color: black; font-weight: bold; padding: 15px;")
        self.btn_load_bin.clicked.connect(self.load_bin_file)

        self.btn_export_csv = QPushButton("💾 EXPORTAR A CSV")
        self.btn_export_csv.setStyleSheet("background-color: #444; color: #888; font-weight: bold; padding: 15px;")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self.export_csv)

        self.txt_analysis_log = QTextEdit()
        self.txt_analysis_log.setReadOnly(True)
        self.txt_analysis_log.setStyleSheet("background-color: #050505; color: #ddd; font-family: Consolas;")
        self.txt_analysis_log.append("Listo para analizar archivos .bin del IMU.")

        left_layout.addWidget(lbl_title)
        left_layout.addSpacing(15)
        left_layout.addWidget(self.btn_load_bin)
        left_layout.addWidget(self.btn_export_csv)
        left_layout.addSpacing(15)
        left_layout.addWidget(QLabel("Registro de Operaciones:"))
        left_layout.addWidget(self.txt_analysis_log)

        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.lbl_plot_placeholder = QLabel("La visualización aparecerá aquí")
        self.lbl_plot_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_plot_placeholder.setStyleSheet("color: #666; font-size: 16px;")
        self.plot_layout.addWidget(self.lbl_plot_placeholder)

        layout.addWidget(left_panel)
        layout.addWidget(self.plot_container, stretch=1)

        self.imu_ts, self.imu_ax, self.imu_ay, self.imu_az = [], [], [], []
        self.imu_gx, self.imu_gy, self.imu_gz = [], [], []
        self.pps_ts = [] 
        self.csv_rows = [] 
        self.canvas = None
        self.toolbar = None

        return tab

    def load_bin_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Abrir archivo Binario IMU", "", "Bin Files (*.bin);;All Files (*)")
        if file_name:
            self.txt_analysis_log.append(f"\n--- Cargando: {os.path.basename(file_name)} ---")
            QApplication.processEvents() 
            self.decode_binary(file_name)

    def decode_binary(self, filepath):
        self.imu_ts.clear(); self.imu_ax.clear(); self.imu_ay.clear(); self.imu_az.clear()
        self.imu_gx.clear(); self.imu_gy.clear(); self.imu_gz.clear()
        self.pps_ts.clear()
        self.csv_rows.clear()
        self.root_atm_unix = None # Inicializamos vacío

        PACKET_SIZE = 18
        ACCEL_SCALE = 4096.0
        GYRO_SCALE = 65.5

        try:
            filesize = os.path.getsize(filepath)
            with open(filepath, "rb") as f:
                raw_data = f.read()

            self.txt_analysis_log.append(f"Tamaño archivo: {filesize/1024/1024:.2f} MB")
            
            packet_count = 0
            gps_count = 0
            errors = 0
            idx = 0
            t0 = None
            limit = len(raw_data) - PACKET_SIZE

            while idx <= limit:
                header = struct.unpack_from("<H", raw_data, idx)[0]
                
                if header == 0xBBAA:
                    try:
                        p = struct.unpack_from("<L hhhhhh", raw_data, idx + 2)
                        micros = p[0]
                        if t0 is None: t0 = micros

                        t_sec = (micros - t0) / 1e6
                        ax, ay, az = p[1]/ACCEL_SCALE, p[2]/ACCEL_SCALE, p[3]/ACCEL_SCALE
                        gx, gy, gz = p[4]/GYRO_SCALE, p[5]/GYRO_SCALE, p[6]/GYRO_SCALE

                        self.imu_ts.append(t_sec)
                        self.imu_ax.append(ax); self.imu_ay.append(ay); self.imu_az.append(az)
                        self.imu_gx.append(gx); self.imu_gy.append(gy); self.imu_gz.append(gz)
                        
                        self.csv_rows.append([t_sec, ax, ay, az, gx, gy, gz, 0])

                        packet_count += 1
                        idx += PACKET_SIZE
                    except:
                        errors += 1; idx += 1

                elif header == 0xBBCC: 
                    try:
                        p = struct.unpack_from("<L", raw_data, idx + 2)
                        micros = p[0]
                        if t0 is None: t0 = micros
                        
                        t_sec = (micros - t0) / 1e6
                        self.pps_ts.append(t_sec)
                        
                        self.csv_rows.append([t_sec, "", "", "", "", "", "", 1])
                        
                        gps_count += 1
                        idx += PACKET_SIZE
                    except:
                        errors += 1; idx += 1
                
                elif header == 0xBBDD:
                    idx += PACKET_SIZE
                    
                # --- DETECCIÓN DEL MARCADOR ABSOLUTO EN EL ANÁLISIS ---
                elif header == 0xFFEE:
                    try:
                        p = struct.unpack_from("<d", raw_data, idx + 2)
                        unix_ts = p[0]
                        import datetime
                        atm_time = datetime.datetime.fromtimestamp(unix_ts, tz=datetime.timezone.utc)
                        self.txt_analysis_log.append(f"⏱️ ATM (Sello Raíz): {atm_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                        
                        # GUARDAMOS EL ATM EN LA INSTANCIA PARA USARLO LUEGO
                        self.root_atm_unix = unix_ts 
                        idx += PACKET_SIZE
                    except:
                        errors += 1; idx += 1
                        
                else:
                    errors += 1; idx += 1

            if packet_count > 0 or gps_count > 0:
                duration = self.imu_ts[-1] if self.imu_ts else 0
                hz = packet_count / duration if duration > 0 else 0

                self.txt_analysis_log.append(f"✓ Paquetes IMU: {packet_count}")
                self.txt_analysis_log.append(f"📡 Pulsos GPS (PPS): {gps_count}")
                self.txt_analysis_log.append(f"⚠ Bytes Descartados: {errors}")
                self.txt_analysis_log.append(f"⏱ Duración: {duration:.2f} s")
                self.txt_analysis_log.append(f"⚡ Frecuencia IMU: {hz:.1f} Hz")

                self.btn_export_csv.setEnabled(True)
                self.btn_export_csv.setStyleSheet("background-color: #d1861a; color: white; font-weight: bold; padding: 15px;")
                self.plot_data()
            else:
                self.txt_analysis_log.append("❌ ERROR: No se encontraron datos válidos.")

        except Exception as e:
            self.txt_analysis_log.append(f"❌ Error crítico: {str(e)}")

    def plot_data(self):
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
        if self.toolbar:
            self.plot_layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
        self.lbl_plot_placeholder.hide()

        plt.style.use('dark_background')
        
        # GRÁFICOS (GridSpec) ---
        fig = plt.figure(figsize=(12, 6), dpi=100)
        gs = gridspec.GridSpec(2, 2, width_ratios=[2.5, 1]) 
        fig.patch.set_facecolor('#121212') 

        ax1 = fig.add_subplot(gs[0, 0]) # Acelerómetro (Izquierda Arriba)
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1) # Giroscopio (Izquierda Abajo)
        ax3 = fig.add_subplot(gs[:, 1]) # PPS Drift (Derecha Completa)

        step = 1
        if len(self.imu_ts) > 20000:
            step = int(len(self.imu_ts) / 10000)
            self.txt_analysis_log.append(f"Gráfico: Mostrando 1 de cada {step} puntos.")

        t = self.imu_ts[::step]

        # 1. Gráfico Acelerómetro
        ax1.set_facecolor('#1a1a1a')
        ax1.plot(t, self.imu_ax[::step], lw=0.8, color='#ff6b6b', label='Ax')
        ax1.plot(t, self.imu_ay[::step], lw=0.8, color='#4ecdc4', label='Ay')
        ax1.plot(t, self.imu_az[::step], lw=0.8, color='#f7fff7', label='Az')
        # Líneas verticales del PPS removidas para mayor claridad visual
        ax1.set_ylabel('Aceleración (g)', color='white')
        ax1.legend(loc='upper right', facecolor='#333333', edgecolor='none', labelcolor='white')
        ax1.grid(True, color='#444444', linestyle='--', alpha=0.3)

        # 2. Gráfico Giroscopio
        ax2.set_facecolor('#1a1a1a')
        ax2.plot(t, self.imu_gx[::step], lw=0.8, color='#ff9ff3', label='Gx')
        ax2.plot(t, self.imu_gy[::step], lw=0.8, color='#feca57', label='Gy')
        ax2.plot(t, self.imu_gz[::step], lw=0.8, color='#54a0ff', label='Gz')
        # Líneas verticales del PPS removidas para mayor claridad visual
        ax2.set_ylabel('Giroscopio (°/s)', color='white')
        ax2.set_xlabel('Tiempo de Adquisición (s)', color='white')
        ax2.legend(loc='upper right', facecolor='#333333', edgecolor='none', labelcolor='white')
        ax2.grid(True, color='#444444', linestyle='--', alpha=0.3)

        # 3. Gráfico de Deriva de Reloj (Clock Drift del PPS)
        ax3.set_facecolor('#1a1a1a')
        if len(self.pps_ts) > 1:
            # Calculamos Delta t entre pulsos
            intervals = np.diff(self.pps_ts)
            pps_x = self.pps_ts[:-1]
            
            # Gráfico de los intervalos medidos por el ESP32 vs el Tiempo
            ax3.plot(pps_x, intervals, marker='o', markersize=4, linestyle='-', color='#00ffaa', lw=1)
            # Línea ideal atómica de 1.000000 segundos
            ax3.axhline(y=1.0, color='red', linestyle='--', label='1.0s Ideal (Atómico)')
            
            mean_int = np.mean(intervals)
            std_int = np.std(intervals)
            
            ax3.set_title(f"Deriva (Clock Drift ESP32)\nMedia: {mean_int:.6f} s\nDesv: ±{std_int:.6f} s", color='white', fontsize=10)
            ax3.set_ylabel('Δt entre pulsos medido (s)', color='white')
            ax3.set_xlabel('Tiempo de Adquisición (s)', color='white')
            ax3.legend(loc='upper right', facecolor='#333333', edgecolor='none', labelcolor='white')
        else:
            # Si grabaste muy poco o no había señal GPS
            ax3.text(0.5, 0.5, "Datos GPS Insuficientes\n(Se requieren > 2 pulsos)", 
                     ha='center', va='center', color='#ff5555', transform=ax3.transAxes, fontsize=12)
            
        ax3.grid(True, color='#444444', linestyle='--', alpha=0.3)
        # --------------------------------------------------------

        plt.subplots_adjust(hspace=0.1, wspace=0.25)

        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self.plot_container)
        
        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas)

    def export_csv(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Guardar CSV", "", "CSV (*.csv)")
        if not save_path: return

        try:
            self.txt_analysis_log.append(f"\nExportando a {os.path.basename(save_path)}...")
            QApplication.processEvents() 
            
            # Aplicamos la suma vectorial del tiempo absoluto si existe el ATM
            export_rows = []
            import datetime
            
            for row in self.csv_rows:
                t_relativo = row[0]
                if getattr(self, 'root_atm_unix', None) is not None:
                    t_absoluto = self.root_atm_unix + t_relativo
                    
                    # Convertimos a hora legible por humanos (UTC)
                    dt_obj = datetime.datetime.fromtimestamp(t_absoluto, tz=datetime.timezone.utc)
                    human_time = dt_obj.strftime('%Y-%m-%d %H:%M:%S.%f')
                    
                    # Guardamos ambas: la matemática y la humana, manteniendo precisión de microsegundos
                    new_row = [f"{t_absoluto:.6f}", human_time] + row[1:] 
                    export_rows.append(new_row)
                else:
                    export_rows.append([row[0], "Sin ATM (Relativo)"] + row[1:]) 
            
            with open(save_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time_Abs_Unix", "Time_UTC_Human", "Ax_g", "Ay_g", "Az_g", "Gx_deg_s", "Gy_deg_s", "Gz_deg_s", "PPS_Sync"])
                writer.writerows(export_rows)
                
            self.txt_analysis_log.append("✓ ¡Exportación exitosa!")
        except Exception as e:
            self.txt_analysis_log.append(f"❌ Error al exportar: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())