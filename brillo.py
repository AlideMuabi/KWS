import pyaudio
import numpy as np
import tensorflow as tf
from scipy.signal import spectrogram
import screen_brightness_control as sbc
import threading
import queue
import time

class KeywordSpotter:
    def __init__(self, model_path, sample_rate=16000, chunk_size=1024):
        """
        Inicializa el detector de palabras clave
        
        Args:
            model_path: Ruta al modelo entrenado (.h5 o SavedModel)
            sample_rate: Frecuencia de muestreo (16kHz para tu modelo)
            chunk_size: Tamaño del buffer de audio
        """
        loaded = tf.saved_model.load("Modelo_5")
        print("Firmas disponibles:", list(loaded.signatures.keys()))
        self.model = loaded 
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.is_running = False

        
        # Buffer para 16000 muestras (1 segundo)
        self.buffer_size = 16000
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        
        # Configuración de PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Umbrales de confianza
        self.confidence_threshold = 0.7
        
        # Control de brillo
        self.current_brightness = sbc.get_brightness()[0]  # Asume primer monitor
        self.brightness_step = 1
        
    def preprocess_audio(self, audio_data):
        """
        Preprocesa el audio crudo para que coincida con la entrada del modelo (forma: (1, 16000))
        """
        # Asegurar que tenga exactamente 16000 muestras
        if len(audio_data) < 16000:
            audio_data = np.pad(audio_data, (0, 16000 - len(audio_data)))
        else:
            audio_data = audio_data[:16000]

        # Normalizar el audio
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)

        # Cambiar la forma a (1, 16000) para el modelo
        return audio_data.reshape(1, 16000).astype(np.float32)

    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback para capturar audio del micrófono"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convertir bytes a numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def process_audio(self):
        """Procesa el audio acumulado y ejecuta predicciones"""
        while self.is_running:
            try:
                # Obtener nuevo chunk de audio
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Actualizar buffer (sliding window)
                self.audio_buffer = np.roll(self.audio_buffer, -len(audio_chunk))
                self.audio_buffer[-len(audio_chunk):] = audio_chunk
                
                # Preprocesar
                processed_audio = self.preprocess_audio(self.audio_buffer)
                
                # Predicción
                output = self.model(tf.constant(processed_audio))
                prediction = list(output.values())[0].numpy()


                
                # Interpretar resultado: [prob_descarga, prob_sube, prob_baja]
                prob_descarga = prediction[0][1]  # ruido de fondo
                prob_sube = prediction[0][2]      # comando "sube"
                prob_baja = prediction[0][0]      # comando "baja"
                
                # Mostrar probabilidades (opcional, para debug)
                # print(f"Descarga: {prob_descarga:.2f}, Sube: {prob_sube:.2f}, Baja: {prob_baja:.2f}")
                
                # Detectar comando
                print(f"🎧 Probabilidades -> Descarga: {prob_descarga:.2f}, Baja: {prob_baja:.2f}, Sube: {prob_sube:.2f}")
                if prob_sube > self.confidence_threshold and prob_sube > prob_baja and prob_sube > prob_descarga:
                    self.execute_command("sube")
                elif prob_baja > self.confidence_threshold and prob_baja > prob_sube and prob_baja > prob_descarga:
                    self.execute_command("baja")
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en procesamiento: {e}")
    
    def execute_command(self, command):
        """Ejecuta el comando detectado"""
        try:
            if command == "sube":
                new_brightness = min(100, self.current_brightness + self.brightness_step)
                sbc.set_brightness(new_brightness)
                self.current_brightness = new_brightness
                print(f"🔆 Subiendo brillo a {new_brightness}%")
                
            elif command == "baja":
                new_brightness = max(0, self.current_brightness - self.brightness_step)
                sbc.set_brightness(new_brightness)
                self.current_brightness = new_brightness
                print(f"🔅 Bajando brillo a {new_brightness}%")
                
        except Exception as e:
            print(f"Error ajustando brillo: {e}")
    
    def start_listening(self):
        """Inicia la captura de audio y procesamiento"""
        try:
            # Configurar stream de audio
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            # Iniciar procesamiento en hilo separado
            self.is_running = True
            self.processing_thread = threading.Thread(target=self.process_audio)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            # Iniciar stream
            self.stream.start_stream()
            
            print("🎙️  Escuchando... Di 'sube' o 'baja' para controlar el brillo")
            print("Presiona Ctrl+C para salir")
            
            # Mantener el programa corriendo
            while self.is_running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n⏹️  Deteniendo...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.stop_listening()
    
    def stop_listening(self):
        """Detiene la captura y limpia recursos"""
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)
        
        self.audio.terminate()
        print("✅ Recursos liberados")

def main():
    # Configuración
    MODEL_PATH = "Modelo_5"  # Tu modelo exportado de TensorFlow
    
    # Crear detector
    spotter = KeywordSpotter(MODEL_PATH)
    
    # Iniciar

    spotter.start_listening()

if __name__ == "__main__":
    main()