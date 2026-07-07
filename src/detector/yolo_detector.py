import os
import cv2
import config

try:
    from ultralytics import YOLO
except ImportError:
    # Fallback to prevent immediate crash if ultralytics is missing (though it shouldn't be)
    YOLO = None

class YoloDetector:
    def __init__(self):
        """
        Carrega o modelo YOLO especificado nas configurações.
        """
        if YOLO is None:
            raise ImportError("A biblioteca ultralytics não está instalada. Rode: pip install ultralytics")
            
        self.model_path = config.YOLO_MODEL_PATH
        
        # We try to load the model. If it doesn't exist, we fallback to a pre-trained yolov8n
        # just so the user can test the code without crashing if they haven't downloaded a custom plate model yet.
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
        else:
            print(f"Aviso: Modelo YOLO em {self.model_path} não encontrado.")
            print("Carregando o modelo base yolov8n.pt para não quebrar a execução...")
            self.model = YOLO('yolov8n.pt')

    def detect(self, image):
        """
        Realiza a inferência com o modelo YOLO na imagem fornecida.
        
        Args:
            image: Imagem BGR (numpy array).
            
        Returns:
            Uma tupla (x, y, w, h, conf) se uma placa for encontrada.
            Caso contrário, retorna None.
        """
        # Executa inferência
        results = self.model(image, verbose=False)
        
        best_box = None
        max_conf = -1.0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extrair confiança
                conf = float(box.conf[0])
                # Para simplificação, pegamos o box de maior confiança
                # (Se o modelo yolov8n base for usado e detectar carros, vamos pegar a maior conf)
                if conf > max_conf:
                    max_conf = conf
                    best_box = box
                    
        if best_box is not None:
            # Obter coordenadas (x_center, y_center, width, height) ou xyxy
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            
            # x, y no padrão de x_min, y_min
            x, y = x1, y1
            
            return (x, y, w, h, max_conf)
            
        return None
