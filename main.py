import cv2
import sys
import os

from src.preprocessing import preprocess_plate_for_segmentation
from src.detector.yolo_detector import YoloDetector
from src.character_segmentation import segment_characters
from src.character_recognition import CharacterRecognizer
from src.visualization import (
    show_pipeline_steps,
    show_plate_detection,
    show_segmented_characters,
    show_final_result
)
from src.utils import generate_mock_templates, generate_mock_test_image

def main(image_path):
    print(f"Processing image: {image_path}")
    
    # 0. Load Image
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        sys.exit(1)
        
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image.")
        sys.exit(1)
        
    # 1. Plate Detection with YOLO
    print("Detecting license plate with YOLO...")
    detector = YoloDetector()
    best_box = detector.detect(image)
    
    show_plate_detection(image, best_box)
    
    if best_box is None:
        print("Failed to detect a license plate with YOLO.")
        sys.exit(1)
        
    # Crop Plate (exact bounding box to avoid dark borders)
    x, y, w, h, conf = best_box
    y1, y2 = max(0, y), min(image.shape[0], y + h)
    x1, x2 = max(0, x), min(image.shape[1], x + w)
    
    plate_img = image[y1:y2, x1:x2].copy()
    
    # 2. Preprocessing Pipeline for the cropped plate
    print("Running preprocessing pipeline on cropped plate (Grayscale -> Gaussian -> Threshold -> Morphology)...")
    preprocessed, intermediate_results = preprocess_plate_for_segmentation(plate_img)
    
    # Display preprocessing steps
    show_pipeline_steps(plate_img, intermediate_results)
    
    # 3. Character Segmentation
    print("Segmenting characters...")
    characters, binary_plate, char_contours = segment_characters(plate_img)
    
    show_segmented_characters(characters)
    
    if not characters:
        print("Failed to segment any characters.")
        sys.exit(1)
        
    # 4. Character Recognition
    print("Recognizing characters...")
    recognizer = CharacterRecognizer()
    plate_text = recognizer.recognize_plate(characters)
    
    print(f"=====================================")
    print(f"Recognized Plate: {plate_text}")
    print(f"=====================================")
    
    # 5. Final Result Visualization
    show_final_result(image, plate_text, best_box)

if __name__ == "__main__":
    # Setup step: generate templates if they don't exist
    generate_mock_templates()
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Check for real images first
        images = sorted([f for f in os.listdir("data/images") if f.endswith(('.jpg', '.png'))])
        if images:
            print("=====================================")
            print("Selecione a imagem que deseja testar:")
            for i, img_name in enumerate(images):
                print(f"{i + 1}. {img_name}")
            print("=====================================")
            
            try:
                choice = input("Digite o número da imagem (ou pressione Enter para escolha aleatória): ").strip()
                if not choice:
                    import random
                    img_path = os.path.join("data/images", random.choice(images))
                    print(f"Nenhuma escolha feita. Imagem aleatória selecionada: {img_path}")
                else:
                    index = int(choice) - 1
                    if 0 <= index < len(images):
                        img_path = os.path.join("data/images", images[index])
                    else:
                        print("Opção inválida. Usando a primeira imagem da lista.")
                        img_path = os.path.join("data/images", images[0])
            except ValueError:
                print("Entrada inválida. Usando a primeira imagem da lista.")
                img_path = os.path.join("data/images", images[0])
        else:
            print("Nenhum caminho de imagem fornecido e nenhuma imagem encontrada. Gerando mock...")
            img_path = "data/images/test.jpg"
            generate_mock_test_image(img_path)
            
    main(img_path)
