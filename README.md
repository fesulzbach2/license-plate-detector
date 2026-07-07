# License Plate Detector

Este projeto implementa um sistema completo de detecção e reconhecimento de placas de carro utilizando **exclusivamente técnicas clássicas de Processamento Digital de Imagens**. 
Foi desenvolvido com foco didático para disciplinas de Visão Computacional, evitando completamente o uso de Inteligência Artificial, Redes Neurais, Machine Learning ou detectores prontos (como YOLO ou Haar Cascades).

A biblioteca OpenCV é restrita apenas a operações básicas (leitura/escrita de imagens, desenho de primitivas geométricas e extração de contornos). Os algoritmos de filtragem, detecção de bordas, limiarização, operações morfológicas e reconhecimento (Template Matching) foram implementados manualmente usando **Python** e **NumPy**.

## Objetivo

Implementar um pipeline que recebe uma imagem contendo um carro e seja capaz de:
1. Detectar automaticamente a placa através de suas propriedades geométricas.
2. Desenhar um retângulo ao redor dela e recortá-la.
3. Binarizar e segmentar individualmente cada caractere da placa.
4. Reconhecer os caracteres extraídos utilizando Template Matching.

## Estrutura do Projeto

```
license-plate-detector/
├── README.md
├── requirements.txt
├── main.py                    # Ponto de entrada do pipeline
├── config.py                  # Constantes e limiares do sistema
├── data/
│   ├── images/                # Imagens de teste
│   ├── templates/             # Templates A-Z, 0-9 para o reconhecimento
│   └── results/               # Resultados visuais
├── src/
│   ├── algorithms/            # Implementações matemáticas manuais
│   │   ├── convolution.py
│   │   ├── gaussian.py
│   │   ├── sobel.py
│   │   ├── threshold.py
│   │   ├── morphology.py
│   │   └── template_matching.py
│   ├── preprocessing.py       # Pipeline de pré-processamento
│   ├── plate_detector.py      # Lógica de extração do candidato à placa
│   ├── character_segmentation.py # Lógica de recorte dos caracteres
│   ├── character_recognition.py  # Reconhecimento via Template Matching
│   ├── visualization.py       # Funções didáticas usando matplotlib
│   └── utils.py               # Funções de suporte
```

## Instalação

Crie um ambiente virtual (opcional, mas recomendado) e instale as dependências. As dependências são mínimas.

```bash
pip install -r requirements.txt
```

## Como Executar

Para rodar o projeto, execute o arquivo `main.py` passando o caminho da imagem como argumento.

```bash
python main.py data/images/sua_imagem.jpg
```

**Nota:** Se nenhum caminho for passado, o sistema gerará automaticamente uma imagem sintética (`data/images/test.jpg`) e "templates" mockados (caso não existam em `data/templates/`) para validar a execução.

## O Pipeline

1. **Escala de Cinza:** Conversão manual (método da luminosidade).
2. **Filtro Gaussiano:** Convolução manual de um kernel gerado a partir da função matemática Gaussiana para remoção de ruídos.
3. **Sobel e Magnitude de Gradiente:** Convolução manual com os kernels Sobel X e Y para detectar variações de intensidade, destacando linhas verticais (características comuns em caracteres de placas).
4. **Limiarização (Threshold):** Transformação da imagem para formato binário usando o método de Otsu (manual).
5. **Morfologia Matemática:** Aplicação de Dilatação e Fechamento Morfológico manuais para fundir as bordas e formar um bloco retangular coerente que corresponda à placa.
6. **Detecção de Componentes Conectados (OpenCV):** Busca por contornos que obedeçam às propriedades esperadas de uma placa (Área, Razão de Aspecto e Retangularidade).
7. **Recorte:** O melhor candidato é escolhido baseado em um escore.
8. **Segmentação dos Caracteres:** A placa extraída é binarizada novamente, filtrada (abertura morfológica) e seus caracteres são localizados, ordenados da esquerda para a direita e recortados individualmente.
9. **Template Matching:** O algoritmo compara cada caractere segmentado com templates locais, calculando o erro absoluto (SAD - Sum of Absolute Differences) para decidir o caractere mais provável.

## Descrição dos Algoritmos Manuais

Os arquivos presentes em `src/algorithms/` contêm a implementação didática dos filtros:

*   **`convolution.py`**: Motor genérico de convolução 2D responsável por extrair um ROI (Region of Interest), multiplicar pelo kernel correspondente e somar os valores.
*   **`gaussian.py`**: Calcula as probabilidades de uma distribuição Gaussiana bidimensional para criar a matriz de pesos.
*   **`sobel.py`**: Define e aplica as matrizes aproximadas da derivada de 1ª ordem para encontrar bordas.
*   **`morphology.py`**: Operações morfológicas (Max para dilatação e Min para erosão) iteradas sobre as vizinhanças dos pixels.
*   **`threshold.py`**: Calcula variâncias interclasses para maximizar a separação de fundo/frente em imagens.
*   **`template_matching.py`**: Redimensiona a entrada e o template e avalia pixel a pixel qual possui menor divergência em seus valores.

## Limitações e Resultados Esperados

*   **Performance:** Devido ao aninhamento múltiplo de loops iterando na escala do pixel (em Python puro com uso brando do vetor NumPy), os algoritmos de **convolução** são substancialmente mais lentos em comparação com as implementações originais em C++ (do OpenCV).
*   **Variação de Ângulo:** O sistema foi dimensionado para placas razoavelmente alinhadas à câmera. Como o template matching e as restrições de aspecto assumem retângulos perfeitos, perspectivas distorcidas poderão falhar em detectar a placa ou reconhecer os caracteres.
*   **Iluminação e Sujeira:** Reflexos fortes no formato de uma placa ou letreiros com proporções idênticas causarão ruído no template matching, uma vez que a abordagem não abstrai a forma (como faria uma CNN), baseando-se estritamente na distribuição de pixels brancos contra os pretos no caractere segmentado.
