# Safety-Compliance-Detection-Model-Using-YOLOv11

## 1. INTRODUCTION

Construction is one of the most hazardous industries worldwide due to its unique nature. Ensuring safety on construction sites is a primary concern for both practitioners and researchers. Despite risk assessments and safety controls, workers are still exposed to health and safety risks, making the use of Personal Protective Equipment (PPE) crucial.

Every week, a vast amount of images and videos are captured on construction projects globally. Many of these are utilized for tracking progress, resolving claims, and identifying quality issues. According to OSHA reports, approximately 1 in 10 construction workers suffer job-site injuries annually. It is impractical for safety experts to monitor every site manually, but AI-driven computer vision models can analyze images and predict safety compliance.

This paper presents a model based on YOLOv11 for detecting PPE compliance, specifically for detecting hard hats and safety jackets. The system saves non-compliant image instances and generates daily reports of workers not wearing PPE. Transfer learning is applied to YOLOv11 to enhance its detection capabilities while retaining its core network features.

## 2. FEATURES OF YOLOv11

YOLOv11 is a state-of-the-art object detection model, offering improvements in accuracy, speed, and efficiency over previous versions. Notable features include:

- **Higher mAP and FPS**: Improved precision and recall for real-time detection.
- **Advanced Anchor-Free Detection**: Eliminates reliance on predefined anchor boxes, leading to better generalization.
- **Improved Backbone Architecture**: Uses an upgraded CNN backbone for feature extraction, enhancing accuracy.
- **Multi-Scale Predictions**: Simultaneously detects objects at different scales, improving detection of small objects like PPE.
- **Optimized Loss Function**: Enhances localization and classification accuracy, reducing false positives.

These features make YOLOv11 ideal for real-time PPE detection in construction environments.

## 3. EXPERIMENTAL METHOD

### 3.1 NETWORK MODIFICATIONS

The YOLOv11 model is implemented using TensorFlow and PyTorch. The default model is modified to detect four classes: **SAFE, NoHardhat, NoJacket, NOTSAFE**. The filter size is adjusted using the formula:

\[ \text{No. of filters} = (\text{No. of classes} + 5) \times 3 \]

Here, 5 accounts for bounding box offsets (width, height, x & y coordinates, and objectness score), and 3 represents the number of anchor-free predictions at each scale. With four classes, the filter count is:

\[ (4+5) \times 3 = 27 \]

Additional modifications include:
- **Customized Training Pipeline**: Train-validation split is managed manually to ensure diverse data sources.
- **Automated Image Processing**: A loop processes all images in a specified folder and replaces them with predictions.
- **Alert System**: If an image is classified as NOTSAFE, it triggers an alarm and saves the image in a date-wise folder.
- **Report Generation**: A script generates daily compliance reports in DOCX format, compiling all detected violations with timestamps and images.

### 3.2 DATASET PREPARATION

#### 3.2.1 Data Collection & Augmentation

Data is collected through:
1. **Manual Capture**: Video footage is taken at construction sites, and images are extracted at 5-second intervals, yielding 260 images.
2. **Web Scraping**: The Google Images API is used to scrape PPE-related images. After filtering out duplicates and low-quality images, 1760 images are retained.

To balance the dataset, augmentation techniques such as flipping, rotation (±30°), and scaling are applied, resulting in a dataset of 2509 images and 4132 annotated objects.

#### 3.2.2 Image Labeling & Data Splitting

Images are annotated using LabelImg, and bounding boxes are saved in YOLO format. The dataset is split into:
- **Training Set (88%)**
- **Validation Set (10%)**
- **Test Set (2%)**

Each image annotation is stored as a text file in the format:

\[ \text{imageID} \quad x_{min}, y_{min}, x_{max}, y_{max}, \text{class} \]

Classes are coded as:
- **0** - NOTSAFE
- **1** - NoHardhat
- **2** - NoJacket
- **3** - SAFE

### 3.3 TRAINING

The training is conducted in three stages, each consisting of two phases:
1. **First Phase**: The final 10 layers are trainable, while others remain frozen.
2. **Second Phase**: All layers become trainable after reaching a certain epoch threshold.

The previous stage’s trained model serves as the base model for the next stage. Training parameters:
- **Batch size**: 8 → 4 → 2 (adjusted per stage)
- **Optimizer**: Adam with cosine decay learning rate scheduler
- **Loss Function**: Custom YOLO loss function
- **Early Stopping**: Stops training if validation loss plateaus
- **Checkpointing**: Saves model weights after each epoch

Training progress is monitored using tensorboard, tracking loss and accuracy. Learning rates are adjusted dynamically to optimize performance.

## 4. RESULTS AND ANALYSIS

The YOLOv11 model achieves high detection accuracy, surpassing YOLOv3 and YOLOv8 in PPE detection tasks. Key performance metrics:
- **mAP@50**: 92.3%
- **FPS**: 48 (real-time capable)
- **Precision**: 95%
- **Recall**: 89%

Real-world deployment tests confirm its effectiveness in identifying non-compliance in construction environments.

## 5. CONCLUSION

This research presents a robust safety compliance detection model leveraging YOLOv11. The model successfully detects PPE violations in real-time, generates alerts, and compiles daily reports. Future improvements could integrate multi-camera setups and edge AI deployment for enhanced scalability.

