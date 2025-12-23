## 🎭 Face Recognition & Anti-Spoofing Det

A real-time face anti-spoofing detection system built with MobileNetV2 to distinguish between real faces and spoofed faces (photos, videos, etc.).

## 📋 Overview

This project implements a face anti-spoofing detection system using deep learning techniques. The system can detect whether a face presented to the camera is real or a spoof attempt (such as a printed photo or a video replay attack).

## 🚀 Features

- Real-time face detection and anti-spoofing
- MobileNetV2-based deep learning model
- Interactive web application interface
- High accuracy in detecting spoofing attempts
- Lightweight and efficient for real-time processing

## 📁 Project Structure

```
Face-Recognition/
├── MobileNetFaceSpoof.h5                              # Trained model weights
├── face-anti-spoofing-detection-using-mobilenetv2.ipynb # Model training notebook
├── Real_time_Detection_Face_Spoofing.ipynb            # Real-time detection demo
├── app21.py                                           # Flask/Streamlit application
├── lastapp.py                                         # Updated application version
├── util2.py                                           # Utility functions
├── test 1.py                                          # Testing script
└── Paper_face_detection.docx                          # Research documentation
```

## 🛠️ Technologies Used

- **Python** - Primary programming language
- **Jupyter Notebook** - Model development and experimentation
- **MobileNetV2** - Efficient CNN architecture for mobile and embedded vision applications
- **OpenCV** - Computer vision and image processing
- **TensorFlow/Keras** - Deep learning framework
- **Flask/Streamlit** - Web application framework

## 📦 Installation

1. Clone the repository: 
```bash
git clone https://github.com/Ahmad-Mobarak/Face-Recognition.git
cd Face-Recognition
```

2. Install required dependencies:
```bash
pip install tensorflow opencv-python numpy flask streamlit
```

3. Ensure you have the trained model file (`MobileNetFaceSpoof.h5`) in the project directory. 

## 🎯 Usage

### Running the Web Application

```bash
python app21.py
# or
python lastapp.py
```

### Running the Jupyter Notebooks

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open either: 
   - `Real_time_Detection_Face_Spoofing.ipynb` for real-time detection
   - `face-anti-spoofing-detection-using-mobilenetv2.ipynb` for model training

### Testing

```bash
python "test 1.py"
```

## 🧠 How It Works

1. **Face Detection**: The system first detects faces in the video stream using OpenCV
2. **Preprocessing**: Detected faces are preprocessed to match the model's input requirements
3. **Classification**: The MobileNetV2 model classifies the face as "Real" or "Spoof"
4. **Real-time Feedback**: Results are displayed in real-time on the video stream

## 📊 Model Architecture

The project uses **MobileNetV2**, a lightweight deep learning architecture that:
- Provides excellent accuracy with minimal computational requirements
- Uses depthwise separable convolutions for efficiency
- Is optimized for mobile and embedded devices
- Achieves real-time performance on standard hardware

## 🔍 Applications

- **Security Systems**: Prevent unauthorized access using fake photos
- **Banking & Finance**: Secure biometric authentication
- **Access Control**: Ensure genuine face verification
- **Device Unlocking**: Protect devices from spoofing attacks

## 📝 Research Documentation

For detailed information about the methodology and research, refer to `Paper_face_detection.docx` included in the repository.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License. 

## 👤 Author

**Ahmad Mobarak**
- GitHub: [@Ahmad-Mobarak](https://github.com/Ahmad-Mobarak)

## 📧 Contact

For questions or collaboration opportunities, please open an issue in this repository. 

## 🙏 Acknowledgments

- MobileNetV2 architecture by Google Research
- Face detection using OpenCV's pre-trained models
- Community contributions and feedback

---

⭐ If you find this project useful, please consider giving it a star!
