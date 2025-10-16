# 🌍 Transformer-Based Speech Recognition System for Low-Resource Languages

### 🎓 Mini Research Project – 2025  
**Developed using TensorFlow, Keras, Flask, HTML, CSS, and JavaScript**

---

## 🧠 Overview

Modern speech recognition models such as **Whisper** and **Wav2Vec2.0** have achieved great success on high-resource languages like English and Mandarin.  
However, these models often fail to perform well for **low-resource languages** due to limited training data, lack of linguistic diversity, and high computational costs.  

Our research addresses the **low-resource language recognition gap** by developing a **Transformer-based Speech Recognition System** optimized for **low-resource languages** using publicly available multilingual datasets and data augmentation techniques.

---

## 🚩 Problem Statement

Current multilingual ASR (Automatic Speech Recognition) models are limited to around **100 languages**, whereas there are **over 7000 languages spoken worldwide**.  
Low-resource languages face significant challenges such as:

- 🧩 **Data Scarcity** – A lack of high-quality, labeled audio-text pairs.  
- ⚙️ **Domain Mismatch** – Existing datasets often don't match the real-world use cases for these languages.  
- 💻 **Computational Constraints** – Large-scale ASR models demand significant GPU/TPU resources.  
- 🗣️ **Dialect Diversity** – A single language can have numerous dialects, further fragmenting available data.

---

## 💡 Research Gap

| Identified Gap | Description |
|----------------|-------------|
| Limited Coverage | High-quality ASR systems trained on only a small subset of global languages. |
| Lack of Balanced Datasets | Imbalanced multilingual corpora with dominance of English and European languages. |
| Computational Barriers | Training large models on low-resource datasets is often infeasible. |
| Dialect Variation | Absence of datasets that represent multiple dialects per language. |

---

## 🧩 Proposed Solution

Our research introduces a **transformer-based ASR model** that focuses on improving low-resource language recognition accuracy through:

1. **Data Aggregation** – Combining multiple open multilingual datasets.  
2. **Bible Speech Dataset** – Using the globally distributed Bible speech corpus (available in 1000+ languages).  
3. **Data Augmentation** – Applying audio transformations to enhance robustness.  
4. **Performance Filtering** – Excluding datasets with **WER > 10%** to control computational cost.  
5. **Cross-lingual Fine-tuning** – Leveraging transformer architectures inspired by **XLS-R** and **Whisper**.

---

## ⚙️ Methodology

### 🔹 1. Data Aggregation

We collected and combined publicly available multilingual speech datasets:

| Dataset | Description | Source |
|----------|--------------|--------|
| **Bible Speech Dataset** | Audio aligned with Bible text in 1000+ languages | Global Open Dataset |
| **VoxPopuli (VP-400K)** | 400K hours of multilingual speech | Facebook AI |
| **Multilingual Librispeech (MLS)** | Public-domain audiobooks in multiple languages | OpenSpeech |
| **CommonVoice (CV)** | Crowd-sourced speech data | Mozilla |
| **VoxLingua107 (VL)** | 107-language dataset | University of Helsinki |
| **BABEL (BBL)** | Telephone speech corpus | Linguistic Data Consortium |

---

### 🔹 2. Data Augmentation Techniques

To overcome data scarcity and improve model robustness, we applied the following augmentations:

- 🎚️ **Band-Stop Filtering:** Simulates environmental noise by removing selected frequency bands.  
- 🔊 **Gaussian Noise Injection:** Adds random background noise to improve generalization.  
- 🎵 **Pitch Shifting:** Alters pitch to simulate accent and speaker variations.  

These techniques help the model learn invariant features and perform better under diverse acoustic conditions.

---

### 🔹 3. Training Process

- **Frameworks:** TensorFlow, Keras  
- **Model Architecture:** Transformer Encoder–Decoder  
- **Optimizer:** AdamW  
- **Loss Function:** CTC (Connectionist Temporal Classification)  
- **Evaluation Metric:** Word Error Rate (WER)

#### ✳️ Dataset Filtering Rule
During training:
> If **WER > 10%**, that dataset is removed from further training to save computation and maintain quality.

---

## 🧮 Architecture Overview

```
Input Audio  🎤
       ↓
Feature Extraction (STFT)
       ↓
Transformer Encoder-Decoder (Self-Attention Layers)
       ↓
CTC Decoder
       ↓
Recognized Text Output 📜
```

---

## 🧑‍💻 Technology Stack

| Layer | Tools / Frameworks |
|--------|--------------------|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | Python (Flask) |
| **Model Development** | TensorFlow, Keras |
| **Preprocessing** | NumPy, Pandas, Librosa |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Flask API with simple web interface |
| **Version Control** | Git & GitHub |

---

## 🌐 System Workflow

1. User uploads an **audio file** (e.g., `.wav`, `.mp3`).  
2. Flask backend sends the audio to the trained ASR model.  
3. Transformer-based model performs **speech-to-text inference**.  
4. Recognized text is sent back to the web interface.  
5. User views transcription output and system metrics.

---

## 🧰 Project Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mufeezmujassir/speech-to-text.git
cd transformer-speech-recognition
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
```

### 3️⃣ Activate Environment
For **Windows**:
```bash
venv\Scripts\activate
```
For **Linux / macOS**:
```bash
source venv/bin/activate
```

### 4️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 5️⃣ Run the Flask Application
```bash
python start_app.py
```

### 6️⃣ Open in Browser
Navigate to:
```
http://127.0.0.1:8000
```

---

## 📈 Model Evaluation

| Metric | Description |
|--------|-------------|
| **WER (Word Error Rate)** | Measures transcription accuracy |
| **CER (Character Error Rate)** | Evaluates fine-grained performance |
| **Loss Curve** | Tracks model convergence and stability |

---

## 📚 Research Inspirations

Our work builds upon and extends insights from state-of-the-art research in multilingual ASR.

1. **Scaling Speech Technology to 1,000+ Languages**  
   *Vineel Pratap, Andros Tjandra, Bowen Shi, et al.*  
   *Meta AI, 2023*  
   [arXiv:2305.13516v1](https://arxiv.org/abs/2305.13516)

2. **XLS-R: Self-Supervised Cross-Lingual Speech Representation Learning at Scale**  
   *Arun Babu, Changhan Wang, Andros Tjandra, et al.*  
   *Meta AI, 2021*  
   [arXiv:2111.09296v3](https://arxiv.org/abs/2111.09296)

3. **Whisper Turns Stronger: Augmenting Wav2Vec 2.0 for Superior ASR in Low-Resource Languages**  
   *Or Haim Anidjar, Revital Marbel, Roi Yozevitch, 2024*  
   [arXiv:2501.00425v1](https://arxiv.org/abs/2501.00425)

4. **Speech Language Models for Under-Represented Languages: Insight**  
   *Yaya Sy, Dioula Doucouré, Christophe Cerisara, 2025*  
   [arXiv:2509.15362v2](https://arxiv.org/abs/2509.15362)

---

## 🧑‍🔬 Experimental Results

✅ Improved ASR performance for low-resource languages  
✅ Demonstrated effective dataset aggregation strategy  
✅ Achieved significant **WER reduction** compared to baseline models  
✅ Scalable model architecture for further fine-tuning across new languages  

---

## 🎯 Future Enhancements

- 🔹 Integrate **Whisper + XLS-R hybrid** architecture  
- 🔹 Real-time ASR with GPU inference optimization  
- 🔹 Expansion of dataset with regional dialect coverage  
- 🔹 Web-based visualization dashboard for attention maps  
- 🔹 Deployment via **Hugging Face Spaces** or **Streamlit**

---

## 🧾 License

This project is released under the **MIT License**.  
You may freely use, modify, and distribute with proper attribution.

---

## 💬 Acknowledgements

Special thanks to:
- **Meta AI** and **Google AI** for open-source research inspiration  
- **Mozilla CommonVoice** for multilingual dataset contributions  
- **Hugging Face** for pre-trained transformer resources  
- **SLIIT Faculty** for guidance and support throughout this research  

---

> _"Empowering every voice — one language at a time."_ 🌎🎙️

---

## 👨‍💻 Developed By

**Mufeez Mujassir, Sisitha, Kulitha, Nadee Withana**  
_Sri Lanka Institute of Information Technology (SLIIT)_  
📧 **mufeezmujassir80@gmail.com**
🔗 [GitHub Profile](https://github.com/mufeezmujassir)
