# 🩺 Medical Domain Identifier using NLP & Machine Learning

---

## 📖 Overview

This project focuses on classifying patient prescriptions and clinical notes into their respective **medical specialties** (like cardiology, orthopedics, dermatology, etc.) using **Natural Language Processing (NLP)** and **Machine Learning**. It helps automate categorization in hospital management systems, making patient routing and document handling more efficient.

---

## 🛠️ Technologies Used

| Tool / Library     | Purpose                                      |
|--------------------|----------------------------------------------|
| Python             | Core programming language                    |
| Pandas, NumPy      | Data manipulation and analysis               |
| Scikit-learn       | ML model training, TF-IDF, PCA               |
| NLTK / spaCy       | Text preprocessing, lemmatization            |
| Matplotlib, Seaborn| Data visualization                           |
| Jupyter Notebook   | Interactive development environment          |

---

## ⚙️ How It Works

1. **Dataset**: Loads medical transcription records with labeled medical domains.
2. **Preprocessing**: Cleans text by removing punctuation, stopwords, and applies lemmatization.
3. **Feature Extraction**: Uses **TF-IDF** to convert text into numerical features.
4. **Dimensionality Reduction**: Applies **PCA** to reduce feature space and noise.
5. **Model Training**: Trains a **Random Forest Classifier** on transformed data.
6. **Prediction**: Predicts the medical domain for unseen clinical text data.

---

## 🧠 Model Pipeline

```plaintext
+---------------------+     +-------------------------+     +-------------------------+
| Medical Transcripts | --> | Data Preprocessing      | --> | TF-IDF Vectorization    |
|   (Raw Dataset)     |     | (Cleaning + Lemmatize)  |     | (Feature Extraction)    |
+---------------------+     +-------------------------+     +-------------------------+
                                                                  |
                                                                  v
                                                        +-------------------------+
                                                        | Principal Component     |
                                                        | Analysis (PCA)          |
                                                        | (Dim. Reduction)        |
                                                        +-------------------------+
                                                                  |
                                                                  v
                                                        +-------------------------+
                                                        | Random Forest Classifier|
                                                        | (Training + Prediction) |
                                                        +-------------------------+
                                                                  |
                                                                  v
                                                        +-------------------------+
                                                        | Patient Classification  |
                                                        | by Medical Specialty     |
                                                        +-------------------------+


---
```
This diagram:
- Starts from the left with raw input
- Flows to the right for early steps
- Then continues vertically for the ML stages
- Fits cleanly into GitHub-flavored Markdown

Would you like a stylized image version or Mermaid.js diagram next?

 
## 📊 Output

- ✅ Cleaned and vectorized text data
- ✅ Dimensionality-reduced feature set using PCA
- ✅ Final classification of records into correct medical domains
- ✅ Confusion matrix, accuracy score, and precision-recall evaluation

---

## ✅ Conclusion

This system demonstrates an effective application of NLP and machine learning to the healthcare domain. It successfully classifies medical documents based on their textual content into appropriate specialties, enabling better automation in clinical workflows. With more data and advanced models (like BERT), the system can be expanded for production-level use in hospitals and telemedicine platforms.

##  How to Run the Project

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ABHISHEKJULA07/Medical-domain-identification-NLP
2. **Install the Required Packages**
   ```bash
   pip install -r requirements.txt
3. **Run the Notebook**
   ```bash
   jupyter notebook.
4. **Open and run**
   ```bash
   model.ipynb

   ```bash

## 👨‍💻 Developed By
**Abhishek Jula**, Dhruv Uniyal, Tarun Sri Sai Vadlapatla, John Nikhil Bonam, Ms. A. Madhavi  
🔗 [LinkedIn](https://www.linkedin.com/in/abhi-jula0711)  
💻 [GitHub](https://github.com/ABHISHEKJULA07)  
🌐 [Portfolio](https://abhipinku66.wixsite.com/07112000)  
📧 abhishekjula018@gmail.com
