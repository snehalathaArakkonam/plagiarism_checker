# 🔍 PlagioScan — AI-Powered Plagiarism Checker

A complete, professional **Streamlit** plagiarism detection web application built with pure Python.
Designed as a portfolio/college project.

---

## 🚀 Quick Start

```bash
# 1. Clone or copy this folder
cd plagiarism_checker

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## 📦 Features

| Feature | Details |
|---|---|
| 🔍 Plagiarism Check | TF-IDF + Cosine Similarity against 12 sample paragraphs |
| 🎨 Sentence Highlighting | Red = high match, Yellow = moderate, Green = original |
| ⚖️ Compare Two Texts | Side-by-side diff with SequenceMatcher |
| 📋 Report History | Last 8 reports auto-saved with pickle |
| 📥 PDF Export | Full report via fpdf2 |
| 📎 File Upload | .txt and .docx support |
| 📊 Live Word Count | Real-time words / characters / sentences |
| 🌙 Dark Theme | Professional dark UI with custom CSS |

---

## 📁 Project Structure

```
plagiarism_checker/
├── app.py              ← Main application (single file)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

Reports are saved to `plagioscan_reports.pkl` in the same directory.

---

## ⚙️ Tech Stack

- **Streamlit** – UI framework
- **scikit-learn** – TF-IDF vectorisation + cosine similarity
- **difflib** – Sentence-level sequence matching
- **fpdf2** – PDF report generation
- **python-docx / docx2txt** – .docx file parsing
- **pickle** – Local report persistence

---

## ⚠️ Disclaimer

This is a **demo application** built for educational and portfolio purposes.  
Results are based on a **hardcoded sample database** and do not represent real-world plagiarism detection.  
Not intended for commercial or academic integrity enforcement use.

---

*Built with ❤️ using Python & Streamlit*
