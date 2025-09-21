# 🛍️ Product Recommendation System  

A **hybrid product recommendation system** built using **Python, Machine Learning, and Streamlit**.  
This project demonstrates how data-driven recommendations can enhance customer experience in e-commerce by combining **popularity-based filtering** and **association rule mining**.  

---

## 💡 Skills Highlighted  
- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Recommendation Systems (Popularity-based + Association Rules)  
- Python (pandas, numpy, scikit-learn, mlxtend)  
- Data Visualization (matplotlib, seaborn)  
- Web App Development with Streamlit  
- End-to-End ML Workflow (EDA → Modeling → Deployment)  

---

## 🚀 Project Overview  
- **Objective:** Build a recommender system that suggests relevant products to users.  
- **Dataset:** [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail) (Transactions from a UK-based retailer).  
- **Approach:**  
  - Data cleaning & preprocessing (handling nulls, removing invalid transactions).  
  - Exploratory Data Analysis (EDA) for insights into customer behavior.  
  - Popularity-based recommendation for trending items.  
  - Association Rule Mining (Apriori algorithm) to suggest items frequently bought together.  
- **Frontend:** Interactive **Streamlit app** for live recommendations.  

---

## 🧑‍💻 Tech Stack  
- **Languages:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, mlxtend, scikit-learn  
- **Web App:** Streamlit  
- **Other Tools:** Jupyter Notebook  

---

## 📂 Project Structure  
```
├── Product_Recommendation.ipynb   # EDA & model building  
├── recommender.py                 # Core recommendation logic  
├── streamlit_app.py               # Streamlit frontend  
├── Online Retail.xlsx             # Dataset (not uploaded to GitHub due to size)  
├── requirements.txt               # Dependencies  
└── README.md                      # Project documentation  
```

---

## ⚙️ Installation & Setup  
1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/product-recommendation.git
   cd product-recommendation
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:  
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 🎯 Usage  
- Explore **popular products** in the dataset.  
- Get **item-to-item recommendations** (products frequently bought together).  
- Interactive visualization of insights via the Streamlit app.  

🔹 Example:  
If a customer buys *“White Mug”*, the system may recommend *“Tea Spoon Set”* based on past purchase patterns.  

---

## 🌟 Key Learning Outcomes  
- Hands-on experience in **data preprocessing and cleaning**.  
- Applied **EDA** to derive business insights.  
- Implemented a **hybrid recommendation system** (popularity + association rules).  
- Built an interactive app using **Streamlit**.  
- Strengthened understanding of **real-world ML workflows**.  

---

## 🔮 Future Enhancements  
- Add **collaborative filtering** (user-user or item-item similarity).  
- Deploy app on **cloud platforms** (Heroku / AWS / GCP).  

---

## 🌐 Streamlit App  
🔗 [Click here to try the app](https://hybridrecodemo-ffjwcqwvkx4pghnaguflm8.streamlit.app/)  
 
