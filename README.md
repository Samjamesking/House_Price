# 🏠 House Price Prediction Dashboard

A full-stack web application built with **Django** that predicts house prices based on various property features. Features a sleek glassmorphism UI with real-time price estimation, interactive charts, and prediction history.

---

## 🌟 Features

- 🔐 **Login & Register** pages with modern UI
- 📊 **Dashboard** with prediction statistics and history
- 🏡 **Real-time Price Estimator** based on property inputs
- 📈 **Interactive Charts** — price trends and comparisons
- 🗺️ **Location-based pricing** (Downtown, Suburb, etc.)
- 💾 **Prediction History** stored in SQLite database
- 🗑️ **Delete predictions** from history
- 💰 **Currency formatted** price display
- 📱 **Responsive Design** — works on all screen sizes

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Django (Python) |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite3 |
| Styling | Glassmorphism + Vanilla CSS |
| Charts | Chart.js |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/house-price-prediction.git
cd house-price-prediction

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install django

# 5. Run migrations
python manage.py migrate

# 6. Start the development server
python manage.py runserver
```

### Access the App
Open your browser and go to: **http://localhost:8000**

---

## 📁 Project Structure

```
house-price-prediction/
│
├── house_prediction_project/   # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── predictor/                  # Main app
│   ├── templates/              # HTML templates
│   │   └── predictor/
│   │       ├── login.html
│   │       ├── register.html
│   │       ├── dashboard.html
│   │       └── result.html
│   ├── static/                 # CSS, JS, images
│   ├── models.py               # Prediction model
│   ├── views.py                # Business logic
│   └── urls.py                 # URL routing
│
├── manage.py
├── .gitignore
└── README.md
```

---

## 🧠 How Prediction Works

The app uses a **rule-based pricing formula** (mock ML model):

```
Price = base_price + (sqft × 150) + (bedrooms × 20,000)
      + (bathrooms × 15,000) + (floors × 10,000)
      + parking_bonus + furnishing_bonus - age_penalty
      × random_factor (±5%)
```

> **Note:** This is a demonstration model. A real ML model (e.g., trained with scikit-learn on the Bangalore Housing Dataset) can be integrated.

---

## 📸 Screenshots

> Dashboard with glassmorphism UI and prediction history

---

## 🔮 Future Improvements

- [ ] Integrate a real ML model (scikit-learn / XGBoost)
- [ ] Add user authentication (Django Auth)
- [ ] Real dataset integration (Kaggle Housing Data)
- [ ] Deploy to cloud (Heroku / Render / AWS)
- [ ] Add map visualization with Leaflet.js

---

## 👤 Author

**Stavi** — Built for Hackathon 🚀

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
