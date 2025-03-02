import streamlit as st
import sqlite3
import os
import pandas as pd
from datetime import datetime
import pytesseract
from PIL import Image
import io
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from prophet import Prophet
import spacy
from sklearn.ensemble import IsolationForest
import numpy as np
import hashlib

# Cache expensive operations for performance
@st.cache_resource
def load_classifier():
    try:
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception as e:
        st.error(f"Error loading classifier: {e}")
        return None

@st.cache_resource
def load_sentiment_model():
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None

@st.cache_resource
def load_sentence_transformer():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading sentence transformer: {e}")
        return None

@st.cache_resource
def load_ner_model():
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            # If model is not found, download it first
            import os
            os.system("python -m spacy download en_core_web_sm")
            return spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Error loading NER model: {e}")
        return None

# Database initialization
def init_db():
    try:
        conn = sqlite3.connect("expenses.db", check_same_thread=False)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS expenses 
                     (id INTEGER PRIMARY KEY, amount REAL, description TEXT, category TEXT, 
                      date TEXT, receipt_path TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS categories 
                     (id INTEGER PRIMARY KEY, name TEXT UNIQUE)''')
        c.execute('''CREATE TABLE IF NOT EXISTS budgets 
                     (id INTEGER PRIMARY KEY, category TEXT UNIQUE, amount REAL)''')
        c.execute("INSERT OR IGNORE INTO categories (name) VALUES ('Other')")
        c.execute('''CREATE TABLE IF NOT EXISTS users 
                     (id INTEGER PRIMARY KEY, 
                      username TEXT UNIQUE, 
                      password TEXT NOT NULL,
                      email TEXT,
                      phone TEXT,
                      is_admin BOOLEAN DEFAULT FALSE)''')
        # Add default admin user if not exists
        admin_password = hash_password("admin123")  # Using the hash_password function defined earlier
        c.execute("""INSERT OR IGNORE INTO users 
                    (username, password, is_admin) 
                    VALUES (?, ?, ?)""", 
                    ("admin", admin_password, True))
        conn.commit()
        return conn
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return None

# Database operations
def get_categories(conn):
    try:
        c = conn.cursor()
        c.execute("SELECT name FROM categories")
        return [row[0] for row in c.fetchall()] or ["Other"]
    except Exception as e:
        st.error(f"Error fetching categories: {e}")
        return ["Other"]

def add_category(conn, category):
    try:
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO categories (name) VALUES (?)", (category,))
        conn.commit()
    except Exception as e:
        st.error(f"Error adding category: {e}")

def remove_category(conn, category):
    try:
        if category != "Other":
            c = conn.cursor()
            c.execute("UPDATE expenses SET category = 'Other' WHERE category = ?", (category,))
            c.execute("DELETE FROM categories WHERE name = ?", (category,))
            c.execute("DELETE FROM budgets WHERE category = ?", (category,))
            conn.commit()
    except Exception as e:
        st.error(f"Error removing category: {e}")

def get_total_expenses(conn):
    try:
        c = conn.cursor()
        c.execute("SELECT SUM(amount) FROM expenses")
        total = c.fetchone()[0]
        return total if total else 0
    except Exception as e:
        st.error(f"Error fetching total expenses: {e}")
        return 0

def get_expenses_by_category(conn):
    try:
        c = conn.cursor()
        c.execute("SELECT category, SUM(amount) FROM expenses GROUP BY category")
        return dict(c.fetchall())
    except Exception as e:
        st.error(f"Error fetching expenses by category: {e}")
        return {}

def get_recent_expenses(conn, limit=5):
    try:
        c = conn.cursor()
        c.execute("SELECT * FROM expenses ORDER BY date DESC LIMIT ?", (limit,))
        return c.fetchall()
    except Exception as e:
        st.error(f"Error fetching recent expenses: {e}")
        return []

def get_budget(conn, category):
    try:
        c = conn.cursor()
        c.execute("SELECT amount FROM budgets WHERE category = ?", (category,))
        result = c.fetchone()
        return float(result[0]) if result else 0.0
    except Exception as e:
        st.error(f"Error fetching budget for {category}: {e}")
        return 0.0

def set_budget(conn, category, amount):
    try:
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO budgets (category, amount) VALUES (?, ?)", 
                  (category, float(amount)))
        conn.commit()
    except Exception as e:
        st.error(f"Error setting budget: {e}")

# AI functions
def categorize_description(classifier, description, categories):
    if not classifier:
        return "Other"
    try:
        result = classifier(description, categories, multi_label=False)
        return result["labels"][0]
    except Exception as e:
        st.error(f"Error categorizing description: {e}")
        return "Other"

def extract_text_from_receipt(image):
    try:
        text = pytesseract.image_to_string(Image.open(image))
        return text if text.strip() else "No text detected"
    except Exception as e:
        st.error(f"Error processing receipt: {e}")
        return "Error processing receipt"

def analyze_sentiment(sentiment_model, text):
    if not sentiment_model:
        return "Neutral"
    try:
        result = sentiment_model(text)
        return result[0]["label"].capitalize()
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return "Neutral"

def predict_expenses(conn):
    try:
        c = conn.cursor()
        c.execute("SELECT date, amount FROM expenses ORDER BY date")
        data = c.fetchall()
        if len(data) < 2:
            return "Not enough data for prediction"
        df = pd.DataFrame(data, columns=["ds", "y"])
        df["ds"] = pd.to_datetime(df["ds"])
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        return forecast[["ds", "yhat"]].tail(30).to_dict("records")
    except Exception as e:
        st.error(f"Error predicting expenses: {e}")
        return "Prediction failed"

def detect_anomalies(conn):
    try:
        c = conn.cursor()
        c.execute("SELECT amount FROM expenses")
        amounts = [row[0] for row in c.fetchall()]
        if not amounts:
            return []
        avg = np.mean(amounts)
        std = np.std(amounts)
        threshold = 2 * std
        return [a for a in amounts if abs(a - avg) > threshold]
    except Exception as e:
        st.error(f"Error detecting anomalies: {e}")
        return []

def detect_duplicates(conn, sentence_model):
    if not sentence_model:
        return []
    try:
        c = conn.cursor()
        c.execute("SELECT id, description FROM expenses")
        expenses = c.fetchall()
        if len(expenses) < 2:
            return []
        descriptions = [exp[1] for exp in expenses]
        embeddings = sentence_model.encode(descriptions, convert_to_tensor=True)
        duplicates = []
        for i in range(len(expenses)):
            for j in range(i + 1, len(expenses)):
                similarity = util.cos_sim(embeddings[i], embeddings[j]).item()
                if similarity > 0.9:
                    duplicates.append((expenses[i][0], expenses[j][0], similarity))
        return duplicates
    except Exception as e:
        st.error(f"Error detecting duplicates: {e}")
        return []

def assess_budget_risk(conn):
    try:
        c = conn.cursor()
        c.execute("SELECT category, SUM(amount) FROM expenses GROUP BY category")
        expenses = dict(c.fetchall())
        risks = {}
        for category in expenses:
            budget = get_budget(conn, category)
            if budget > 0:
                risk = expenses[category] / budget
                risks[category] = min(risk, 1.0)  # Cap at 100%
        return risks
    except Exception as e:
        st.error(f"Error assessing budget risk: {e}")
        return {}

def recommend_approval(conn):
    try:
        c = conn.cursor()
        c.execute("SELECT amount FROM expenses")
        amounts = [row[0] for row in c.fetchall()]
        if not amounts:
            return []
        iso_forest = IsolationForest(contamination=0.1)
        anomalies = iso_forest.fit_predict(np.array(amounts).reshape(-1, 1))
        return [amt for amt, pred in zip(amounts, anomalies) if pred == -1]
    except Exception as e:
        st.error(f"Error recommending approvals: {e}")
        return []

def vendor_spending_analysis(conn, ner_model):
    if not ner_model:
        return {}
    try:
        c = conn.cursor()
        c.execute("SELECT description, amount FROM expenses")
        data = c.fetchall()
        vendors = {}
        for desc, amount in data:
            doc = ner_model(desc)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    vendors[ent.text] = vendors.get(ent.text, 0) + amount
        return vendors
    except Exception as e:
        st.error(f"Error analyzing vendor spending: {e}")
        return {}

# Pages
def dashboard(conn):
    st.header("Dashboard")
    total = get_total_expenses(conn)
    st.metric("Total Expenses", f"₹{total:.2f}")
    expenses_by_category = get_expenses_by_category(conn)
    st.bar_chart(expenses_by_category)
    st.subheader("Recent Expenses")
    recent = get_recent_expenses(conn)
    for exp in recent:
        st.write(f"{exp[4]} - {exp[3]}: ₹{exp[1]}")

def enter_expense(conn, classifier):
    st.header("Enter Expense")
    with st.form("expense_form"):
        amount = st.number_input("Amount", min_value=0.01, step=0.01)
        description = st.text_input("Description")
        categories = get_categories(conn)
        category = st.selectbox("Category", categories)
        receipt = st.file_uploader("Upload Receipt", type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button("Submit")
        if submitted:
            if not description:
                st.error("Description is required")
            else:
                try:
                    receipt_path = None
                    if receipt:
                        os.makedirs("receipts", exist_ok=True)
                        receipt_path = f"receipts/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{receipt.name}"
                        with open(receipt_path, "wb") as f:
                            f.write(receipt.getbuffer())
                    auto_category = categorize_description(classifier, description, categories)
                    category = auto_category if auto_category != "Other" else category
                    c = conn.cursor()
                    c.execute("INSERT INTO expenses (amount, description, category, date, receipt_path) VALUES (?, ?, ?, ?, ?)",
                              (amount, description, category, datetime.now().strftime("%Y-%m-%d"), receipt_path))
                    conn.commit()
                    st.success("Expense added successfully!")
                except Exception as e:
                    st.error(f"Error saving expense: {e}")

def delete_receipt(conn, expense_id):
    """Delete receipt file and update database."""
    try:
        # First get the receipt path
        c = conn.cursor()
        c.execute("SELECT receipt_path FROM expenses WHERE id = ?", (expense_id,))
        result = c.fetchone()
        
        if result and result[0]:
            receipt_path = result[0]
            # Delete the physical file
            if os.path.exists(receipt_path):
                os.remove(receipt_path)
            
            # Update database to remove receipt path
            c.execute("UPDATE expenses SET receipt_path = NULL WHERE id = ?", (expense_id,))
            conn.commit()
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting receipt: {e}")
        return False

def receipts(conn):
    st.header("Receipts")
    
    # Get all receipts
    c = conn.cursor()
    c.execute("SELECT id, date, description, receipt_path FROM expenses WHERE receipt_path IS NOT NULL")
    receipts = c.fetchall()
    
    if not receipts:
        st.info("No receipts found in the database.")
        return
    
    # Display receipts in a more organized way
    for rec in receipts:
        expense_id, date, description, receipt_path = rec
        
        # Create a container for each receipt
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Date:** {date}")
                st.write(f"**Description:** {description}")
            
            with col2:
                if os.path.exists(receipt_path):
                    with open(receipt_path, "rb") as f:
                        st.download_button(
                            f"Download Receipt",
                            f,
                            file_name=os.path.basename(receipt_path),
                            key=f"download_{expense_id}"
                        )
                else:
                    st.warning("Receipt file missing")
            
            with col3:
                if st.button("Delete Receipt", key=f"delete_{expense_id}"):
                    if delete_receipt(conn, expense_id):
                        st.success("Receipt deleted successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to delete receipt")
        
        # Add a separator between receipts
        st.divider()
    
    # Add bulk delete option
    st.subheader("Bulk Operations")
    if st.button("Delete All Downloaded Receipts"):
        deleted_count = 0
        for rec in receipts:
            if delete_receipt(conn, rec[0]):
                deleted_count += 1
        
        if deleted_count > 0:
            st.success(f"Successfully deleted {deleted_count} receipts!")
            st.rerun()
        else:
            st.warning("No receipts were deleted")

def reports(conn):
    st.header("Reports")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    if start_date > end_date:
        st.error("Start date must be before end date")
        return
    try:
        c = conn.cursor()
        c.execute("SELECT * FROM expenses WHERE date BETWEEN ? AND ?", 
                  (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
        data = c.fetchall()
        if data:
            df = pd.DataFrame(data, columns=["ID", "Amount", "Description", "Category", "Date", "Receipt Path"])
            # Format the Amount column to show Rupee symbol
            df['Amount'] = df['Amount'].apply(lambda x: f"₹{x:.2f}")
            st.dataframe(df)
            csv = df.to_csv(index=False)
            st.download_button("Download Report", csv, "report.csv", "text/csv")
        else:
            st.info("No expenses found in this date range")
    except Exception as e:
        st.error(f"Error generating report: {e}")

def budgeting(conn):
    if not st.session_state.get("admin", False):
        st.error("Admin access required")
        return
    st.header("Budgeting")
    categories = get_categories(conn)
    for category in categories:
        current_budget = float(get_budget(conn, category))
        budget = st.number_input(
            f"Budget for {category}", 
            min_value=0.0,
            value=current_budget,
            step=0.01,
            format="%.2f",
            key=category
        )
        if st.button(f"Set Budget for {category}", key=f"btn_{category}"):
            set_budget(conn, category, float(budget))
            st.success(f"Budget for {category} set to ₹{budget:.2f}")

def ai_insights(conn, classifier, sentiment_model, sentence_model, ner_model):
    st.header("AI Insights")
    st.subheader("Predicted Expenses")
    forecast = predict_expenses(conn)
    if isinstance(forecast, list):
        st.line_chart(pd.DataFrame(forecast).set_index("ds")["yhat"])
    else:
        st.write(forecast)

    st.subheader("Anomalies")
    anomalies = detect_anomalies(conn)
    st.write(anomalies if anomalies else "No anomalies detected")

    st.subheader("Potential Duplicates")
    duplicates = detect_duplicates(conn, sentence_model)
    for dup in duplicates:
        st.write(f"Expense IDs {dup[0]} and {dup[1]} - Similarity: {dup[2]:.2f}")

    st.subheader("Budget Overrun Risk")
    risks = assess_budget_risk(conn)
    for cat, risk in risks.items():
        st.write(f"{cat}: {risk*100:.2f}%")

    st.subheader("Approval Recommendations")
    anomalies = recommend_approval(conn)
    st.write(anomalies if anomalies else "No anomalies for review")

    st.subheader("Vendor Spending")
    vendors = vendor_spending_analysis(conn, ner_model)
    if vendors:
        # Format vendor spending with Rupee symbol
        formatted_vendors = {k: f"₹{v:.2f}" for k, v in vendors.items()}
        st.bar_chart(vendors)  # Use original values for chart
        # Display formatted values in a table
        st.write("Vendor-wise Spending:")
        for vendor, amount in formatted_vendors.items():
            st.write(f"{vendor}: {amount}")

def admin_settings(conn):
    if not st.session_state.get("admin", False):
        st.error("Admin access required")
        return
    st.header("Admin Settings")
    new_category = st.text_input("New Category")
    if st.button("Add Category"):
        if new_category:
            add_category(conn, new_category)
            st.success(f"Category '{new_category}' added")
        else:
            st.error("Category name cannot be empty")
    
    category_to_remove = st.selectbox("Remove Category", [c for c in get_categories(conn) if c != "Other"])
    if st.button("Remove Category"):
        remove_category(conn, category_to_remove)
        st.success(f"Category '{category_to_remove}' removed")

def add_user(conn, username, password, email, phone):
    try:
        c = conn.cursor()
        hashed_password = hash_password(password)
        c.execute("""INSERT INTO users 
                    (username, password, email, phone, is_admin) 
                    VALUES (?, ?, ?, ?, ?)""", 
                    (username, hashed_password, email, phone, False))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error adding user: {e}")
        return False

def verify_user(conn, username, password):
    try:
        c = conn.cursor()
        c.execute("SELECT password, is_admin FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        if result and verify_password(result[0], password):
            return True, result[1]  # Returns (success, is_admin)
        return False, False
    except Exception as e:
        st.error(f"Error verifying user: {e}")
        return False, False

def get_user_profiles(conn):
    try:
        c = conn.cursor()
        c.execute("SELECT username, email, phone FROM users WHERE is_admin = FALSE")
        return c.fetchall()
    except Exception as e:
        st.error(f"Error fetching user profiles: {e}")
        return []

def get_user_by_username(conn, username):
    """Fetch user details by username."""
    try:
        c = conn.cursor()
        c.execute("""SELECT username, email, phone 
                    FROM users 
                    WHERE username = ? AND is_admin = FALSE""", (username,))
        return c.fetchone()
    except Exception as e:
        st.error(f"Error fetching user details: {e}")
        return None

def update_user(conn, username, new_email, new_phone, new_password=None):
    """Update user information."""
    try:
        c = conn.cursor()
        if new_password:
            hashed_password = hash_password(new_password)
            c.execute("""UPDATE users 
                        SET email = ?, phone = ?, password = ?
                        WHERE username = ? AND is_admin = FALSE""", 
                     (new_email, new_phone, hashed_password, username))
        else:
            c.execute("""UPDATE users 
                        SET email = ?, phone = ?
                        WHERE username = ? AND is_admin = FALSE""", 
                     (new_email, new_phone, username))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error updating user: {e}")
        return False

def delete_user(conn, username):
    """Delete a user account."""
    try:
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE username = ? AND is_admin = FALSE", 
                 (username,))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error deleting user: {e}")
        return False

def admin_dashboard(conn):
    st.header("Admin Dashboard")
    
    # User Management Section
    st.subheader("User Management")
    
    # Add New User Form
    with st.expander("Add New User"):
        with st.form("add_user_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_email = st.text_input("Email")
            new_phone = st.text_input("Phone")
            
            if st.form_submit_button("Add User"):
                if all([new_username, new_password, new_email, new_phone]):
                    valid, message = validate_user_input(new_username, new_password, new_email, new_phone)
                    if valid:
                        if add_user(conn, new_username, new_password, new_email, new_phone):
                            st.success(f"User {new_username} added successfully")
                    else:
                        st.error(message)
                else:
                    st.error("All fields are required")
    
    # Edit Users Section
    st.subheader("Edit Users")
    users = get_user_profiles(conn)
    if users:
        # Display users in a dataframe
        df = pd.DataFrame(users, columns=["Username", "Email", "Phone"])
        st.dataframe(df)
        
        # User editing interface
        with st.expander("Edit User"):
            # Select user to edit
            usernames = [user[0] for user in users]
            selected_user = st.selectbox("Select User to Edit", usernames)
            
            if selected_user:
                user_data = get_user_by_username(conn, selected_user)
                if user_data:
                    with st.form("edit_user_form"):
                        st.write(f"Editing user: {selected_user}")
                        new_email = st.text_input("New Email", value=user_data[1])
                        new_phone = st.text_input("New Phone", value=user_data[2])
                        new_password = st.text_input("New Password (leave blank to keep current)", type="password")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.form_submit_button("Update User"):
                                valid, message = validate_user_input(
                                    selected_user,
                                    new_password if new_password else "dummy",
                                    new_email,
                                    new_phone
                                )
                                if valid:
                                    if update_user(conn, selected_user, new_email, new_phone, 
                                                 new_password if new_password else None):
                                        st.success(f"User {selected_user} updated successfully")
                                else:
                                    st.error(message)
                        
                        with col2:
                            if st.form_submit_button("Delete User", type="primary"):
                                if delete_user(conn, selected_user):
                                    st.success(f"User {selected_user} deleted successfully")
                                    st.rerun()
    else:
        st.info("No regular users registered yet")

    # Display current user statistics
    st.subheader("User Statistics")
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE is_admin = FALSE")
    user_count = c.fetchone()[0]
    st.metric("Total Users", user_count)

# Main app
def main():
    st.title("College Club Expense Management System")
    
    # Initialize session state
    if "admin" not in st.session_state:
        st.session_state.admin = False
    if "user" not in st.session_state:
        st.session_state.user = None

    # Database connection
    conn = init_db()
    if not conn:
        st.error("Failed to initialize database. Please try again.")
        return

    # Load AI models
    classifier = load_classifier()
    sentiment_model = load_sentiment_model()
    sentence_model = load_sentence_transformer()
    ner_model = load_ner_model()

    # Sidebar navigation with access control
    available_pages = ["Login/Logout"]
    if st.session_state.user:
        available_pages.extend(["Dashboard", "Enter Expense"])
        if st.session_state.admin:
            available_pages.extend(["Admin Dashboard", "Receipts", "Reports", 
                                  "Budgeting", "AI Insights", "Admin Settings"])
    
    page = st.sidebar.selectbox("Navigate", available_pages)

    # Show login status and username
    if st.session_state.user:
        st.sidebar.write(f"Logged in as: {st.session_state.user}")
        if st.session_state.admin:
            st.sidebar.write("(Admin)")

    # Login/Logout page
    if page == "Login/Logout":
        if st.session_state.user:
            if st.button("Logout"):
                st.session_state.admin = False
                st.session_state.user = None
                st.success("Logged out successfully")
        else:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                success, is_admin = verify_user(conn, username, password)
                if success:
                    st.session_state.user = username
                    st.session_state.admin = is_admin
                    st.success(f"Logged in as {'admin' if is_admin else 'user'}")
                else:
                    st.error("Invalid credentials")

    # Page routing with access control
    elif page == "Dashboard" and st.session_state.user:
        dashboard(conn)
    elif page == "Enter Expense" and st.session_state.user:
        enter_expense(conn, classifier)
    elif page == "Admin Dashboard" and st.session_state.admin:
        admin_dashboard(conn)
    elif page == "Receipts" and st.session_state.admin:
        receipts(conn)
    elif page == "Reports" and st.session_state.admin:
        reports(conn)
    elif page == "Budgeting" and st.session_state.admin:
        budgeting(conn)
    elif page == "AI Insights" and st.session_state.admin:
        ai_insights(conn, classifier, sentiment_model, sentence_model, ner_model)
    elif page == "Admin Settings" and st.session_state.admin:
        admin_settings(conn)

def hash_password(password):
    """Create a secure hash of the password."""
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000  # Number of iterations
    )
    return salt + key

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user."""
    salt = stored_password[:32]
    stored_key = stored_password[32:]
    key = hashlib.pbkdf2_hmac(
        'sha256',
        provided_password.encode('utf-8'),
        salt,
        100000  # Number of iterations
    )
    return key == stored_key

def validate_user_input(username, password, email=None, phone=None):
    """Validate user input for registration and login."""
    try:
        # Username validation
        if not username or len(username.strip()) < 3:
            return False, "Username must be at least 3 characters long"
        
        # Password validation (only if provided)
        if password and password != "dummy":
            if len(password) < 6:
                return False, "Password must be at least 6 characters long"
            if not any(c.isupper() for c in password):
                return False, "Password must contain at least one uppercase letter"
            if not any(c.islower() for c in password):
                return False, "Password must contain at least one lowercase letter"
            if not any(c.isdigit() for c in password):
                return False, "Password must contain at least one number"
        
        # Email validation (if provided)
        if email:
            email = email.strip()
            if not '@' in email or not '.' in email:
                return False, "Invalid email format"
            if len(email.split('@')[0]) < 1 or len(email.split('@')[1]) < 3:
                return False, "Invalid email format"
        
        # Phone validation (if provided)
        if phone:
            phone = phone.strip()
            # Remove common phone number formatting characters
            cleaned_phone = phone.replace('-', '').replace(' ', '').replace('(', '').replace(')', '').replace('+', '')
            if not cleaned_phone.isdigit():
                return False, "Phone number should contain only digits, spaces, and these special characters: + - ( )"
            if len(cleaned_phone) < 10:
                return False, "Phone number is too short"
        
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

if __name__ == "__main__":
    main()