# 🚀 College Club ExpenseWise: Smarter Spending for Brighter Futures ✨

Tired of messy spreadsheets and lost receipts? 💸 Say hello to ExpenseWise, the AI-powered expense management system built for college clubs like yours! 🎉

ExpenseWise isn't just another expense tracker. It's your club's financial co-pilot, bringing intelligence and ease to managing money, so you can focus on what truly matters: making your club awesome! 🌟

## 💡 The Problem We Solve

Running a college club is exhilarating, but let's face it, expense management can be a headache. Juggling receipts, chasing reimbursements, and guessing where your money is going? 😫 We've been there!

ExpenseWise is designed to banish those financial frustrations, giving you:
- Clarity: Know exactly where every penny is spent. 📊
- Control: Stay on top of your budget and avoid nasty surprises. 🎯
- Insights: Unlock the power of AI to make smarter financial decisions. 🧠
- Time Savings: Spend less time on paperwork, more time on club activities. ⏳

## ✨ ExpenseWise: Your AI-Powered Financial Superhero 🦸

ExpenseWise is packed with features to make expense management a breeze, and even... dare we say... fun? 😉

### 📸 Snap & Track Expenses in Seconds:
- Effortless Entry: Record expenses with just a few taps. Amount, description, category, date - done! ✅
- Receipt Magic: Upload receipt images and ditch the paper clutter. 🪄 Download them anytime, anywhere.
- Category Genius: Our AI learns! It intelligently suggests expense categories, so you don't have to guess. ✨

### 📊 Reports That Actually Make Sense:
- Crystal Clear Reports: Generate beautiful, detailed expense reports for any period. 📈
- CSV Power: Download reports in CSV format for deep dives and record-keeping mastery. 🤓
- Visual Insights: See your spending at a glance with interactive dashboards. Where's your money really going? 🤔

### 💰 Budgeting, Supercharged:
- Set It & Forget It (Almost!): Define budgets for different categories and stay on track. 🛤️
- Budget Guardian: ExpenseWise keeps an eye on your budget utilization and alerts you to potential overspending. 🚨

### 🧠 AI: Your Secret Weapon for Smart Spending:
- Category Autopilot: Zero-shot AI classification automagically suggests expense categories from descriptions. ✨
- Sentiment Decoder: Understand the feeling behind your spending with sentiment analysis. Are those "team bonding" expenses really bonding? 🤔
- Future Teller (Expense Prediction): Prophet time-series forecasting helps you see into the future and plan budgets like a pro. 🔮
- Anomaly Alert System: Isolation Forest anomaly detection flags unusual expenses. Was that $500 pizza order really for a club meeting? 🍕🤨
- Duplicate Detective: Semantic similarity AI sniffs out potential duplicate entries, keeping your data squeaky clean. 🔍
- Budget Risk Radar: ExpenseWise assesses budget risk per category, giving you a heads-up on potential trouble spots. 🚦
- Approval AI Assistant: Get AI-powered recommendations for expense approvals based on detected anomalies. Let the AI be the bad cop (sometimes). 👮‍♂️
- Vendor Vision: NER-powered vendor spending analysis reveals where your club's money is flowing. Are you really getting the best deals? 🧐

### 🔒 Admin Power & User Harmony:
- Secure User Fortress: Robust user registration and login system keeps things safe. 🛡️
- Admin Command Center: Admin roles for managing users, categories, and budgets with ease. 👑
- User Profiles for Everyone: Admins manage user profiles, keeping your club organized. 🧑‍🤝‍🧑

### ⚙️ Admin Settings - Tweak to Perfection:
- Category Control: Manage expense categories like a boss. Add, remove, rename, re-categorize - you're in charge! 🗂️
- User Management Made Easy: Configure and manage user accounts without the headache. 😌

### 🔐 Security First, Always:
- Password Fortress: Passwords are hashed like Fort Knox with hashlib.pbkdf2_hmac. 💪
- Input Sanity Check: Validation on user inputs prevents errors and keeps the bad guys out. 🛡️
- Admin-Only Zone: Sensitive admin features are locked down tighter than a drum. 🔒

## 🛠️ Built with ❤️ and Cutting-Edge Tech

ExpenseWise is powered by a stellar lineup of technologies:
- Streamlit: For the slick, interactive web app experience. 💻
- Python: The magic behind the scenes. 🐍
- SQLite: Your trusty local database sidekick for development. 🗄️
- Pandas: Data wrangling and reporting ninja. 🐼
- Pytesseract & Pillow: OCR dream team for receipt text extraction. 📜
- Hugging Face Transformers: AI brainpower for categorization and sentiment. 🤗
- Sentence Transformers: Semantic similarity superhero for duplicate detection. 🦸
- Prophet (Facebook): Time-traveling expense forecaster. 🕰️
- spaCy: Vendor-identifying NER wizard. 🧙‍♂️
- scikit-learn (Isolation Forest): Anomaly-spotting sentinel. 🌲
- hashlib: Password security guardian. 🛡️

## 🚀 Get ExpenseWise Running Locally - Fast!

### ⚙️ Prerequisites - Gear Up!
- Python 3.8+: Make sure you've got Python 3.8 or newer installed. Get it here 🐍
- pip: Python's package installer - usually comes with Python.
- Tesseract OCR Engine (Optional): For receipt image superpowers.
  - Ubuntu: sudo apt-get install tesseract-ocr 🐧
  - macOS: brew install tesseract (via Homebrew) 🍎
  - Windows: Download from here & add to PATH. 🪟

### ⬇️ Installation - Easy Peasy!

1. Clone the Repo:
```bash
git clone https://github.com/Starbuster2004/ExpenseWise-ECEsa.git
cd [your-repository-directory]
```
(Replace placeholders with your repo details)

2. Virtual Environment - Best Practice!
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows
```

3. Install the Goodies:
```bash
pip install -r requirements.txt
```

4. Launch ExpenseWise!
```bash
streamlit run app.py
```
Open your browser to http://localhost:8501 and prepare to be amazed! ✨

## 🤖 AI Deep Dive - How the Magic Works

ExpenseWise's AI features are designed to give you superpowers in expense management:
- ✨ Zero-Shot Category Sorcery: facebook/bart-large-mnli model predicts categories like a mind-reader.
- 🤔 Sentiment Oracle: distilbert-base-uncased-finetuned-sst-2-english model deciphers the emotional tone of your expenses.
- 🔮 Prophet Expense Gazer: Prophet library forecasts future spending trends with uncanny accuracy.
- 🚨 Isolation Forest Sentinel: scikit-learn's Isolation Forest algorithm stands guard, detecting expense anomalies.
- 🕵️ Duplicate Data Detective: all-MiniLM-L6-v2 Sentence Transformer model spots expense twins.
- 🏢 Vendor Visionary: spaCy's en_core_web_sm NER model reveals vendor spending patterns.

## 🛡️ Security - We've Got Your Back

ExpenseWise takes security seriously:
- 🔒 Password Armor: hashlib password hashing keeps your credentials safe and sound.
- ✅ Input Validation Shield: Input validation prevents common vulnerabilities and data mishaps.
- 👑 Admin Fortress: Admin features are locked down tighter than a bank vault.

## ⚠️ Important: Database Note for Real-World Use

For serious, persistent data storage (like for your actual club!), local SQLite is NOT recommended for deployment. You'll want to use a robust, cloud-based database. Check out options like облакобег, PlanetScale, or cloud SQLite services for deployment.

## 🤝 Join the ExpenseWise Community!

We welcome contributions! Want to make ExpenseWise even better?
1. Fork this repo. 🍴
2. Create your feature branch. 🌿
3. Code your heart out! 💻
4. Submit a Pull Request! 🚀

## 📜 License - Open Source Goodness

ExpenseWise is released under the MIT License. It's free, open source, and ready for you to explore and improve!

## 📞 Let's Connect!

Questions? Issues? Brilliant ideas? We'd love to hear from you! Open an issue on GitHub or send us an email.

---

ExpenseWise - Spend Smarter, Achieve More! ✨

