import random
import re
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta

nltk.download("punkt")
nltk.download("wordnet")

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

# ---------- LOAD MODEL ----------
model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
responses_map = joblib.load("responses.pkl")

# ---------- CONTEXT ----------
user_context = {}

def reset_context():
    user_context.clear()
    user_context.update({
        "current_flow": None,     # refund | exchange | track
        "awaiting_booking_id": False,
        "awaiting_reason": False,
        "awaiting_anything_else": False,
        "awaiting_handoff": False,
        "booking_id": None,
        "reason": None,
        "ticket_id": None,
        "chat_ended": False
    })

reset_context()

# ---------- DATA ----------
GREETINGS = ["hi", "hello", "hey", "good morning", "good evening"]

REFUND_REASONS = [
    "Received a damaged or defective product",
    "Product did not match the description",
    "Wrong item delivered",
    "Delivery was delayed",
    "Changed my mind / no longer needed the product"
]

EXCHANGE_REASONS = [
    "Size does not fit",
    "Wrong color or variant received",
    "Product damaged",
    "Want a different model",
    "Received incorrect item"
]

ORDER_STATUSES = [
    ("Order confirmed", "Processing"),
    ("Shipped", "In transit"),
    ("Out for delivery", "Arriving today"),
    ("Delivered", "Completed")
]

# ---------- HELPERS ----------
def preprocess(text):
    return " ".join(
        lemmatizer.lemmatize(w)
        for w in nltk.word_tokenize(text.lower())
    )

def is_booking_id(text):
    return bool(re.search(r"[A-Za-z0-9#]{5,}", text))

def generate_ticket():
    return f"SUP-{random.randint(100000, 999999)}"

def get_delivery_date():
    return datetime.now() - timedelta(days=random.randint(1, 14))

def within_7_days(date):
    return (datetime.now() - date).days <= 7

def order_status():
    return random.choice(ORDER_STATUSES)

def format_options(options):
    return "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))

# ---------- CHATBOT ----------
def chatbot_reply(message):
    msg = message.lower().strip()

    # 🔚 CHAT ENDED
    if user_context["chat_ended"]:
        return "🔚 This chat has ended.\nPlease start a new chat for further assistance."

    # 👋 GREETING
    if msg in GREETINGS:
        return random.choice([
            "Hi 👋 How can I help you today?",
            "Hello! 😊 What can I assist you with?",
            "Hey there! I'm here to help."
        ])

    # ---------- HIGH PRIORITY INTENTS ----------
    if "track" in msg:
        user_context.update({
            "current_flow": "track",
            "awaiting_booking_id": True
        })
        return "📦 Sure! Please share your booking or order ID to track your order."

    if "refund" in msg:
        user_context.update({
            "current_flow": "refund",
            "awaiting_booking_id": True
        })
        return "Sure 😊 I can help with a refund. Please share your booking ID."

    if "exchange" in msg:
        user_context.update({
            "current_flow": "exchange",
            "awaiting_booking_id": True
        })
        return "Sure 😊 I can help with an exchange. Please share your booking ID."

    # ---------- BOOKING ID ----------
    if user_context["awaiting_booking_id"]:
        if not is_booking_id(message):
            return "Please share a valid booking or order ID."

        user_context["booking_id"] = message
        user_context["awaiting_booking_id"] = False

        if user_context["current_flow"] == "track":
            status, note = order_status()
            user_context["awaiting_anything_else"] = True
            return (
                f"📦 Order Status for {message}: {status}\n"
                f"🚚 {note}\n"
                f"🔗 Track here: https://tracking.company.com/{message}\n\n"
                "Can I help you with anything else?"
            )

        if user_context["current_flow"] == "refund":
            user_context["awaiting_reason"] = True
            return (
                f"Thanks for sharing your booking ID ({message}). ✅\n\n"
                "Please select the reason for refund:\n\n"
                + format_options(REFUND_REASONS)
            )

        if user_context["current_flow"] == "exchange":
            user_context["awaiting_reason"] = True
            return (
                f"Thanks for sharing your booking ID ({message}). ✅\n\n"
                "Please select the reason for exchange:\n\n"
                + format_options(EXCHANGE_REASONS)
            )

    # ---------- REASON ----------
    if user_context["awaiting_reason"]:
        if not msg.isdigit():
            return "Please reply with a number from the list."

        choice = int(msg) - 1
        options = REFUND_REASONS if user_context["current_flow"] == "refund" else EXCHANGE_REASONS

        if choice not in range(len(options)):
            return "Please select a valid option."

        user_context["reason"] = options[choice]
        user_context["awaiting_reason"] = False

        delivered = get_delivery_date()
        eligible = within_7_days(delivered)
        ticket = generate_ticket()
        user_context["ticket_id"] = ticket

        if eligible:
            user_context["awaiting_anything_else"] = True
            return (
                f"✅ Your {user_context['current_flow']} request has been initiated.\n\n"
                f"🧾 Ticket ID: {ticket}\n"
                f"📌 Reason: {user_context['reason']}\n\n"
                "You will receive updates shortly.\n\n"
                "Can I help you with anything else?"
            )
        else:
            user_context["awaiting_handoff"] = True
            return (
                f"⚠️ Your order is beyond the 7-day window.\n\n"
                f"🧾 Ticket ID: {ticket}\n"
                f"📌 Reason: {user_context['reason']}\n\n"
                "Would you like me to connect you to a support agent?\n\n"
                "1️⃣ Yes\n"
                "2️⃣ No"
            )

    # ---------- HANDOFF ----------
    if user_context["awaiting_handoff"]:
        if msg == "1":
            user_context["chat_ended"] = True
            return (
                "📞 You’re now being connected to a customer support agent.\n\n"
                f"🧾 Ticket ID: {user_context['ticket_id']}\n\n"
                "Thank you for your patience.\n\n"
                "🔚 Chat ended. Start a new chat anytime."
            )
        if msg == "2":
            user_context["awaiting_handoff"] = False
            return "No problem 😊 How else can I help?"

        return "Please reply with **1** or **2**."

    # ---------- ANYTHING ELSE ----------
    if user_context["awaiting_anything_else"]:
        if msg in ["yes", "yeah", "yep", "sure"]:
            user_context["awaiting_anything_else"] = False
            return "Sure 😊 What can I help you with?"
        if msg in ["no", "nope", "nah", "thanks", "thank you"]:
            user_context["chat_ended"] = True
            return "You're welcome 😊 Chat ended. Start a new chat anytime!"

    # ---------- ML FALLBACK ----------
    vec = vectorizer.transform([preprocess(message)])
    probs = model.predict_proba(vec)[0]

    if max(probs) < 0.4:
        return (
            "I can help you with:\n"
            "1️⃣ Refund\n"
            "2️⃣ Exchange\n"
            "3️⃣ Track order\n\n"
            "Please tell me what you’d like to do."
        )

    tag = model.classes_[probs.argmax()]
    return random.choice(responses_map.get(tag, ["How can I help you?"]))

# ---------- ROUTES ----------
@app.route("/")
def home():
    reset_context()   # RESET ON REFRESH
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    return jsonify({"reply": chatbot_reply(request.json.get("message", ""))})

if __name__ == "__main__":
    app.run(debug=True)
