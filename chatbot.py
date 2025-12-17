import random
import re
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta

# ---------------- SETUP ---------------- #

nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
app = Flask(__name__)

# ---------------- LOAD TF-IDF MODEL ---------------- #

model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
responses_map = joblib.load("responses.pkl")

# ---------------- CONVERSATION CONTEXT ---------------- #

user_context = {
    "current_flow": None,
    "awaiting_booking_id": False,
    "awaiting_refund_reason": False,
    "awaiting_exchange_reason": False,
    "awaiting_anything_else": False,
    "awaiting_handoff_confirmation": False,
    "awaiting_tracking_id": False,   # 👈 NEW
    "booking_id": None,
    "refund_reason": None,
    "exchange_reason": None,
    "user_ticket_id": None,
    "chat_ended": False
}

def reset_user_context():
    user_context.update({
        "current_flow": None,
        "awaiting_booking_id": False,
        "awaiting_refund_reason": False,
        "awaiting_exchange_reason": False,
        "awaiting_anything_else": False,
        "awaiting_handoff_confirmation": False,
        "booking_id": None,
        "refund_reason": None,
        "exchange_reason": None,
        "user_ticket_id": None,
        "chat_ended": False

    })



# ---------------- RULE-BASED DATA ---------------- #

GREETINGS = ["hi", "hello", "hey", "good morning", "good evening"]
GREETING_RESPONSES = [
    "Hi 👋 How can I help you today?",
    "Hello! 😊 What can I assist you with?",
    "Hey there! I'm here to help."
]

EXCHANGE_KEYWORDS = ["exchange", "replace", "replacement", "change product"]

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

POLICY_NOTICE = (
    "📄 As per our company policy, refunds and exchanges are allowed "
    "within **7 days of delivery**.\n\n"
    "Please review the detailed policy on our website for complete information.\n\n"
    "A customer support associate will still review your request and reach out shortly."
)

ORDER_STATUSES = [
    ("Order confirmed", "Processing"),
    ("Shipped", "In transit"),
    ("Out for delivery", "Arriving today"),
    ("Delivered", "Completed")
]

FRUSTRATION_WORDS = ["angry", "ridiculous", "worst", "useless", "annoyed"]

PROACTIVE_MENU = (
    "I can help you with the following:\n\n"
    "1️⃣ Request a refund\n"
    "2️⃣ Exchange a product\n"
    "3️⃣ Track order status\n"
    "4️⃣ View company policies\n\n"
    "Please reply with 1, 2, 3, or 4."
)

VAGUE_INPUTS = ["help", "issue", "problem", "support"]

NEGATIVE_WORDS = [
    "angry", "frustrated", "frustrating", "ridiculous", "worst",
    "annoyed", "useless", "terrible", "disappointed", "hate"
]

POSITIVE_WORDS = [
    "thanks", "thank you", "great", "awesome", "good", "helpful"
]

EMPATHY_RESPONSES = [
    "I understand this can be frustrating 😔",
    "Sorry about the trouble you're facing.",
    "I get how this can be annoying — let me help."
]

POSITIVE_RESPONSES = [
    "Glad to hear that 😊",
    "Happy to help!",
    "You're welcome!"
]


HANDOFF_PROMPT = (
    "This request requires manual review.\n\n"
    "Would you like me to connect you to a customer support agent now?\n\n"
    "1️⃣ Yes, connect me\n"
    "2️⃣ No, I’ll continue with the bot"
)



# ---------------- HELPERS ---------------- #

def generate_ticket_id():
    return f"SUP-{random.randint(100000, 999999)}"


def get_mock_order_status():
    return random.choice(ORDER_STATUSES)


def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def is_booking_id(text):
    return bool(re.search(r"[A-Za-z0-9#]{5,}", text))

def format_refund_reasons():
    text = "Please select the reason for your return:\n\n"
    for i, reason in enumerate(REFUND_REASONS, start=1):
        text += f"{i}. {reason}\n"
    return text

def format_exchange_reasons():
    text = "Please select the reason for exchange:\n\n"
    for i, reason in enumerate(EXCHANGE_REASONS, start=1):
        text += f"{i}. {reason}\n"
    return text

def get_mock_delivery_date():
    return datetime.now() - timedelta(days=random.randint(1, 14))

def is_within_7_days(delivery_date):
    return (datetime.now() - delivery_date).days <= 7

def extract_model_hint(text, max_words=10):
    words = text.replace("\n", " ").split()
    short = " ".join(words[:max_words])
    return short.rstrip(".") + "."

# ---------------- MAIN CHATBOT LOGIC ---------------- #

def chatbot_reply(message):
    msg = message.lower().strip()


    # ---- Chat already ended ----
    if user_context.get("chat_ended", False):
        return (
            "🔚 This chat has ended.\n"
            "Please start a new chat if you need further assistance."
        )

    # ---- Greeting ----
    if msg in GREETINGS:
        return random.choice(GREETING_RESPONSES)

    # ---- Human handoff confirmation (HIGHEST PRIORITY) ----
    if user_context["awaiting_handoff_confirmation"]:
        if msg == "1":
            ticket_id = user_context.get("user_ticket_id", "N/A")

            user_context["chat_ended"] = True   # 👈 HARD STOP

            return (
                "📞 You’re now being connected to a customer support agent.\n\n"
                f"🧾 Ticket ID: {ticket_id}\n\n"
                "An agent will reach out to you shortly. Thank you for your patience.\n\n"
                "🔚 Chat ended.\n"
                "Start a new chat if you need further assistance."
            )

        if msg == "2":
            user_context["awaiting_handoff_confirmation"] = False
            return "No problem 😊 I’m here. How else can I help you?"

        return "Just let me know with **1** or **2** 😊"


    # ---- Follow-up after completion (Yes / No) ----
    if user_context["awaiting_anything_else"]:
        clean_msg = re.sub(r"[^\w\s]", "", msg)

        if any(word in clean_msg for word in ["yes", "yeah", "yep", "sure", "ok", "okay"]):
            user_context["awaiting_anything_else"] = False
            return "Sure 😊 How else can I assist you today?"

        if any(word in clean_msg for word in ["no", "nope", "nah", "no thanks", "thank you", "thanks"]):
            user_context["awaiting_anything_else"] = False
            reset_user_context()

            return (
                "You're welcome 😊 Thank you for contacting support. Have a great day! 👋\n\n"
                "🔚 Chat ended.\n"
                "Start a new chat if you need further assistance."
            )


        return "Please reply with Yes or No."

    # ---- Order status (simple intent) ----
    if "track" in msg or "order status" in msg:
        user_context["awaiting_tracking_id"] = True
        return (
            "Sure 😊 I can help track your order.\n\n"
            "Please share your booking or order ID."
        )



    # ---- Awaiting booking ID ----
    if user_context["awaiting_booking_id"]:
        if is_booking_id(message):
            user_context["booking_id"] = message
            user_context["awaiting_booking_id"] = False

            if user_context["current_flow"] == "refund":
                user_context["awaiting_refund_reason"] = True
                return f"Thanks for sharing your booking ID ({message}). ✅\n\n" + format_refund_reasons()

            if user_context["current_flow"] == "exchange":
                user_context["awaiting_exchange_reason"] = True
                return f"Thanks for sharing your booking ID ({message}). ✅\n\n" + format_exchange_reasons()

        return "Please share a valid booking or transaction ID."

    if any(word in msg for word in FRUSTRATION_WORDS):
        return (
            "I understand this can be frustrating 😔\n"
            "Let me help resolve this for you.\n\n"
            "Could you please share your booking ID?"
        )
    

    # ---- Awaiting tracking ID ----
    if user_context["awaiting_tracking_id"]:
        if is_booking_id(message):
            user_context["awaiting_tracking_id"] = False
            user_context["booking_id"] = message

            status, note = get_mock_order_status()

            user_context["awaiting_anything_else"] = True

            tracking_link = f"https://track.yourcompany.com/{message}"

            return (
                f"📦 Order Status for {message}: {status}\n"
                f"🚚 {note}\n\n"
                f"🔗 Track your order here:\n{tracking_link}\n\n"
                "Can I help you with anything else?"
            )


        return "Please share a valid booking or order ID."


    # ---- Awaiting refund reason ----
    # ---- Awaiting refund reason ----
    if user_context["awaiting_refund_reason"]:

        if not msg.isdigit():
            return "Please reply with a number between 1 and 5."

        choice = int(msg)

        if not (1 <= choice <= len(REFUND_REASONS)):
            return "Please select a valid option (1–5)."

        # Save refund reason
        user_context["refund_reason"] = REFUND_REASONS[choice - 1]
        user_context["awaiting_refund_reason"] = False
        user_context["awaiting_anything_else"] = True

        delivery_date = get_mock_delivery_date()
        days = (datetime.now() - delivery_date).days
        eligible = is_within_7_days(delivery_date)

        if eligible:
            ticket_id = generate_ticket_id()
            user_context["user_ticket_id"] = ticket_id

            return (
                "✅ Your refund request has been successfully initiated.\n\n"
                f"🧾 Ticket ID: {ticket_id}\n"
                f"📌 Refund Reason: {user_context['refund_reason']}\n\n"
                "You will receive updates shortly.\n\n"
                "Can I help you with anything else?"
            )


        else:
            ticket_id = generate_ticket_id()
            user_context["user_ticket_id"] = ticket_id

            user_context["awaiting_handoff_confirmation"] = True

            return (
                "⚠️ Your order is beyond the 7-day refund window.\n\n"
                f"🧾 Ticket ID: {ticket_id}\n"
                f"📌 Refund Reason: {user_context['refund_reason']}\n\n"
                "This request requires manual review.\n\n"
                "Would you like me to connect you to a customer support agent now?\n\n"
                "1️⃣ Yes, connect me\n"
                "2️⃣ No, I’ll continue with the bot"
            )

    # ---- Awaiting exchange reason ----
    # ---- Awaiting exchange reason ----
    if user_context["awaiting_exchange_reason"]:

        if not msg.isdigit():
            return "Please reply with a number between 1 and 5."

        choice = int(msg)

        if not (1 <= choice <= len(EXCHANGE_REASONS)):
            return "Please select a valid option (1–5)."

        # Save exchange reason
        user_context["exchange_reason"] = EXCHANGE_REASONS[choice - 1]
        user_context["awaiting_exchange_reason"] = False
        user_context["awaiting_anything_else"] = True

        delivery_date = get_mock_delivery_date()
        days = (datetime.now() - delivery_date).days
        eligible = is_within_7_days(delivery_date)

        if eligible:
            ticket_id = generate_ticket_id()
            user_context["user_ticket_id"] = ticket_id

            return (
                "✅ Your exchange request has been successfully initiated.\n\n"
                f"🧾 Ticket ID: {ticket_id}\n"
                f"📌 Exchange Reason: {user_context['exchange_reason']}\n\n"
                "Our logistics team will contact you for pickup and replacement.\n\n"
                "Can I help you with anything else?"
            )


        else:
            ticket_id = generate_ticket_id()
            user_context["user_ticket_id"] = ticket_id

            user_context["awaiting_handoff_confirmation"] = True

            return (
                "⚠️ Your order is beyond the 7-day exchange window.\n\n"
                f"🧾 Ticket ID: {ticket_id}\n"
                f"📌 exchange Reason: {user_context['exchange_reason']}\n\n"
                "This request requires manual review.\n\n"
                "Would you like me to connect you to a customer support agent now?\n\n"
                "1️⃣ Yes, connect me\n"
                "2️⃣ No, I’ll continue with the bot"
            )

    # ---- Exchange intent (rule-based) ----
    if any(word in msg for word in EXCHANGE_KEYWORDS):
        user_context["current_flow"] = "exchange"
        user_context["awaiting_booking_id"] = True
        return "Sure 😊 I can help with an exchange. Please share your booking or order ID."

    # ---- ML intent detection ----
    vec = vectorizer.transform([preprocess(message)])
    probs = model.predict_proba(vec)[0]

    if max(probs) < 0.4:
        return (
            "I'm not sure I understood that 😅\n\n"
            "I can help you with:\n"
            "1️⃣ Refund\n"
            "2️⃣ Exchange\n"
            "3️⃣ Order status\n"
            "4️⃣ Company policies\n\n"
            "Please choose an option."
        )

    tag = model.classes_[probs.argmax()]

    # ---- Refund intent ----
    if "refund" in tag:
        user_context["current_flow"] = "refund"
        user_context["awaiting_booking_id"] = True
        user_context["current_flow"] = "refund"
        user_context["awaiting_booking_id"] = True

        return (
            "Sure 😊 I can help you with your refund.\n\n"
            "Please share your booking or transaction ID so I can assist you further."
        )

    return random.choice(responses_map.get(tag, ["How can I assist you further?"]))


# ---------------- FLASK ROUTES ---------------- #

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    return jsonify({"reply": chatbot_reply(user_message)})

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    print("🌐 Customer Support Chatbot running at http://127.0.0.1:5000")
    app.run(debug=True)
