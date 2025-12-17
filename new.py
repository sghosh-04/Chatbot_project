import random
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
import re

nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

# ---------- Load ML components ----------
model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
responses_map = joblib.load("responses.pkl")

# ---------- Rule-based phrases ----------
GREETINGS = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
THANKS = ["thanks", "thank you", "thx", "appreciate it"]
GOODBYES = ["bye", "goodbye", "exit", "quit", "see you"]

GREETING_RESPONSES = [
    "Hi 👋 How can I help you today?",
    "Hello! 😊 What can I assist you with?",
    "Hey there! I'm here to help."
]

THANKS_RESPONSES = [
    "You're welcome! 😊",
    "Happy to help!",
    "Anytime! Let me know if you need anything else."
]

GOODBYE_RESPONSES = [
    "Goodbye! Have a great day 🌟",
    "Thanks for contacting support. Take care!",
    "See you soon! 👋"
]

# ---------- Company policy message ----------
POLICY_MESSAGE = (
    "Sure! 📄 Here's a quick overview of our company policies:\n"
    "- Orders can be cancelled within 24 hours of placement\n"
    "- Refunds are processed within 5–7 business days\n"
    "- Warranty coverage depends on the product category\n\n"
    "For detailed information, please visit our full policy page on the website."
)

# ---------- Booking ID detection ----------
def extract_booking_id(text):
    match = re.search(r"\b[A-Z0-9]{6,}\b", text.upper())
    return match.group() if match else None

# ---------- Preprocessing ----------
def preprocess(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# ---------- Main chatbot logic ----------
def chatbot_response(msg):
    msg_clean = msg.lower().strip()

    # ---- Rule-based friendliness ----
    if msg_clean in GREETINGS:
        return random.choice(GREETING_RESPONSES)

    if msg_clean in THANKS:
        return random.choice(THANKS_RESPONSES)

    if msg_clean in GOODBYES:
        return random.choice(GOODBYE_RESPONSES)

    # ---- Policy-related ----
    if "policy" in msg_clean or "terms" in msg_clean or "refund policy" in msg_clean:
        return POLICY_MESSAGE

    # ---- Order / Product handling ----
    if any(word in msg_clean for word in ["order", "booking", "product", "refund", "delivery"]):
        booking_id = extract_booking_id(msg_clean)
        if booking_id:
            return (
                f"Thanks for sharing your booking ID ({booking_id}). "
                "Let me check the details and assist you further."
            )
        else:
            return "Could you please provide your booking or order ID so I can assist you better?"

    # ---- ML-based intent detection ----
    processed = preprocess(msg)
    vec = vectorizer.transform([processed])
    probs = model.predict_proba(vec)[0]

    if max(probs) < 0.4:
        return (
            "I'm not completely sure I understood that 🤔. "
            "Could you please rephrase or provide more details?"
        )

    tag = model.classes_[probs.argmax()]
    return random.choice(responses_map.get(tag, ["How can I help you further?"]))

# ---------- Chat loop ----------
print("🤖 Customer Support Bot")
print(random.choice(GREETING_RESPONSES))
print("Type 'quit' to exit\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Bot:", random.choice(GOODBYE_RESPONSES))
        break

    print("Bot:", chatbot_response(user_input))
