AI Powered Customer Support Chatbot

A production-style customer support chatbot built using Machine Learning + rule-based logic to handle real customer workflows such as refunds, exchanges, order tracking, and human escalation.
This project focuses on how real support systems work, not just how chatbots reply.

Features:
Intelligent Intent Detection
Uses TF IDF + ML classification for intent recognition

Supports:
Refund requests
Exchange requests
Order tracking
Policy inquiries

Context-Aware Conversations:
Maintains conversation state across messages
Understands what information is required next
Prevents invalid or out of sequence inputs

Order Tracking With Verification:
Requires booking or order ID before showing status
Displays realistic order states
Provides a simulated tracking link

Refund & Exchange Workflows:
Multi-step flow
Intent detection
Booking ID verification
Reason selection
Policy eligibility check 
Automatically initiates eligible requests
Escalates ineligible requests for manual review

Policy-Aware Decision Logic:
Enforces company rules 
Differentiates between:
1. Auto-approved cases
2. Manual review cases

Ticket Generation & Human Escalation:
Generates unique support ticket IDs
Seamless handoff to human agents
Proper chat termination after escalation

Professional Dark-Themed UI:
Clean, modern interface
Built with HTML, CSS, JavaScript
Designed for real-world use and portfolio presentation

Architecture Overview:
User Input
   ↓
Intent Detection (ML)
   ↓
Conversation State Manager
   ↓
Policy & Business Rules
   ↓
Auto Resolution / Human Escalation

A hybrid approach ensures reliability, accuracy, and realistic behavior.

Tech Stack:
Backend: Python, Flask
Machine Learning: TF-IDF, Scikit-learn
NLP: NLTK
Frontend: HTML, CSS, JavaScript

Project Structure:
chatbot_project/
├── chatbot.py              
├── intent_model.pkl        
├── tfidf_vectorizer.pkl    
├── responses.pkl           
├── templates/
│   └── index.html          
├── README.md
