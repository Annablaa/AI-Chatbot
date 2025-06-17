import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random
from typing import List, Dict
import os
from datetime import datetime

class BookRecommenderChatbot:
    def __init__(self, csv_file_path: str, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Load and process books data
        self.books_df = self.load_books_data(csv_file_path)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.book_embeddings = self.create_embeddings()
        
        # Chat personality and context
        self.conversation_history = []
        self.user_preferences = {}
        self.greeting_messages = [
            "Hello! I'm your personal book recommender! 📚 What kind of literary adventure are you in the mood for today?",
            "Hi there, fellow bookworm! 🐛 Ready to discover your next great read?",
            "Hey! I'm here to help you find amazing books! What's catching your interest lately?",
        ]
        
        print(f"✨ Loaded {len(self.books_df)} books into my recommendation system!")
        print(random.choice(self.greeting_messages))
    
    def load_books_data(self, csv_file_path: str) -> pd.DataFrame:
        """Load and clean books data from CSV"""
        try:
            df = pd.read_csv(csv_file_path)
            
            # Ensure required columns exist
            required_columns = ['title', 'author', 'description']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean data
            df = df.dropna(subset=required_columns)
            df['combined_text'] = df['title'] + ' ' + df['author'] + ' ' + df['description']
            df['title'] = df['title'].str.strip()
            df['author'] = df['author'].str.strip()
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading books data: {str(e)}")
    
    def create_embeddings(self) -> np.ndarray:
        print("🔄 Creating book embeddings... This might take a moment!")
        embeddings = self.vectorizer.fit_transform(self.books_df['combined_text'])
        print("✅ Embeddings created successfully!")
        return embeddings.toarray()
    
    def find_similar_books(self, query: str, num_recommendations: int = 5) -> List[Dict]:
        """Find books similar to the query using cosine similarity"""
        # Transform query using the same vectorizer
        query_embedding = self.vectorizer.transform([query]).toarray()
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.book_embeddings)[0]
        
        # Get top similar books
        similar_indices = np.argsort(similarities)[::-1][:num_recommendations]
        
        recommendations = []
        for idx in similar_indices:
            if similarities[idx] > 0.01:  # Minimum similarity threshold
                book = self.books_df.iloc[idx]
                recommendations.append({
                    'title': book['title'],
                    'author': book['author'],
                    'description': book['description'],
                    'similarity_score': similarities[idx]
                })
        
        return recommendations
    
    def search_by_title(self, title_query: str) -> List[Dict]:
        """Search for books by title match"""
        title_matches = self.books_df[
            self.books_df['title'].str.contains(title_query, case=False, na=False)
        ]
        
        results = []
        for _, book in title_matches.head(10).iterrows():
            results.append({
                'title': book['title'],
                'author': book['author'],
                'description': book['description'],
                'match_type': 'title_match'
            })
        
        return results
    
    def get_recommendations_by_author(self, author_query: str) -> List[Dict]:
        """Get books by specific author"""
        author_matches = self.books_df[
            self.books_df['author'].str.contains(author_query, case=False, na=False)
        ]
        
        results = []
        for _, book in author_matches.head(8).iterrows():
            results.append({
                'title': book['title'],
                'author': book['author'],
                'description': book['description'],
                'match_type': 'author_match'
            })
        
        return results
    
    def generate_human_response(self, recommendations: List[Dict], user_input: str) -> str:
        """Generate a human-like response using Gemini API"""
        if not recommendations:
            return self.generate_no_results_response(user_input)
        
        # Prepare context for Gemini
        books_context = ""
        for i, book in enumerate(recommendations[:3], 1):
            books_context += f"{i}. '{book['title']}' by {book['author']}\n"
            books_context += f"   Description: {book['description'][:150]}...\n\n"
        
        prompt = f"""
        You are an enthusiastic, knowledgeable book recommender chatbot with a warm, friendly personality. 
        A user asked: "{user_input}"
        
        Here are the top book recommendations I found:
        {books_context}
        
        Please respond in a conversational, engaging way that:
        1. Acknowledges their request warmly
        2. Introduces the recommendations with enthusiasm
        3. Briefly explains why these books might appeal to them
        4. Uses a friendly, bookish tone with occasional emojis
        5. Asks a follow-up question to keep the conversation going
        6. Keeps the response concise but engaging (2-3 paragraphs max)
        
        Don't just list the books - make it feel like a conversation with a knowledgeable friend!
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return self.fallback_response(recommendations, user_input)
    
    def generate_no_results_response(self, user_input: str) -> str:
        """Generate response when no books are found"""
        responses = [
            f"Hmm, I couldn't find any books matching '{user_input}' in my collection. Could you try describing what kind of story or genre you're interested in? I'd love to help you discover something new! 🔍",
            f"I don't have any exact matches for '{user_input}', but don't worry! Tell me more about what you enjoy reading - maybe a favorite genre, mood, or theme? I'm sure I can find something you'll love! 📖",
            f"No direct matches for '{user_input}' came up, but that just means we get to explore! What kind of books usually capture your attention? Mystery, romance, sci-fi, literary fiction? Let's find your next great read! ✨"
        ]
        return random.choice(responses)
    
    def fallback_response(self, recommendations: List[Dict], user_input: str) -> str:
        """Fallback response if Gemini API fails"""
        if not recommendations:
            return self.generate_no_results_response(user_input)
        
        response = f"Great choice! Based on '{user_input}', here are some fantastic books I think you'll enjoy:\n\n"
        
        for i, book in enumerate(recommendations[:3], 1):
            response += f"📚 **{book['title']}** by {book['author']}\n"
            response += f"   {book['description'][:120]}...\n\n"
        
        response += "Would you like more details about any of these, or shall I find something else for you? 😊"
        return response
    
    def detect_intent(self, user_input: str) -> Dict:
        """Detect user intent and extract relevant information"""
        user_input_lower = user_input.lower()
        
        # Check for author queries
        author_patterns = [
            r"books? by (.+)", r"(.+)'s books?", r"author (.+)",
            r"written by (.+)", r"from (.+) author"
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                return {
                    'intent': 'author_search',
                    'query': match.group(1).strip()
                }
        
        # Check for title queries
        title_patterns = [
            r"book called (.+)", r"title (.+)", r"book titled (.+)",
            r"looking for (.+)", r"find (.+)"
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                return {
                    'intent': 'title_search',
                    'query': match.group(1).strip()
                }
        
        # Default to similarity search
        return {
            'intent': 'similarity_search',
            'query': user_input
        }
    
    def chat(self, user_input: str) -> str:
        """Main chat function"""
        if not user_input.strip():
            return "I'm here and ready to help! What kind of book are you looking for? 😊"
        
        # Add to conversation history
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'user': user_input,
            'type': 'user_input'
        })
        
        # Handle special commands
        if user_input.lower() in ['hello', 'hi', 'hey']:
            return random.choice(self.greeting_messages)
        
        if user_input.lower() in ['bye', 'goodbye', 'exit', 'quit']:
            return "Happy reading! Come back anytime you need more book recommendations! 📚✨"
        
        # Detect intent and get recommendations
        intent_data = self.detect_intent(user_input)
        
        if intent_data['intent'] == 'author_search':
            recommendations = self.get_recommendations_by_author(intent_data['query'])
        elif intent_data['intent'] == 'title_search':
            recommendations = self.search_by_title(intent_data['query'])
        else:
            recommendations = self.find_similar_books(intent_data['query'])
        
        # Generate human-like response
        response = self.generate_human_response(recommendations, user_input)
        
        # Add to conversation history
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'bot': response,
            'recommendations': len(recommendations),
            'type': 'bot_response'
        })
        
        return response
    
    def start_interactive_session(self):
        """Start an interactive chat session"""
        print("\n" + "="*60)
        print("📚 BOOK RECOMMENDER CHATBOT 📚")
        print("="*60)
        print("Type 'quit' or 'exit' to end the conversation")
        print("-"*60)
        
        while True:
            try:
                user_input = input("\n🗣️  You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye','Thank you']:
                    print(f"\n🤖 Bot: {self.chat(user_input)}")
                    break
                
                if not user_input:
                    continue
                
                print(f"\n🤖 Bot: {self.chat(user_input)}")
                
            except KeyboardInterrupt:
                print(f"\n\n🤖 Bot: Thanks for chatting! Happy reading! 📚✨")
                break
            except Exception as e:
                print(f"\n🤖 Bot: Oops! I encountered an error: {str(e)}")
                print("Let's try again! What book are you looking for?")


if __name__ == "__main__":
    CSV_FILE_PATH = "book.csv" 
    GEMINI_API_KEY = "AIzaSyBhgcSoIOKd8AoOxO5ypYwJyiJnhtkkieE"
    
    try:
        chatbot = BookRecommenderChatbot(CSV_FILE_PATH, GEMINI_API_KEY)
        chatbot.start_interactive_session()
        
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        print("Please check your CSV file path and Gemini API key!")


def get_book_recommendations(query: str, csv_path: str, api_key: str) -> str:
    try:
        chatbot = BookRecommenderChatbot(csv_path, api_key)
        return chatbot.chat(query)
    except Exception as e:
        return f"Error getting recommendations: {str(e)}"
