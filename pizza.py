import google.generativeai as genai
import json
from typing import Dict, Any


class PizzeriaBot:
    def __init__(self, api_key: str):
        """Initialize the pizzeria chatbot with Gemini API"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Available ingredients database
        self.available_ingredients = {
            "meats": ["pepperoni", "sausage", "ham", "bacon", "chicken", "beef", "salami"],
            "vegetables": ["mushrooms", "onions", "bell peppers", "tomatoes", "olives", 
                          "spinach", "arugula", "basil", "garlic"],
            "cheeses": ["mozzarella", "cheddar", "parmesan", "feta", "goat cheese"],
            "sauces": ["tomato sauce", "white sauce", "pesto", "bbq sauce", "ranch"]
        }
        
        # Pizza sizes and prices
        self.pizza_sizes = {
            "small": {"price": 12.99, "description": "10 inch"},
            "medium": {"price": 15.99, "description": "12 inch"},
            "large": {"price": 18.99, "description": "14 inch"},
            "extra large": {"price": 21.99, "description": "16 inch"}
        }
        
        # Current order state
        self.current_order = {
            "size": None,
            "ingredients": [],
            "total_price": 0.0,
            "confirmed": False
        }
        
        # Function definitions for Gemini
        self.functions = [
            {
                "name": "create_pizza_with_size",
                "description": "Create a pizza with specific size and automatically add main ingredient (like pepperoni pizza)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "size": {
                            "type": "string",
                            "description": "The size of the pizza (small, medium, large, extra large)"
                        },
                        "main_ingredient": {
                            "type": "string", 
                            "description": "Main ingredient for the pizza (like pepperoni for pepperoni pizza)"
                        }
                    },
                    "required": ["size"]
                }
            },
            {
                "name": "add_ingredient",
                "description": "Add an ingredient to the pizza order",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ingredient": {
                            "type": "string",
                            "description": "The ingredient to add to the pizza"
                        }
                    },
                    "required": ["ingredient"]
                }
            },
            {
                "name": "set_pizza_size",
                "description": "Set the size of the pizza",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "size": {
                            "type": "string",
                            "description": "The size of the pizza (small, medium, large, extra large)"
                        }
                    },
                    "required": ["size"]
                }
            },
            {
                "name": "remove_ingredient",
                "description": "Remove an ingredient from the pizza order",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ingredient": {
                            "type": "string",
                            "description": "The ingredient to remove from the pizza"
                        }
                    },
                    "required": ["ingredient"]
                }
            },
            {
                "name": "show_current_order",
                "description": "Display the current pizza order details",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "confirm_order",
                "description": "Confirm the pizza order",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "show_available_ingredients",
                "description": "Show all available ingredients",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]

    def normalize_ingredient(self, ingredient: str) -> str:
        """Normalize ingredient name to handle typos and variations"""
        ingredient = ingredient.lower().strip()
        
        # Common typo mappings
        typo_map = {
            'peperoni': 'pepperoni',
            'pepperoni': 'pepperoni',
            'mozarela': 'mozzarella',
            'mozzarela': 'mozzarella',
            'mozzarella': 'mozzarella',
            'mushroom': 'mushrooms',
            'mushroms': 'mushrooms',
            'mashrooms': 'mushrooms',
            'onion': 'onions',
            'tomatoe': 'tomatoes',
            'olive': 'olives',
            'bellpepper': 'bell peppers',
            'bell pepper': 'bell peppers',
            'green pepper': 'bell peppers',
            'cheese': 'mozzarella'  # Default cheese
        }
        
        return typo_map.get(ingredient, ingredient)

    def create_pizza_with_size(self, size: str, main_ingredient: str = None) -> str:
        """Create a pizza with size and optionally add main ingredient"""
        size = size.lower().strip()
        if size in self.pizza_sizes:
            self.current_order["size"] = size
            price = self.pizza_sizes[size]["price"]
            description = self.pizza_sizes[size]["description"]
            
            response = f"Perfect! I've set your pizza size to {size} ({description}) for ${price:.2f}."
            
            # If main ingredient is specified, add it
            if main_ingredient:
                normalized_ingredient = self.normalize_ingredient(main_ingredient)
                all_ingredients = []
                for category in self.available_ingredients.values():
                    all_ingredients.extend([ing.lower() for ing in category])
                
                if normalized_ingredient in all_ingredients:
                    for category in self.available_ingredients.values():
                        for orig_ingredient in category:
                            if orig_ingredient.lower() == normalized_ingredient:
                                self.current_order["ingredients"].append(orig_ingredient)
                                response += f" I've also added {orig_ingredient} to your pizza."
                                break
            
            response += " Would you like to add any other toppings?"
            return response
        else:
            available_sizes = ", ".join(self.pizza_sizes.keys())
            return f"I'm sorry, we don't have that size. Available sizes are: {available_sizes}"

    def add_ingredient(self, ingredient: str) -> str:
        """Add ingredient to pizza if available"""
        normalized_ingredient = self.normalize_ingredient(ingredient)
        
        # Check if ingredient is available
        all_ingredients = []
        for category in self.available_ingredients.values():
            all_ingredients.extend([ing.lower() for ing in category])
        
        if normalized_ingredient in all_ingredients:
            if normalized_ingredient not in [ing.lower() for ing in self.current_order["ingredients"]]:
                # Find original case of ingredient
                for category in self.available_ingredients.values():
                    for orig_ingredient in category:
                        if orig_ingredient.lower() == normalized_ingredient:
                            self.current_order["ingredients"].append(orig_ingredient)
                            return f"Great! I've added {orig_ingredient} to your pizza. Anything else you'd like to add?"
            else:
                return f"You already have {normalized_ingredient} on your pizza. Would you like to add something else?"
        else:
            return f"I'm sorry, we don't have {ingredient} available. Let me show you what we have available!"

    def set_pizza_size(self, size: str) -> str:
        """Set pizza size"""
        size = size.lower().strip()
        if size in self.pizza_sizes:
            self.current_order["size"] = size
            price = self.pizza_sizes[size]["price"]
            description = self.pizza_sizes[size]["description"]
            return f"Perfect! I've set your pizza size to {size} ({description}) for ${price:.2f}. What toppings would you like?"
        else:
            available_sizes = ", ".join(self.pizza_sizes.keys())
            return f"I'm sorry, we don't have that size. Available sizes are: {available_sizes}"

    def remove_ingredient(self, ingredient: str) -> str:
        """Remove ingredient from pizza"""
        ingredient = ingredient.lower().strip()
        for i, ing in enumerate(self.current_order["ingredients"]):
            if ing.lower() == ingredient:
                removed = self.current_order["ingredients"].pop(i)
                return f"I've removed {removed} from your pizza. Is there anything else you'd like to change?"
        return f"I don't see {ingredient} on your current pizza. Would you like to see your current order?"

    def show_current_order(self) -> str:
        """Display current order"""
        if not self.current_order["size"]:
            return "You haven't selected a pizza size yet. Would you like to start with choosing a size?"
        
        size_info = self.pizza_sizes[self.current_order["size"]]
        order_summary = f"Here's your current order:\n"
        order_summary += f"• Size: {self.current_order['size'].title()} ({size_info['description']}) - ${size_info['price']:.2f}\n"
        
        if self.current_order["ingredients"]:
            order_summary += f"• Toppings: {', '.join(self.current_order['ingredients'])}\n"
        else:
            order_summary += "• Toppings: None yet\n"
        
        order_summary += f"• Total: ${size_info['price']:.2f}\n\n"
        order_summary += "Would you like to add more toppings or confirm your order?"
        
        return order_summary

    def confirm_order(self) -> str:
        """Confirm the pizza order"""
        if not self.current_order["size"]:
            return "Please select a pizza size first before confirming your order."
        
        if not self.current_order["ingredients"]:
            return "Would you like to add some toppings to your pizza before confirming? We have lots of delicious options!"
        
        self.current_order["confirmed"] = True
        size_info = self.pizza_sizes[self.current_order["size"]]
        
        confirmation = f"🍕 Order Confirmed! 🍕\n\n"
        confirmation += f"Your {self.current_order['size']} pizza with {', '.join(self.current_order['ingredients'])} "
        confirmation += f"will be ready in 15-20 minutes.\n\n"
        confirmation += f"Total: ${size_info['price']:.2f}\n\n"
        confirmation += "Thank you for choosing our pizzeria! We'll start preparing your delicious pizza right away!"
        
        return confirmation

    def show_available_ingredients(self) -> str:
        """Show all available ingredients"""
        ingredients_text = "Here are all our available ingredients:\n\n"
        for category, items in self.available_ingredients.items():
            ingredients_text += f"🍕 {category.title()}:\n"
            ingredients_text += f"   {', '.join(items)}\n\n"
        ingredients_text += "What would you like to add to your pizza?"
        return ingredients_text

    def execute_function(self, function_name: str, parameters: Dict[str, Any]) -> str:
        """Execute the requested function"""
        if function_name == "create_pizza_with_size":
            return self.create_pizza_with_size(
                parameters.get("size", ""), 
                parameters.get("main_ingredient", None)
            )
        elif function_name == "add_ingredient":
            return self.add_ingredient(parameters.get("ingredient", ""))
        elif function_name == "set_pizza_size":
            return self.set_pizza_size(parameters.get("size", ""))
        elif function_name == "remove_ingredient":
            return self.remove_ingredient(parameters.get("ingredient", ""))
        elif function_name == "show_current_order":
            return self.show_current_order()
        elif function_name == "confirm_order":
            return self.confirm_order()
        elif function_name == "show_available_ingredients":
            return self.show_available_ingredients()
        else:
            return "I'm sorry, I couldn't understand that request. Could you please rephrase?"

    def chat(self, user_message: str) -> str:
        """Main chat function that processes user input"""
        try:
            # Handle simple confirmation and completion phrases
            user_lower = user_message.lower().strip()
            
            # If customer says "that's all" or similar, show order or confirm
            if any(phrase in user_lower for phrase in ['that all', 'thats all', "that's all", 'done', 'finish', 'complete']):
                if self.current_order['size'] and self.current_order['ingredients']:
                    return self.confirm_order()
                else:
                    return self.show_current_order()
            
            # Create the system prompt with conversation history awareness
            system_prompt = f"""
            You are a friendly pizzeria assistant. REMEMBER the conversation context!
            
            CURRENT ORDER STATUS: {json.dumps(self.current_order)}
            
            Available functions:
            - add_ingredient: Add toppings to pizza (only use when customer specifically wants to add a topping)
            - set_pizza_size: Set pizza size (use when customer mentions size)
            - show_current_order: Display current order
            - confirm_order: Confirm the pizza order
            - show_available_ingredients: Show available toppings
            
            CRITICAL RULES:
            1. REMEMBER what has already been set in the current order
            2. If size is already set, don't ask for size again
            3. When customer mentions "pepperoni" initially, they want a pepperoni pizza - ask for size, then add pepperoni automatically
            4. When customer says "cheese" and size is set, add mozzarella (default cheese)
            5. Handle typos: "peperoni"="pepperoni", "mozarela"="mozzarella"
            6. Don't repeat questions or lose context
            
            CONVERSATION LOGIC:
            - If no size set and customer wants pizza type: ask for size
            - If size set but no ingredients: add the ingredients they mentioned
            - If they confirm (yes/ok/that's all): confirm order or show current order
            - Don't ask the same question twice!
            
            Available ingredients: {json.dumps(self.available_ingredients)}
            Pizza sizes: {json.dumps(self.pizza_sizes)}
            """
            
            # Generate response with function calling
            response = self.model.generate_content(
                f"{system_prompt}\n\nCustomer: {user_message}",
                tools=[{"function_declarations": self.functions}]
            )
            
            # Check if the model wants to call a function
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_name = part.function_call.name
                        parameters = dict(part.function_call.args)
                        
                        # Special handling for common scenarios
                        if function_name == "set_pizza_size" and not self.current_order['ingredients']:
                            # If setting size and we know they want pepperoni from context, add it
                            result = self.execute_function(function_name, parameters)
                            # Don't auto-add ingredients here, let the conversation flow naturally
                            return result
                        else:
                            function_result = self.execute_function(function_name, parameters)
                            return function_result
                    elif hasattr(part, 'text') and part.text:
                        return part.text
            
            # If no function call, return a helpful response
            return "I'd be happy to help you with your pizza order! What would you like?"
            
        except Exception as e:
            return f"I'm sorry, I encountered an error: {str(e)}. Could you please try again?"

def main():
    # You need to set your Gemini API key here
    API_KEY = "AIzaSyCWJx6bpJ6LNiVbwpKZ2xya7CH_X-ny8Es"  # Replace with your actual API key
    
    if API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("Please set your Gemini API key in the API_KEY variable")
        return
    
    bot = PizzeriaBot(API_KEY)
    
    print("🍕 Welcome toPizzeria! 🍕")
    print("I'm your pizza assistant. I can help you create the perfect pizza!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Thank you for visiting Tony's Pizzeria! Have a great day! 🍕")
            break
        
        if user_input:
            response = bot.chat(user_input)
            print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()