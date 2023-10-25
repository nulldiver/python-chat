import time
from classes.chat import ChatBot

if __name__ == '__main__':
    """
    This is a simple chatbot example that uses the ChatBot class.
    """
    chat = ChatBot(
        system_message="You are a helpful virtual assistant. You answer questions in a friendly and concise manner."
    )
    try:
        print ("Chat started. Type Ctrl+C to exit.")
        while True:
            user_input = input("\n\nYou: ")
            chat.add_message('user', user_input)
            message = chat.get_response()
            full_text = ""
            print ("\nAssistant: ", end="", flush=True)
            for message_chunk in message:
                if message_chunk is None:
                    break
                sleep_attempts = 0
                while message_chunk is None and sleep_attempts < 10:
                    time.sleep(0.5)
                    sleep_attempts += 1
                    message_chunk = next(message)
                print(message_chunk, end='', flush=True)
                full_text += message_chunk
            chat.add_message('assistant', full_text)
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting chat...")
    finally:
        chat.cleanup()
