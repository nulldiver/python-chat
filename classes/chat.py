import os
import random
import string
import threading
import time
from typing import Optional, List, Dict, Any, Union, Tuple, Callable, Iterator
import openai
import openai.error as openai_error
import tiktoken
from requests.exceptions import ConnectionError, Timeout


class ChatBot:
    """
    A chatbot that uses OpenAI's API to generate responses to user input.

    Attributes:
        api_key (str): The API key to use for OpenAI's API. Defaults to the OPENAI_API_KEY environment variable.
        model_name (str): The name of the model to use for OpenAI's API. Defaults to "gpt-3.5-turbo".
        system_message (str): The system message to direct the chatbot when it is started. Defaults to None.
        token_threshold (int): The maximum number of tokens to allow in the message history before summarizing
            the oldest messages.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo",
                 system_message: Optional[str] = None, token_threshold: int = 1400):
        if api_key:
            self.api_key: str = api_key
        else:
            self.api_key: str = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("API key not provided and OPENAI_API_KEY environment variable not set.")

        openai.api_key = self.api_key
        self.model_name: str = model_name
        self.messages: List[Dict[str, str]] = []
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.token_threshold: int = token_threshold
        self.lock: threading.Lock = threading.Lock()
        self.summary_thread: Optional[threading.Thread] = None
        self.is_summarizing: bool = False
        self.add_message('system', system_message)

    def cleanup(self) -> None:
        """
        Cleans up the chatbot by joining the summary thread if it is still running.

        Returns:
            None
        """
        if self.summary_thread and self.summary_thread.is_alive():
            self.summary_thread.join()
        self.messages = []

    def _count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in the given text.

        Args:
            text (str): The text to count the tokens of.

        Returns:
            int: The number of tokens in the given text.
        """
        return len(self.encoder.encode(text))

    def add_message(self, role: str, content: str) -> None:
        """
        Adds a message to the message history. If the total number of tokens in the message history exceeds the
        token threshold, the chatbot will summarize the oldest messages in the history.

        Args:
            role (str): The role of the message. Either 'user' or 'assistant'.
            content (str): The content of the message.

        Returns:
            None
        """
        with self.lock:
            self.messages.append({'role': role, 'content': content})
        total_tokens: int = sum(self._count_tokens(message['content']) for message in self.messages)
        if total_tokens > self.token_threshold:
            self._summarize_old_messages(self.token_threshold * 0.35)

    def _summarize_old_messages(self, token_threshold: float = 500) -> None:
        """
        Summarizes the oldest messages in the message history.

        Args:
            token_threshold (float): The maximum number of tokens to allow in the message history before summarizing
                the oldest messages.

        Returns:
            None
        """
        if self.is_summarizing:
            return
        self.is_summarizing = True
        self.summary_thread = threading.Thread(target=self._summarize_threaded, args=(token_threshold,))
        self.summary_thread.start()

    def _summarize_threaded(self, token_threshold: float) -> None:
        """
        Summarizes the oldest messages in the message history.

        Args:
            token_threshold (float): The maximum number of tokens to allow in the message history before summarizing
                the oldest messages.

        Returns:
            None
        """
        with self.lock:  # Lock the shared resource while modifying
            total_tokens: int = 0
            messages_to_summarize: List[Dict[str, str]] = []
            i = 1
            while i < len(self.messages) - 1:
                message = self.messages[i]
                message_tokens: int = self._count_tokens(message['content'])
                if total_tokens + message_tokens <= token_threshold:
                    messages_to_summarize.append(message)
                    total_tokens += message_tokens
                    i += 1
                    if message['role'] == 'user' and self.messages[i]['role'] == 'assistant':
                        messages_to_summarize.append(self.messages[i])
                        total_tokens += self._count_tokens(self.messages[i]['content'])
                        i += 1
                if total_tokens > token_threshold:
                    break
            if messages_to_summarize:
                text_to_summarize: str = "\n".join(
                    [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages_to_summarize])
                summarized_text: str = self._get_summary(text_to_summarize)
                replacement_index: int = 1 + len(messages_to_summarize)
                self.messages = [self.messages[0], {'role': 'system', 'content': summarized_text}] + self.messages[
                                                                                                     replacement_index:]
            self.is_summarizing = False

    def _get_summary(self, text: str) -> str:
        """
        Gets a summary of the given text.

        Args:
            text (str): The text to summarize.

        Returns:
            str: The summary of the given text.
        """
        summary_messages: List[Dict[str, str]] = [{'role': 'system',
                                                   'content': 'You take fragments of chat conversation provided by the '
                                                              'user and output a short and very concise summary of the '
                                                              'conversation. You reply only with the summary.'},
                                                  {'role': 'user',
                                                   'content': f"Here is a chat conversation for you to summarize:"
                                                              f" {text}"}]
        response: Dict[str, Any] = self._completion_with_backoff(model=self.model_name, messages=summary_messages,
                                                                 temperature=0.5,
                                                                 max_tokens=100)
        summarized_text: str = response['choices'][0]['message']['content']
        summarized_text = summarized_text.strip()
        return summarized_text

    @staticmethod
    def retry_with_exponential_backoff(
            initial_delay: float = 1,
            exponential_base: float = 2,
            jitter: bool = True,
            max_retries: int = 10,
            rate_limit_errors: Tuple[type, ...] = (openai_error.RateLimitError,),
            network_errors: Tuple[type, ...] = (
            ConnectionError, Timeout, openai_error.Timeout, openai_error.APIConnectionError),
            api_errors: Tuple[type, ...] = (
            openai_error.APIError, openai_error.InvalidRequestError, openai_error.AuthenticationError,)
    ) -> Callable:
        """
        Decorator that retries a function with exponential backoff.

        Args:
            initial_delay (float): The initial delay in seconds.
            exponential_base (float): The base of the exponential backoff.
            jitter (bool): Whether to add jitter to the exponential backoff.
            max_retries (int): The maximum number of retries.
            rate_limit_errors (Tuple[type, ...]): The errors to retry on.
            network_errors (Tuple[type, ...]): The errors to retry on.
            api_errors (Tuple[type, ...]): The errors to break on.

        Returns:
            Callable: The decorated function.
        """

        def decorator(func: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                num_retries: int = 0
                delay: float = initial_delay
                while True:
                    try:
                        return func(*args, **kwargs)
                    except rate_limit_errors:
                        num_retries += 1
                        if num_retries > max_retries:
                            raise Exception(f"Maximum number of retries ({max_retries}) exceeded due to rate limits.")
                        delay *= exponential_base * (1 + jitter * random.random())
                        time.sleep(delay)
                    except network_errors:
                        num_retries += 1
                        if num_retries > max_retries:
                            raise Exception(
                                f"Maximum number of retries ({max_retries}) exceeded due to network errors.")
                        print("Network error encountered. Retrying...")
                        delay *= exponential_base * (1 + jitter * random.random())
                        time.sleep(delay)
                    except api_errors as e:
                        print(f"API Error encountered: {e}. Stopping further attempts.")
                        break
                    except Exception as e:
                        raise e

            return wrapper

        return decorator

    @retry_with_exponential_backoff()
    def _completion_with_backoff(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Calls the OpenAI API with exponential backoff.

        Args:
            **kwargs (Any): The arguments to pass to the OpenAI API.

        Returns:
            Dict[str, Any]: The response from the OpenAI API.
        """
        try:
            response: Dict[str, Any] = openai.ChatCompletion.create(**kwargs)
            return response
        except openai_error.APIError as e:
            raise Exception(f"API error encountered: {e}")
        except Exception as e:
            raise e

    def get_response(self, buffer_threshold: int = 3) -> Iterator[Union[str, None]]:
        """
        Gets a response from the chatbot.

        Args:
            buffer_threshold (int): The number of words to wait for before yielding a response.

        Returns:
            Iterator[Union[str, None]]: The response from the chatbot.
        """

        def is_valid_end_character(ch: str) -> bool:
            return ch.isspace() or ch in string.punctuation

        response = self._completion_with_backoff(
            model=self.model_name,
            messages=self.messages,
            temperature=0.7,
            max_tokens=200,
            stream=True)
        buffer: str = ""
        did_first_yield: bool = False
        for chunk in response:
            # noinspection PyTypeChecker
            chunk_message: Dict[str, Any] = chunk['choices'][0]['delta']
            chunk_content: str = chunk_message.get('content', '')
            buffer += chunk_content
            while len(buffer.split()) >= buffer_threshold and not is_valid_end_character(buffer[-1]):
                try:
                    # noinspection PyTypeChecker
                    next_chunk = next(response)
                    next_chunk_message = next_chunk['choices'][0]['delta']
                    buffer += next_chunk_message.get('content', '')
                except StopIteration:
                    break

            if len(buffer.split()) >= buffer_threshold:
                if not did_first_yield:
                    did_first_yield = True
                yield buffer
                buffer = ""

        if buffer:
            yield buffer

        yield None
