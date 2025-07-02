"""Azure OpenAI asynchronous LLM implementation."""

from typing import AsyncIterator, List, Dict, Any
from openai import AsyncAzureOpenAI, APIError, APIConnectionError, RateLimitError
from loguru import logger

from .stateless_llm_interface import StatelessLLMInterface


class AsyncLLM(StatelessLLMInterface):
    def __init__(
        self,
        deployment_name: str,
        azure_endpoint: str,
        api_version: str,
        llm_api_key: str,
        temperature: float = 1.0,
    ) -> None:
        """Initialize Azure OpenAI LLM."""
        self.deployment_name = deployment_name
        self.temperature = temperature
        self.client = AsyncAzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=llm_api_key,
        )
        logger.info(
            "Initialized Azure OpenAI AsyncLLM with endpoint: %s, deployment: %s",
            azure_endpoint,
            deployment_name,
        )

    async def chat_completion(
        self, messages: List[Dict[str, Any]], system: str | None = None
    ) -> AsyncIterator[str]:
        """Generate chat completion using Azure OpenAI."""
        messages_with_system = messages
        if system:
            messages_with_system = [
                {"role": "system", "content": system},
                *messages,
            ]
        stream = None
        try:
            stream = await self.client.chat.completions.create(
                messages=messages_with_system,
                model=self.deployment_name,
                temperature=self.temperature,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content is None:
                    chunk.choices[0].delta.content = ""
                yield chunk.choices[0].delta.content
        except APIConnectionError as e:
            logger.error(
                "Error calling the chat endpoint: Connection error. %s",
                e,
            )
            yield (
                "Error calling the chat endpoint: Connection error. "
                "Failed to connect to the LLM API."
            )
        except RateLimitError as e:
            logger.error(
                "Error calling the chat endpoint: Rate limit exceeded: {}",
                e.response,
            )
            yield "Error calling the chat endpoint: Rate limit exceeded. Please try again later."
        except APIError as e:
            logger.error("Azure LLM API error occurred: {}", e)
            yield "Error calling the chat endpoint: Error occurred while generating response."
        finally:
            if stream:
                logger.debug("Chat completion finished.")
                await stream.close()
                logger.debug("Stream closed.")
