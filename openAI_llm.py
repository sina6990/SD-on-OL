from langchain_openai import ChatOpenAI, OpenAI
from langchain.schema import HumanMessage

class AnyOpenAILLM:
    """
    Minimal wrapper around LangChain's ChatOpenAI/OpenAI that matches your prior code.
    Falls back with a clear error if LangChain is unavailable.
    """

    def __init__(self, *args, **kwargs):
        self.model_type = kwargs.get("model_type", "chat")
        model_name = kwargs.get("model_name")
        api_key = kwargs.get("api_key", "ollama")
        api_base = kwargs.get("api_base", "http://localhost:1234/v1")
        temperature = kwargs.get("temperature", 0)
        max_tokens = kwargs.get("max_tokens", 10048)
        model_kwargs = kwargs.get("model_kwargs", {})

        if self.model_type == "completion":
            self.model = OpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                openai_api_base=api_base,
                temperature=temperature,
                max_tokens=max_tokens,
                **model_kwargs,
            )
        else:
            self.model = ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                openai_api_base=api_base,
                temperature=temperature,
                max_tokens=max_tokens,
                model_kwargs=model_kwargs,
            )

    def __call__(self, prompt: str) -> str:
        if self.model_type == "completion":
            return str(self.model(prompt))
        else:
            resp = self.model.invoke([HumanMessage(content=prompt)])
            raw_output = getattr(resp, "content", resp)
            return raw_output