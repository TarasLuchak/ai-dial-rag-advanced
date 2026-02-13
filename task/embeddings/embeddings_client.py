import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


#TODO:
# ---
# https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
# ---
# Implement DialEmbeddingsClient:
# - constructor should apply deployment name and api key
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)

class DialEmbeddingsClient:
    _endpoint: str
    _api_key: str

    def __init__(self, deployment_name: str, api_key: str):
        """
        Client for the DIAL Embeddings API.

        :param deployment_name: DIAL/OpenAI deployment name, e.g. 'text-embedding-3-small-1'
        :param api_key: DIAL API key
        """
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")

        self._endpoint = DIAL_EMBEDDINGS.format(model=deployment_name)
        self._api_key = api_key

    def get_embeddings(
            self,
            inputs: list[str],
            dimensions: int | None = None,
            print_request: bool = False,
            **kwargs,
    ) -> dict[int, list[float]]:
        """
        Generate embeddings for the provided list of input strings.

        Returns a mapping where:
          - key: index of input in the original `inputs` list
          - value: embedding vector (list of floats)
        """
        if not inputs:
            return {}

        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json",
        }

        payload: dict = {
            "input": inputs,
            **kwargs,
        }

        # For text-embedding-3-* models we can optionally control the output vector size
        if dimensions is not None:
            payload["dimensions"] = dimensions

        if print_request:
            print(f"Requesting embeddings for {len(inputs)} inputs with params: {set(payload.keys())}")

        response = requests.post(
            url=self._endpoint,
            headers=headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        data = response.json()
        embeddings: dict[int, list[float]] = {}

        for item in data.get("data", []):
            index = item.get("index")
            embedding = item.get("embedding")
            if index is not None and embedding is not None:
                embeddings[int(index)] = embedding

        return embeddings


# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }
