from typing import List
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import (
    BaseModel,
)

from tqdm import tqdm
from utils import get_clip_embeddings, encode_image

class MultimodalEmbeddings(BaseModel, Embeddings):
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        embeddings = []
        for text in texts:
            embedding = get_clip_embeddings(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:

        return self.embed_documents([text])[0]

    def embed_image_text_pairs(self, texts: List[str], images: List[str], batch_size=2) -> List[List[float]]:

        # the length of texts must be equal to the length of images
        assert len(texts)==len(images), "the len of captions should be equal to the len of images"

        embeddings = []
        for path_to_img, text in tqdm(zip(images, texts), total=len(texts)):
            embedding = get_clip_embeddings(text, encode_image(path_to_img))
            embeddings.append(embedding)
        return embeddings