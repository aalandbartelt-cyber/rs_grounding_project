"""TEST FIXTURE ONLY: deterministic random ranking and full-image boxes; not a model."""
import hashlib
import random


class Fixture:
    purpose="TEST_ONLY_NOT_A_MODEL"

    def prepare(self,images):
        self.ids=[r.image_id for r in images]

    def retrieve(self,text):
        ranking=list(self.ids)
        seed=int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8],"big")
        random.Random(seed).shuffle(ranking)
        return ranking

    def propose(self,image):
        return [{"candidate_id":"whole_image","category_id":1,"score":0.0,
                 "bbox":[0.0,0.0,float(image.width),float(image.height)]}]

    def ground(self,image,text):
        box=self.propose(image)[0]
        return {**box}


def create(config):
    if config:
        raise ValueError("Fixture takes no model parameters")
    return Fixture()
