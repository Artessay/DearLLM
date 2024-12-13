import threading
from tqdm import tqdm

from perplexity_server.model import LargeLanguageModel

class ModelPool:
    def __init__(self, model_name_or_path, n_models=1):
        self.n_models = n_models
        self.models = [LargeLanguageModel(model_name_or_path, index=idx) for idx in tqdm(range(n_models))]

        self.lock = threading.Lock()
        self.semaphore = threading.Semaphore(n_models)

    def acquire(self):
        self.semaphore.acquire()
        self.lock.acquire()

        model = self.models.pop()
        # print("Acquired model {}, remain {} models".format(model, len(self.models)))

        self.lock.release()

        return model
    
    def release(self, model):
        self.lock.acquire()

        self.models.append(model)
        # print("Released model {}, remain {} models".format(model, len(self.models)))

        self.lock.release()
        self.semaphore.release()

if __name__ == "__main__":
    from utils import get_model_name_or_path    
    model_name_or_path = get_model_name_or_path()

    pool = ModelPool(model_name_or_path, n_models=3)
    print("model num:", pool.n_models)

    model1 = pool.acquire()
    model2 = pool.acquire()
    model3 = pool.acquire()

    pool.release(model1)
    pool.release(model2)

    model4 = pool.acquire()
    pool.release(model3)

    model5 = pool.acquire()
    pool.release(model4)
    pool.release(model5)