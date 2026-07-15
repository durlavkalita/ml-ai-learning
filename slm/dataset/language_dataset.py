from slm.tokenizer.vocabulary import Vocabulary
from torch.utils.data import Dataset
from slm.utils.dataset import read_file

class LanguageDataset(Dataset[tuple[list[int],int]]):
    def __init__(self, path: str):
        self.content = read_file(path)
        self.vocab = Vocabulary(path)
        self.sentences = []
        self.samples = []
        self.split_into_sentences()
        self.generate_training_set()

    def split_into_sentences(self) -> None:
        sentences = self.content.split(".")
        self.sentences = [sentence.strip() for sentence in sentences if len(sentence)>0]

    def generate_training_set(self) -> None:
        training_list: list[tuple[list[int],int]] = []
        for sentence in self.sentences:
            word_ids = self.vocab.encode_sentence(sentence)
            for i in range(len(word_ids)-1):
                training_list.append((word_ids[0:i+1], word_ids[i+1]))
        self.samples = training_list

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        return self.samples[idx]
    

    
if __name__ == "__main__":
    c = LanguageDataset("slm/data/sample.txt")
    print(c[5])