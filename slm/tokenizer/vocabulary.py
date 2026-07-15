from slm.utils.dataset import read_file

class Vocabulary():
    def __init__(self, path: str = "slm/data/sample.txt") -> None:
        self.path = path
        
        self.word_to_id, self.id_to_word = self.create_vocab()

    def __len__(self) -> int:
        return len(self.word_to_id)
    
    def create_vocab(self):
        file_data = read_file(self.path)
        file_data = file_data.replace("\n","")
        file_data = file_data.replace(".", " ")
        all_words = file_data.split()
        word_to_id : dict[str, int] = {
            "<PAD>": 0,
            "<UNK>": 1
        }
        for word in all_words:
            if word.lower() in word_to_id:
                continue
            word_to_id[word.lower()] = len(word_to_id)
        id_to_word = {v:k for k,v in word_to_id.items()}
        return word_to_id, id_to_word

    def encode(self, word: str) -> int:
        return self.word_to_id[word.lower()] if word in self.word_to_id else self.word_to_id["<UNK>"]
    
    def decode(self, id: int) -> str:
        return self.id_to_word[id]
    
    def encode_sentence(self, sentence: str) -> list[int]:
        cleaned_sentence = sentence.replace("\n", "")
        cleaned_sentence = cleaned_sentence.replace(".", "")
        result = [self.encode(word.lower()) for word in cleaned_sentence.split()]
        return result
    
    def decode_sentence(self, ids: list[int]) -> str:
        result = [self.decode(id) for id in ids]
        return " ".join(result)


