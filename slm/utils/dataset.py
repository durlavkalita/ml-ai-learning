def read_file(path: str) -> str:
    content = ""
    with open(path, 'r') as file:
        content = file.read()
    return content

def split_into_sentences(content: str) -> list[str]:
    sentences = content.split(".")
    return [sentence for sentence in sentences if len(sentence)>0]

def split_into_words(sentence: str) -> list[tuple[list[str], str]]:
    sentence = sentence.replace("\n", "")
    words = sentence.split(" ")
    training_list: list[tuple[list[str],str]] = []
    for i in range(len(words)-1):
        training_list.append((words[0:i+1], words[i+1]))
    return training_list

def main():
    file_content = read_file("slm/data/sample.txt")
    sentences = split_into_sentences(file_content)
    for sentence in sentences:
        print(split_into_words(sentence))

if __name__ == "__main__":
    main()