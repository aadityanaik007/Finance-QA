from sentence_transformers import SentenceTransformer


def main():
    print("Hello from financial-qa!")
    sentences = ["This is an example sentence", "Each sentence is converted"]

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    print(embeddings)



if __name__ == "__main__":
    main()
