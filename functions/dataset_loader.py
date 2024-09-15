import datasets # used to load the dataset
def data_loader():
    ds = datasets.load_dataset("stanfordnlp/imdb")
    return ds["train"]["text"]