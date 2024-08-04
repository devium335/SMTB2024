import os
import pickle
from typing import Any, Literal

import esm
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class DownstreamDataset(Dataset):
    """
    A Dataset class for handling embeddings and labels for downstream tasks.

    Args:
        embeddings (list[torch.Tensor]): A list of embeddings.
        labels (list): A list of labels corresponding to the embeddings.
    """

    def __init__(self, embeddings: list[torch.Tensor], labels: list):
        """
        Initialize the DownstreamDataset with embeddings and labels.

        Args:
            embeddings (list[torch.Tensor]): A list of embeddings.
            labels (list): A list of labels.
        """
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.embeddings)

    def __getitem__(self, idx: int):
        """
        Retrieves the embedding and label at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the embedding and the corresponding label.
        """
        return self.embeddings[idx], self.labels[idx]


class ESMEmbedder:
    """
    A class to handle embeddings using ESM models from the 'esm' library.

    Args:
        num_layers (Literal[6, 12, 30, 33]): The number of layers in the ESM model to use.
    """

    def __init__(self, num_layers: Literal[6, 12, 30, 33]):
        self.num_layers = num_layers
        self.models = {
            6: "esm2_t6_8M_UR50D",
            12: "esm2_t12_35M_UR50D",
            30: "esm2_t30_150M_UR50D",
            33: "esm2_t33_650M_UR50D",
        }

        if self.num_layers not in self.models:
            raise ValueError(
                f"Unsupported number of layers: {self.num_layers}. Supported layers are: {list(self.models.keys())}"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.alphabet = getattr(esm.pretrained, self.models[self.num_layers])()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval().to(self.device)

    def run(
        self,
        data: list[str],
        layers: list[int] | None = None,
        contacts: bool = False,
    ) -> list[dict[Any, dict[Any, Any] | Any]]:
        """
        Compute the embeddings from one ESM Model for the given protein sequences.

        Args:
            data (list[str]): A list of protein sequences given as strings.
            layers (list[int], optional): A list of layers to look at. If None, all layers are used.
            contacts (bool): Boolean flag to extract contacts (default is False).

        Returns:
            list[dict[Any, dict[Any, Any] | Any]]: A list of dictionaries with the embeddings for each protein sequence.
        """
        if layers is None:
            layers = range(self.num_layers + 1)
        results = []
        for prot in tqdm(data):
            batch_labels, batch_strs, batch_tokens = self.batch_converter([("protein", prot)])
            batch_tokens = batch_tokens.to(self.device)  # Ensure tokens are on the GPU
            with torch.no_grad():
                outputs = self.model(batch_tokens, repr_layers=layers, return_contacts=contacts)
                detached_outputs = {}
                for k, v in outputs.items():
                    if isinstance(v, dict):  # Check if value is a dictionary (like "representations")
                        detached_outputs[k] = {k1: v1.detach().cpu() for k1, v1 in v.items()}
                    else:
                        detached_outputs[k] = v.detach().cpu()
                results.append(detached_outputs)
        return results


def get_protlist(df: str):
    """
    Get the protein sequences from the given DataFrame.

    Args:
        df (str): Path to the DataFrame.

    Returns:
        list: A list of protein sequences.
    """
    data = pd.read_csv(df)
    prot_list = []
    d_dict = data.to_dict(orient="index")
    for key in d_dict.keys():
        n = d_dict[key][list(d_dict[key].keys())[0]]
        prot_list.append(n)
    return prot_list


def embeddings_to_dataset(dataframe: pd.DataFrame, embeddings: list[dict[Any, dict[Any, Any] | Any]], layer: int):
    """
    Convert the embeddings to a DownstreamDataset object.

    Args:
        dataframe (pd.DataFrame): DataFrame of the dataset.
        embeddings (list[dict[Any, dict[Any, Any] | Any]]): List of embeddings for each protein sequence.
        layer (int): Layer of the model to use for the embeddings.

    Returns:
        DownstreamDataset: DownstreamDataset object.
    """
    labels = list(dataframe[dataframe.columns[1]])
    embedd_list = []
    for i in range(len(embeddings)):
        embedd_list.append(embeddings[i]["representations"][layer].mean([0, 1]))
    return DownstreamDataset(embedd_list, labels)


def load_dataset(path: str, nlayers: int):
    """
    Preprocess the dataset from the given path and compute the embeddings for the model with nlayers-many layers.

    Args:
        path (str): Path to the dataset.
        nlayers (int): Maximum number of layers of the model to use for embedding.

    Returns:
        list: A list of DownstreamDataset objects, one for each layer of the model.
    """
    save_path = path[:-3] + "pkl"

    # Load results from pkl file
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            return pickle.load(f)

    df = pd.read_csv(path)
    protlist = get_protlist(path)
    protlist_emb = ESMEmbedder(nlayers).run(protlist)
    dataset_list = []
    for i in range(nlayers):
        df_embedded = embeddings_to_dataset(df, protlist_emb, i)
        dataset_list.append(df_embedded)

    # save results
    with open(save_path, "wb") as f:
        pickle.dump(dataset_list, f)

    return dataset_list
