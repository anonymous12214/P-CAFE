import os
import random
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, \
    BartForConditionalGeneration, BartTokenizer
import argparse

import pcafe_utils
import pandas as pd
import json
from pathlib import Path

# import spacy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load JSON configuration
with open(r'C:\Users\kashann\PycharmProjects\PCAFE-MIMIC\Integration\user_config_naama.json', 'r') as f:
    config = json.load(f)

# Get the project path from the JSON
project_path = Path(config["user_specific_project_path"])

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--directory",
                    type=str,
                    default=project_path,
                    help="Directory for saved models")
parser.add_argument("--hidden-dim1",
                    type=int,
                    default=64,
                    help="Hidden dimension")
parser.add_argument("--hidden-dim2",
                    type=int,
                    default=32,
                    help="Hidden dimension")
parser.add_argument("--lr",
                    type=float,
                    default=1e-4,
                    help="Learning rate")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.001,
                    help="l_2 weight penalty")
# change these parameters
parser.add_argument("--num_epochs",
                    type=int,
                    default=100000,
                    help="number of epochs")
parser.add_argument("--val_trials_wo_im",
                    type=int,
                    default=500,
                    help="Number of validation trials without improvement")
parser.add_argument("--fraction_mask",
                    type=int,
                    default=0,
                    help="fraction mask")
parser.add_argument("--run_validation",
                    type=int,
                    default=100,
                    help="after how many epochs to run validation")
parser.add_argument("--batch_size",
                    type=int,
                    default=128,
                    help="bach size")
parser.add_argument("--text_embed_dim",
                    type=int,
                    default=768,
                    help="Text embedding dimension")
parser.add_argument("--reduced_dim",
                    type=int,
                    default=20,
                    help="Reduced dimension for text embedding")
parser.add_argument("--save_dir",
                    type=str,
                    default='guesser_eICU',
                    help="save path")
parser.add_argument(
    "--data",
    type=str,
    default="pcafe_utils.load_time_Series()",
    help=(
        "Dataset loader function to use. Options:\n"
        "  pcafe_utils.load_time_Series()        - eICU time series data\n"
        "  pcafe_utils.load_mimic_text()         - MIMIC-III multi-modal (includes text)\n"
        "  pcafe_utils.load_mimic_time_series()  - MIMIC-III numeric time series data"
    )
)





FLAGS = parser.parse_args(args=[])


class ImageEmbedder(nn.Module):
    def __init__(self):
        super(ImageEmbedder, self).__init__()
        # Define CNN layers for embedding
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Flatten to 128 dimensions
        self.fc2 = nn.Linear(128, 20)
        # Define the image transform (convert to tensor and normalize)
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Resize the image to 28x28 if needed
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5, std=0.5
        ])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First conv + pooling
        x = self.pool(F.relu(self.conv2(x)))  # Second conv + pooling
        x = x.view(-1, 64 * 7 * 7)  # Flatten the output from (batch_size, 64, 7, 7) to (batch_size, 64*7*7)
        x = F.relu(self.fc1(x))  # Embedding output (128 features)
        # embedd to 30 features
        x = F.relu(self.fc2(x))
        return x


def map_features_to_indices(data):
    """
    Map features to indices based on their type.
    If any value in a column is a string, treat the column as text.
    :param data: 2D array-like dataset (list of lists, numpy array, or pandas DataFrame)
    :return: A dictionary mapping feature indices to lists of indices and the total number of features
    """
    index_map = {}
    current_index = 0
    if isinstance(data[0], pd.DataFrame):
        for col_index in range(data[0].shape[1] + 20):
            index_map[col_index] = [current_index]
            current_index += 1


    else:
        data = np.array(data)  # Ensure input is a NumPy array for easier handling

        # Check each column to determine type (string/text or numeric)
        for col_index in range(data.shape[1]):  # Iterate over columns
            column_data = data[:, col_index]  # Extract the entire column

            if any(isinstance(value, str) for value in column_data):  # Check for any string in the column
                index_map[col_index] = list(range(current_index, current_index + 20))  # Text feature
                current_index += 20
            else:
                index_map[col_index] = [current_index]  # Numeric feature
                current_index += 1

    return index_map, current_index


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.num_directions = 2 if bidirectional else 1

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        _, (h_n, _) = self.lstm(x)  # h_n: [num_layers * num_directions, batch, hidden_dim]
        # Take the last layer's output
        h_last = h_n[-self.num_directions:]  # shape: [num_directions, batch, hidden_dim]
        embedding = h_last.permute(1, 0, 2).reshape(x.size(0), -1)  # shape: [batch, hidden_dim * num_directions]
        return embedding



def load_data_function(data_loader_name):
    data_loaders = {
        "load_time_Series": pcafe_utils.load_time_Series,
        "load_mimic_text": pcafe_utils.load_mimic_text,
        "load_mimic_time_series": pcafe_utils.load_mimic_time_series,
    }

    # Return the appropriate function based on the provided name
    return data_loaders.get(data_loader_name, pcafe_utils.load_time_Series)  # Default to load_time_Series



class MultimodalGuesser(nn.Module):
    def __init__(self):
        super(MultimodalGuesser, self).__init__()
        self.device = DEVICE
        # Load the function based on the argument
        data_loader_function = load_data_function(FLAGS.data)
        # Now you can call the loader function to get your data
        self.X, self.y, self.tests_number, self.map_test = data_loader_function()
        self.summarize_text_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(
            self.device)
        self.tokenizer_summarize_text_model = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

        # self.text_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(self.device)
        # self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        local_model_path = os.path.join(os.getcwd(), 'Integration/clinicalBert')
        self.text_model = AutoModel.from_pretrained(local_model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        self.cost_list = [1] * (self.tests_number + 1)
        if isinstance(self.X, list):
            self.cost_list = [0] + [1] * (self.tests_number)
        self.cost_budget = sum(self.cost_list)

        self.img_embedder = ImageEmbedder()
        if isinstance(self.X, list):
            self.time_series_embedder = LSTMEncoder(self.X[0].shape[1], 20).to(self.device)
        self.text_reducer = nn.Linear(FLAGS.text_embed_dim, FLAGS.reduced_dim).to(self.device)
        self.text_reduced_dim = FLAGS.reduced_dim
        self.num_classes = len(np.unique(self.y))
        self.map_feature, self.features_total = map_features_to_indices(self.X)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(self.features_total, FLAGS.hidden_dim1),
            torch.nn.PReLU(),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(FLAGS.hidden_dim1, FLAGS.hidden_dim2),
            torch.nn.PReLU(),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(FLAGS.hidden_dim2, FLAGS.hidden_dim2),
            torch.nn.PReLU(),
        )

        self.logits = nn.Linear(FLAGS.hidden_dim2, self.num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          weight_decay=FLAGS.weight_decay,
                                          lr=FLAGS.lr)
        self.path_to_save = os.path.join(os.getcwd(), FLAGS.save_dir)
        self.layer1 = self.layer1.to(self.device)
        self.layer2 = self.layer2.to(self.device)
        self.layer3 = self.layer3.to(self.device)
        self.logits = self.logits.to(self.device)

    def summarize_text(self, text, max_length=300, min_length=100, length_penalty=2.0, num_beams=4):
        """
        Summarizes a long text using BART.

        Args:
            text (str): The input clinical note.
            max_length (int): The maximum length of the summary.
            min_length (int): The minimum length of the summary.
            length_penalty (float): Length penalty for beam search.
            num_beams (int): Number of beams for beam search.

        Returns:
            str: The summarized text.
        """
        inputs = self.tokenizer_summarize_text_model.encode("summarize: " + text, return_tensors="pt", max_length=1024,
                                                            truncation=True).to(self.device)
        summary_ids = self.summarize_text_model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=length_penalty,
            num_beams=num_beams,
            early_stopping=True
        )
        summary = self.tokenizer_summarize_text_model.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def embed_text(self, text):
        """
        Embeds text using ClinicalBERT.

        Args:
            text (str): The input text (e.g., summarized clinical note).

        Returns:
            torch.Tensor: The ClinicalBERT embeddings.
        """
        text = self.summarize_text(text)
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True,
                                padding="max_length").to(self.device)
        outputs = self.text_model(**inputs)
        # Use the CLS token representation (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :].to(self.device)
        # check this line
        return F.relu(self.text_reducer(cls_embedding))

    def embed_image(self, image_path):
        # Get image embedding using ImageEmbedder
        img = Image.open(image_path).convert('L')  # Open the image
        img = self.img_embedder.transform(img).unsqueeze(0).to(self.device)  # Apply transforms and add batch dimension
        embedding = self.img_embedder(img).to(self.device)
        return embedding

    def is_numeric_value(self, value):
        # Check if the value is an integer, a floating-point number, or a tensor of type float or double
        if isinstance(value, (int, float)):
            return True
        elif isinstance(value, torch.Tensor):
            if value.dtype in [torch.float, torch.float64]:
                return True
        return False

    def is_text_value(self, value):
        # Check if the value is a string
        if isinstance(value, str):
            return True
        else:
            return False

    def is_image_value(self, value):
        # check if value is path that ends with 'png' or 'jpg'
        if isinstance(value, str):
            if value.endswith('png') or value.endswith('jpg'):
                return True
            else:
                return False

    def is_time_series_value(self, value):
        # Check if the value is a time series
        if isinstance(value, pd.DataFrame):
            return True
        else:
            return False

    def text_to_vec(self, text):
        summary = self.summarize_text(text)
        doc = self.nlp(summary)
        vec = doc.vector
        return F.relu(self.text_reducer(torch.tensor(vec).to(self.device)))

    def forward(self, input, mask=None):
        sample_embeddings = []

        # Check if input is a time series (e.g., DataFrame)
        if self.is_time_series_value(input):
            df_history = input.iloc[:-1]  # All rows except the last one
            if df_history.shape[0] > 0:
                x = torch.tensor(df_history.values, dtype=torch.float32, device=self.device).unsqueeze(0)
                x = self.time_series_embedder(x)
                sample_embeddings.append(x)
            else:
                # If there's no history, append a zero vector instead
                embed_dim = self.text_reduced_dim
                sample_embeddings.append(torch.zeros((1, embed_dim), device=self.device))

            recent_values = torch.tensor(input.iloc[-1].values, dtype=torch.float32, device=self.device).unsqueeze(0)
            sample_embeddings.append(recent_values)

        else:
            # Handle non-time-series input (e.g., one row of a DataFrame or list of features)
            for col_index, feature in enumerate(input):
                if self.is_image_value(feature):
                    feature_embed = self.embed_image(feature)
                elif self.is_text_value(feature):
                    feature_embed = self.embed_text(feature)
                elif pd.isna(feature):
                    size = len(self.map_feature.get(col_index, []))
                    feature_embed = torch.zeros((1, size), dtype=torch.float32, device=self.device)
                elif self.is_numeric_value(feature):
                    feature_embed = torch.tensor([[feature]], dtype=torch.float32, device=self.device)
                else:
                    raise ValueError(f"Unsupported feature type: {feature}")

                sample_embeddings.append(feature_embed)

        x = torch.cat(sample_embeddings, dim=1).to(DEVICE)
        if mask is not None:
            # Convert binary_mask (NumPy array) to a PyTorch tensor
            mask = torch.tensor(mask, dtype=x.dtype, device=x.device)
            x = x * mask
        x = x.squeeze(dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Final prediction head
        logits = self.logits(x).to(self.device)

        # Compute softmax probabilities
        if logits.dim() == 2:
            probs = F.softmax(logits, dim=1)
        else:
            probs = F.softmax(logits, dim=-1)

        return probs


def create_mask(model) -> np.array:
    '''
    Mask feature of the input
    :param images: input
    :return: masked input
    '''
    mapping = model.map_feature
    binary_mask = np.zeros(model.features_total)
    for key, value in mapping.items():
        if np.random.rand() < FLAGS.fraction_mask:
            # mask all the keys entries in binary mask
            for i in value:
                binary_mask[i] = 0
        else:
            for i in value:
                binary_mask[i] = 1
    return binary_mask


def create_adverserial_input(sample, label, pretrained_model):
    pretrained_model.eval()

    # Convert the sample into a proper forward-pass input
    if pretrained_model.is_time_series_value(sample):
        # Handle time-series input (e.g., a DataFrame)
        df_history = sample.iloc[:-1]
        embeddings = []

        if df_history.shape[0] > 0:
            x = torch.tensor(df_history.values, dtype=torch.float32, device=pretrained_model.device).unsqueeze(0)
            x = pretrained_model.time_series_embedder(x)
            embeddings.append(x)
        else:
            embed_dim = pretrained_model.text_reduced_dim
            embeddings.append(torch.zeros((1, embed_dim), device=pretrained_model.device))

        recent_values = torch.tensor(sample.iloc[-1].values, dtype=torch.float32,
                                     device=pretrained_model.device).unsqueeze(0)
        embeddings.append(recent_values)

        input_tensor = torch.cat(embeddings, dim=1).squeeze(0)  # shape: [dim]
    else:
        # Flat feature input (e.g., list or Series)
        sample_embeddings = []
        for col_index, feature in enumerate(sample):
            if pretrained_model.is_image_value(feature):
                feature_embed = pretrained_model.embed_image(feature)
            elif pretrained_model.is_text_value(feature):
                feature_embed = pretrained_model.embed_text(feature)
            elif pd.isna(feature):
                size = len(pretrained_model.map_feature.get(col_index, []))
                feature_embed = torch.zeros((1, size), dtype=torch.float32, device=pretrained_model.device)
            elif pretrained_model.is_numeric_value(feature):
                feature_embed = torch.tensor([[feature]], dtype=torch.float32, device=pretrained_model.device)
            else:
                raise ValueError(f"Unsupported feature type: {feature}")
            sample_embeddings.append(feature_embed)

        input_tensor = torch.cat(sample_embeddings, dim=1).squeeze(0)

    # Enable gradient tracking for adversarial attack
    input_tensor = input_tensor.detach().clone().requires_grad_(True)

    # Forward pass through model (excluding the final layer)
    x = pretrained_model.layer1(input_tensor)
    x = pretrained_model.layer2(x)
    x = pretrained_model.layer3(x)
    logits = pretrained_model.logits(x).squeeze(0)
    # Compute softmax probabilities
    if logits.dim() == 2:
        probs = F.softmax(logits, dim=1)
    else:
        probs = F.softmax(logits, dim=-1)

    loss = pretrained_model.criterion(probs.unsqueeze(0), label)

    # Backward pass to compute gradients w.r.t. input
    loss.backward()
    gradient = input_tensor.grad

    # Get the most influential feature index
    absolute_gradients = torch.abs(gradient)
    max_gradients_index = torch.argmax(absolute_gradients, dim=-1).item()

    # Map to original feature index
    for key, value in pretrained_model.map_feature.items():
        if max_gradients_index in value:
            max_gradients_index = key
            break

    # Create binary mask that zeroes out most important feature
    binary_mask = np.ones(pretrained_model.features_total)
    for key, value in pretrained_model.map_feature.items():
        if key == max_gradients_index:
            for i in value:
                binary_mask[i] = 0
    pretrained_model.train()  # Set the model to training mode

    return binary_mask


def plot_running_loss(loss_list):
    import matplotlib.pyplot as plt
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Running Loss')
    plt.show()


def compute_probabilities(j, total_episodes):
    prob_mask = 0.8 + 0.2 * (1 - j / total_episodes)  # Starts at 1, decreases to 0.8
    return prob_mask


def train_model(model,
                nepochs, X_train, y_train, X_val, y_val):
    if isinstance(X_train, list):
        pass
    else:
        X_train = X_train.to_numpy()
        X_val = X_val.to_numpy()
    val_trials_without_improvement = 0
    best_val_auc = 0
    accuracy_list = []
    loss_list = []
    num_samples = len(X_train)
    for j in range(1, nepochs):
        running_loss = 0
        random_indices = np.random.choice(num_samples, size=FLAGS.batch_size, replace=False)
        model.train()  # Set the model to training mode
        model.optimizer.zero_grad()  # Reset gradients before starting the batch
        # Process each sample in the batch
        for i in random_indices:
            input = X_train[i]
            label = torch.tensor([y_train[i]], dtype=torch.long).to(model.device)  # Convert label to tensor
            prob_mask = compute_probabilities(j, nepochs)
            # Decide the action based on the computed probabilities
            if random.random() < prob_mask:
                mask = create_mask(model)
            else:
                mask = create_adverserial_input(input, label, model)

            # Forward pass
            output = model(input, mask)
            loss = model.criterion(output, label)
            running_loss += loss.item()  # Accumulate loss for the batch

            # Backpropagate gradients
            loss.backward()

        # Update model parameters after the entire batch
        model.optimizer.step()

        average_loss = running_loss / len(random_indices)
        loss_list.append(average_loss)

        if j % FLAGS.run_validation == 0:
            new_best_val_auc = val(model, X_val, y_val, best_val_auc)
            accuracy_list.append(new_best_val_auc)
            if new_best_val_auc > best_val_auc:
                best_val_auc = new_best_val_auc
                val_trials_without_improvement = 0
            else:
                val_trials_without_improvement += 1
            if val_trials_without_improvement == FLAGS.val_trials_wo_im:
                print('Did not achieve val AUC improvement for {} trials, training is done.'.format(
                    FLAGS.val_trials_wo_im))
                break
        # print("finished " + str(j) + " out of " + str(nepochs) + " epochs")

    plot_running_loss(loss_list)


def save_model(model):
    '''
    Save the model to a given path
    :param model: model to save
    :param path: path to save the model to
    :return: None
    '''
    path = model.path_to_save
    if not os.path.exists(path):
        os.makedirs(path)
    guesser_filename = 'best_guesser.pth'
    guesser_save_path = os.path.join(path, guesser_filename)
    # save guesser
    if os.path.exists(guesser_save_path):
        os.remove(guesser_save_path)
    torch.save(model.cpu().state_dict(), guesser_save_path + '~')
    os.rename(guesser_save_path + '~', guesser_save_path)
    model.to(DEVICE)


def val(model, X_val, y_val, best_val_auc=0):
    model.eval()

    correct = 0
    y_true = []
    y_pred = []
    y_scores = []  # For probability scores for AUC

    with torch.no_grad():
        for i in range(len(X_val)):
            input = X_val[i]
            label = torch.tensor(y_val[i], dtype=torch.long).to(model.device)
            output = model(input)

            _, predicted = torch.max(output.data, 1)
            if predicted == label:
                correct += 1

            y_true.append(label.item())
            y_pred.append(predicted.item())
            y_scores.append(torch.softmax(output, dim=1)[:, 1].item())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Validation Confusion Matrix:")
    print(conf_matrix)

    # Accuracy
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    print(f'Validation Accuracy: {accuracy:.2f}')

    # AUC-ROC
    auc_roc = roc_auc_score(y_true, y_scores)
    print(f'Validation AUC-ROC: {auc_roc:.2f}')

    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auc_pc = auc(recall, precision)
    print(f'Validation AUC-PC: {auc_pc:.2f}')

    if auc_roc >= best_val_auc:
        save_model(model)
        best_val_auc = auc_roc

    return accuracy


def test(model, X_test, y_test):
    guesser_filename = 'best_guesser.pth'
    guesser_load_path = os.path.join(model.path_to_save, guesser_filename)
    guesser_state_dict = torch.load(guesser_load_path)
    model.load_state_dict(guesser_state_dict)
    model.eval()

    correct = 0
    y_true = []
    y_pred = []
    y_scores = []  # For probability scores for AUC

    with torch.no_grad():
        for i in range(len(X_test)):
            input = X_test[i]
            label = torch.tensor(y_test[i], dtype=torch.long).to(model.device)
            output = model(input)

            _, predicted = torch.max(output.data, 1)
            if predicted == label:
                correct += 1

            y_true.append(label.item())  # Assuming labels is a numpy array
            y_pred.append(predicted.item())
            y_scores.append(torch.softmax(output, dim=1)[:, 1].item())  # Get the probability for class 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Calculate and print confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate accuracy
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    print(f'Test Accuracy: {accuracy:.2f}')

    # AUC-ROC
    auc_roc = roc_auc_score(y_true, y_scores)
    print(f'AUC-ROC: {auc_roc:.2f}')

    # AUC-PC (Precision-Recall AUC)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auc_pc = auc(recall, precision)
    print(f'AUC-PC: {auc_pc:.2f}')


def main():
    '''
    Train a neural network to guess the correct answer
    :return:
    '''
    model = MultimodalGuesser()
    model.to(model.device)
    X_train, X_test, y_train, y_test = train_test_split(model.X,
                                                        model.y,
                                                        test_size=0.1,
                                                        random_state=24)

    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.1,
                                                      random_state=24)

    train_model(model, FLAGS.num_epochs,
                X_train, y_train, X_val, y_val)

    test(model, X_test, y_test)


if __name__ == "__main__":
    os.chdir(FLAGS.directory)
    main()
