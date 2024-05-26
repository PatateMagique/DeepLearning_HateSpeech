import torch
import random
import inspect
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        text = item["text"]
        label = item["label"]

        # Process the image, pt = PyTorch tensors
        processed_image = self.processor(images=image, return_tensors="pt")["pixel_values"]

        # Process the text, pt = PyTorch tensors
        processed_text = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)

        # Process the label
        label = torch.tensor(label)

        return processed_image, processed_text, label

class ModelForClassification(nn.Module):
    def __init__(self, model, model_name = None):
        super(ModelForClassification, self).__init__()
        self.model = model
        self.model_name = model_name
        if model_name == "BLIP":
            self.classifier = nn.Linear(self.model.config.text_config.hidden_size, 2)
        else:
            self.classifier = nn.Linear(self.model.config.projection_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()  # Loss function for classification

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        if self.model_name == "BLIP":
            outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, return_dict=True)
            text_embeds = outputs.last_hidden_state.mean(dim=1)
            image_embeds = outputs.pooler_output if 'pooler_output' in outputs else outputs.last_hidden_state.mean(dim=1)
        else: # For CLIP and ALIGN models
            outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
            text_embeds = outputs.text_embeds
            image_embeds = outputs.image_embeds

        combined_embeds = text_embeds + image_embeds  # Combining embeddings; you can choose a different strategy

        logits = self.classifier(combined_embeds)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return logits, loss

def create_model(model_name, model, topic_initial, device):
    """
    Create a model with specified layers unfrozen for training.
    
    Args:
        model_name (str): Name of the model ("CLIP", "BLIP", or "ALIGN").
        model (nn.Module): The model instance.
        topic_initial (str): The topic initial.
        device (torch.device): The device to move the model to.
        layers_to_unfreeze (list, optional): List of layer names or indices to unfreeze. Defaults to None.
    
    Returns:
        nn.Module: The created model with specified layers unfrozen.
    """

    if model_name == "CLIP":
        # For CLIP, unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True

        # Add the classification head
        model = ModelForClassification(model, model_name)

        # Unfreeze the classification head
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif model_name == "BLIP":
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Add the classification head
        model = ModelForClassification(model, model_name)

        # Unfreeze the classification head
        for param in model.classifier.parameters():
            param.requires_grad = True

        try:
            vision_layers = model.model.vision_model.encoder.layers
            text_layers   = model.model.text_decoder.bert.encoder.layer
        except AttributeError:
            raise AttributeError("Unable to locate the layers in the vision model.")

        for i in range(7, 11):
            for param in vision_layers[i].parameters():
                param.requires_grad = False
            for param in text_layers[i].parameters():
                param.requires_grad = False
    
    elif model_name == "ALIGN":

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Add the classification head
        model = ModelForClassification(model, model_name)

        # Unfreeze the classification head
        for param in model.classifier.parameters():
            param.requires_grad = True

        try:
            vision_layers  = model.model.vision_model.encoder.blocks
            text_layers = model.model.text_model.encoder.layer
        except AttributeError:
            raise AttributeError("Unable to locate the layers in the vision model.")

        for i in range(45, 54):
            for param in vision_layers[i].parameters():
                param.requires_grad = True
        for i in range(7, 11):
            for param in text_layers[i].parameters():
                param.requires_grad = True

    model = model.to(device)
    
    print(f"Successfully created {model_name}_{topic_initial}.")

    return model

def unfreeze_layers(model, layers):
        for name, param in model.named_parameters():
            if any(layer in name for layer in layers):
                param.requires_grad = True

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    
    # Collect all text dictionaries
    text_dicts = [item[1] for item in batch]
    
    # Create a new dictionary to store the concatenated texts
    collated_texts = {}
    
    # Iterate over keys in the first text dictionary to handle padding
    for key in text_dicts[0].keys():
        max_length = max([text_dict[key].shape[1] for text_dict in text_dicts])
        padded_texts = [torch.nn.functional.pad(text_dict[key], (0, max_length - text_dict[key].shape[1])) for text_dict in text_dicts]
        collated_texts[key] = torch.cat(padded_texts, dim=0)
    
    labels = torch.tensor([item[2] for item in batch])
    return images, collated_texts, labels

@torch.no_grad()
def test(model: nn.Module, loader: DataLoader, device: torch.device, return_confidences: bool = False):
    """The test function, computes the F1 score of the current model on the test_loader

    Args:
        model (nn.Module): The model to evaluate
        loader (DataLoader): The test data loader to iterate on the dataset to test

    Returns:
        f1 (float): The F1 score on the given dataset
        loss (float): Averaged loss on the given dataset
        confidences (list of float): List of confidences for each prediction
    """
    model.eval()
    
    preds_dict = {"preds": torch.Tensor(), "labels": torch.Tensor(), 'losses': torch.Tensor(), 'confidences': torch.Tensor()}
    with torch.no_grad():
        for batch in loader:
            images, texts, labels = batch
            images = images.squeeze(1).to(device)
            texts = {key: value.to(device) for key, value in texts.items()}
            labels = labels.to(device)
            
            # Get the parameters of the model's forward method
            forward_params = inspect.signature(model.forward).parameters
            
            # Forward and loss
            if 'labels' in forward_params:
                preds, _ = model(pixel_values=images, input_ids=texts["input_ids"], attention_mask=texts["attention_mask"], labels=labels)
            else:
                preds, _ = model(pixel_values=images, input_ids=texts["input_ids"], attention_mask=texts["attention_mask"])
            
            loss = F.cross_entropy(preds, labels)
            
            # Sigmoid for binary classification to get confidence score
            confidences = torch.sigmoid(preds)[:, 1]  # Get confidence for class 1
            
            # Store values back to the CPU before storing them in preds_dict. 
            preds_dict["preds"] = torch.cat([preds_dict["preds"], preds.argmax(1).cpu()])
            preds_dict["labels"] = torch.cat([preds_dict["labels"], labels.cpu()])
            preds_dict["losses"] = torch.cat([preds_dict["losses"], loss[None].cpu()])
            preds_dict["confidences"] = torch.cat([preds_dict["confidences"], confidences.cpu()])
    
    # Compute metric and loss
    f1 = f1_score(preds_dict["labels"], preds_dict["preds"], average="macro")
    loss = preds_dict["losses"].mean()
    
    # Convert confidences to numpy array
    confidences = preds_dict["confidences"].numpy()
    
    if return_confidences:
        return f1, loss, confidences
    else:
        return f1, loss
    
def test_ensemble(loader: DataLoader, ensemble_predictions: list):
    """The test function, computes the F1 score using ensemble predictions on the test_loader

    Args:
        loader (DataLoader): The test data loader to iterate on the dataset to test
        ensemble_predictions (list of int): List of ensemble predictions for each sample in the dataset

    Returns:
        f1 (float): The F1 score using the ensemble predictions
    """
    labels = []

    for batch in loader:
        _, _, batch_labels = batch
        labels.extend(batch_labels.numpy())

    # Compute the F1 score using the ensemble predictions and true labels
    f1 = f1_score(labels, ensemble_predictions, average="macro")
    
    return f1

def train(model : nn.Module, train_loader : DataLoader, val_loader : DataLoader, n_epochs : int, optimizer : torch.optim.Optimizer, device : torch.device, scheduler = None):
    """Trains the neural network self.model for n_epochs using a given optimizer on the training dataset.
    Outputs the best model in terms of F1 score on the validation dataset.

    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): The training dataloader to iterate on the training dataset
        val_loader (DataLoader): The validation dataloader to iterate on the validation dataset
        n_epochs (int): The number of epochs, i.e. the number of time the model should see each training example
        optimizer (torch.optim.Optimizer): The optimizer function to update the model parameters

    Returns:
        best_model (nn.Module): Best model state dictionary 
        best_f1 (float): Best F1-score on the validation set
        best_epoch (int): Best epoch on validation set
        val_f1s (list of floats): (n_epochs, ) F1-scores for all epochs
        val_losses (list of floats): (n_epochs, ) Losses for all validation epochs
        train_losses(list of floats): (n_epochs, ) Losses for all training epochs
    """

    # Initialize variable to return
    best_model = model.state_dict()
    best_epoch = 0
    best_f1 = 0
    train_losses = []
    val_losses = []
    val_f1s = []
    f1 = 0
    val_loss = 0
    try:
        for epoch in range(n_epochs):
            running_loss = 0.0
            for batch in train_loader:
                images, texts, labels = batch
                images = images.squeeze(1).to(device)
                texts = {key: value.to(device) for key, value in texts.items()}
                labels = labels.to(device)

                # Forward pass

                # Get the parameters of the model's forward method
                forward_params = inspect.signature(model.forward).parameters

                # Check if 'labels' is in the parameters
                if 'labels' in forward_params:
                    outputs, _ = model(pixel_values=images, input_ids=texts["input_ids"], attention_mask=texts["attention_mask"], labels=labels)
                else:
                    outputs, _ = model(pixel_values=images, input_ids=texts["input_ids"], attention_mask=texts["attention_mask"])
                
                loss = F.cross_entropy(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_losses.append(running_loss / len(train_loader))

            with torch.no_grad():
                [f1, val_loss] = test(model, val_loader, device)
                val_f1s.append(f1)
                val_losses.append(val_loss)

                scheduler.step(val_loss)
                current_lr = scheduler.optimizer.param_groups[0]['lr']

                if f1 > best_f1: 
                    best_model = model.state_dict()
                    best_f1 = f1
                    best_epoch = epoch + 1

            print(f'Epoch {epoch + 1} - F1: {f1:.3f} - Validation Loss: {val_loss:.3f} - Training Loss: {loss:.3f} - LR: {current_lr:.6f}')
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        return best_model, best_f1, best_epoch, val_f1s, val_losses, train_losses

    return best_model, best_f1, best_epoch, val_f1s, val_losses, train_losses


    
def plot_training(best_epoch: int, val_accs: list, val_loss: list, train_loss: list, model_name: str):
    """Plot training results of linear classifier
    
    Args:
        best_epoch (int): Best epoch
        val_accs (List): (E,) list of validation measures for each epoch
        val_loss (List): (E,) List of validation losses for each epoch
        train_loss (List): (E,) List of training losses for each epoch
    """

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{model_name} training results", fontsize=16)

    es = np.arange(1, len(val_accs)+1)
    # Plot F1 score
    axes[0].plot(es, val_accs, label="Val")
    axes[0].vlines(best_epoch, ymin=np.min(val_accs), ymax=np.max(val_accs), color='k', ls='--', label="Best epoch")
    axes[0].set_xlabel("Training steps")
    axes[0].set_ylabel("F1-score")
    axes[0].set_title("F1-score")
    axes[0].legend()

    # Plot losses
    axes[1].plot(es, val_loss, label="Val")
    axes[1].plot(es, train_loss, label="Train")
    axes[1].vlines(best_epoch, ymin=np.min(train_loss), ymax=np.max(val_loss), color='k', ls='--', label="Best epoch")
    axes[1].set_xlabel("Training steps")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Losses")
    axes[1].legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust the top padding to make room for the suptitle

def predict_image(model: nn.Module, loader: DataLoader, device: torch.device, index: int, show: bool = True):

    model.eval()
    
    # Only process the first batch
    for batch in loader:
        images, texts, labels = batch
        images = images.squeeze(1).to(device)
        texts = {key: value.to(device) for key, value in texts.items()}
        labels = labels.to(device)
        
        # Forward and loss
        preds, _ = model(pixel_values=images, input_ids=texts["input_ids"], attention_mask=texts["attention_mask"])

        true_label = labels[index].item()
        predicted_label = preds.argmax(1)[index].item()

        if show:
            # Plot the specified image, its prediction, and the true label
            image = images[index].cpu().numpy().transpose(1, 2, 0)  # Assuming images are in (C, H, W) format
            # Ensure image is in the correct range
            image = np.clip(image, 0, 1)

            plt.imshow(image)
            plt.title(f'True Label: {true_label}, Predicted Label: {predicted_label}')
            plt.axis('off')
            plt.show()
        
        break  # Exit after processing the first batch

def predict_image2(model: nn.Module, loader: DataLoader, device: torch.device, index: int, show: bool = True, model_name: str = None, topic_name: str = None):
    """Predict and plot 4 images from the dataset starting at a given index, displaying the confidence and labels.

    Args:
        model (nn.Module): The model to use for predictions.
        loader (DataLoader): The data loader to fetch the dataset.
        device (torch.device): The device to run the model on.
        start_index (int): The starting index to fetch images from the batch.
        show (bool): Whether to display the images or not.
    """
    model.eval()

    # Convert the loader to a list and select a random batch
    batches = list(loader)
    batch = random.choice(batches)

    images, texts, labels = batch
    images = images.squeeze(1).to(device)
    texts = {key: value.to(device) for key, value in texts.items()}
    labels = labels.to(device)
    
    # Forward pass
    preds, _ = model(pixel_values=images, input_ids=texts["input_ids"], attention_mask=texts["attention_mask"])
    
    # Sigmoid for binary classification to get confidence score
    confidences = torch.sigmoid(preds)[:, 1].cpu().detach().numpy()  # Confidence for class 1
    
    if show:
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'Model: {model_name}_{topic_name}')  # Add title to the entire plot
        for i in range(4):
            idx = index + i
            if idx >= len(images):
                break
            true_label = labels[idx].item()
            predicted_label = preds.argmax(1)[idx].item()
            confidence = confidences[idx]

            # Plot the image
            image = images[idx].cpu().numpy().transpose(1, 2, 0)  # Assuming images are in (C, H, W) format
            image = np.clip(image, 0, 1)  # Ensure image is in the correct range

            axs[i].imshow(image)
            axs[i].set_title(f'True: {true_label}, Pred: {predicted_label}\nConfidence: {confidence:.2f}')
            axs[i].axis('off')
        plt.show()

def get_ensemble_predictions(confidences_A, confidences_W, confidences_M):
    ensemble_predictions = []
    num_samples = len(confidences_A)
    
    for i in range(num_samples):
        # if one model as a confidence above 0.5, we predict 1
        if confidences_A[i] > 0.5 or confidences_W[i] > 0.5 or confidences_M[i] > 0.5:
            ensemble_predictions.append(1)
        else:
            ensemble_predictions.append(0)
    
    return ensemble_predictions