# main.py
from glob import glob
import os
import uuid
import torch

from trainer import Trainer
from model_utils import get_data_loaders, SimpleCNN, export_model_onnx
from callbacks import AzureCheckpointCallback, PostgresLoggingCallback

print("Begin model training...")

# -------------------- Config --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = glob("*train_test_split*")[0]  # Automatically find the split dataset directory
BATCH_SIZE = 8
EPOCHS = 5
LR = 0.1
STEP_SIZE = 10

# -------------------- Load Data --------------------
train_loader, test_loader, num_classes = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)

# -------------------- Initialize Model --------------------
model = SimpleCNN(num_classes=num_classes).to(DEVICE)
metadata = {"project_name": "RPS_Classification"}
run_id=str(uuid.uuid4())

# -------------------- Initialize Callbacks --------------------
callbacks = [
    AzureCheckpointCallback(
        project_name=metadata.get("project_name")
    ),
    PostgresLoggingCallback(
        metadata=metadata,
        run_id=run_id
    )
]

# -------------------- Initialize Trainer --------------------
trainer = Trainer(
    model=model,
    train_data_loader=train_loader,
    test_data_loader=test_loader,
    device=DEVICE,
    epochs=EPOCHS,
    lr=LR,
    step_size=STEP_SIZE,
    callbacks=callbacks
)

# -------------------- Train --------------------
trainer.fit()

# -------------------- Export to ONNX --------------------
export_model_onnx(model, output_path="rps_model.onnx")