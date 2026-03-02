import subprocess
import time
import sys
from datetime import datetime

import os
from dotenv import load_dotenv

# Postgres and checkpoint from env values
load_dotenv(".env.postgres")
load_dotenv(".env.postgres.keys")
load_dotenv(".env.checkpoints")
load_dotenv("/app/env_folder/.env.postgres") 
load_dotenv("/app/env_postgres_keys")
load_dotenv("/app/env_checkpoints")

print(os.getenv("ML_WRITER_USER"))
print(os.getenv("ML_WRITER_PW"))
print(os.getenv("HOST"))
print(os.getenv("TRAIN_DATABASE"))
print(os.getenv("TRAIN_LOG_TABLE"))

def main():
    try:
        print("Running preprocessing...")
        subprocess.run([sys.executable, "setup_training.py"], check=True)
        print("Processing successful!")

        print("\nTraining the model...")
        subprocess.run([sys.executable, "train_model.py"], check=True)
        print("Training successful!")

    except subprocess.CalledProcessError as e:
        print(f"Step failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    start_time = time.time()
    dt = datetime.fromtimestamp(start_time)
    print(f"Time begins now {dt}")

    main()

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60


    print(f"Execution time: {minutes} min {seconds:.2f} sec")
