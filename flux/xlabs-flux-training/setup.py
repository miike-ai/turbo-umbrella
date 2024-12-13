import os
import subprocess

repo_url = "https://github.com/miike-ai/turbo-umbrella.git"
repo_name = "turbo-umbrella"
training_dir = os.path.join(repo_name, "flux", "xlabs-flux-training")
training_script = "train_flux_deepspeed_controlnet.py"
config_path = "train_configs/test_canny_controlnet.yaml"

try:
    # Clone the repository if it doesn't exist
    if not os.path.exists(repo_name):
        print("Cloning repository...")
        subprocess.run(["git", "clone", repo_url], check=True)
    else:
        print(f"Repository {repo_name} already exists, skipping clone.")

    # Change directory to the training folder
    os.chdir(training_dir)

    # Install dependencies
    print("Installing dependencies from requirements.txt...")
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

    # Run the training script
    print("Launching training script...")
    subprocess.run([
        "accelerate", "launch", training_script,
        "--config", config_path
    ], check=True)

    print("Training completed successfully!")

except subprocess.CalledProcessError as e:
    print(f"An error occurred while running the script: {e}")
    exit(1)