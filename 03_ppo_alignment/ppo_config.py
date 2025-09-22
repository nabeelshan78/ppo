import torch

BASE_MODEL_NAME = "gpt2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RM_CHECKPOINT_PATH = "./reward_model_output_new"
FULL_SFT_CHECKPOINT_PATH = "./full_sft_final_checkpoint"


# --- PPO Configuration ---
PPO_CONFIG = {
    "model_name": BASE_MODEL_NAME,
    "learning_rate": 1.41e-5,
    "batch_size": 16,
    "mini_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "kl_penalty": "kl",
    "init_kl_coef": 0.2,
    "adap_kl_ctrl": True
}