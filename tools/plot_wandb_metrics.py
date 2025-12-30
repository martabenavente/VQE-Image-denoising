import wandb
import pandas as pd
import matplotlib.pyplot as plt

def plot_wandb_metric(
    entity: str,
    project: str,
    run_ids: list,
    metric: str,
    x_key: str = "_step",
    samples: int = 50,
    title: str | None = None,
    save_path: str | None = None,
):
    api = wandb.Api()
    plt.figure()

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")

        history = run.history(keys=[x_key, metric], samples=samples)
        df = pd.DataFrame(history).dropna(subset=[x_key, metric])
        df = df.sort_values(x_key)

        if df.empty:
            print(f"Run {run_id} vacío")
            continue

        plt.plot(df[x_key], df[metric], label=run.name or run_id)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    if title:
        plt.title(title, fontsize=16)

    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

## "9i6147kh", "yza4lhfz", "i8n54mib", "3vgy3hul", "a6a4s2f2", "vw5twt42", "2kj65eqh", "pi05mtdl", "dlk2udo3",
## "ipg51vvf", "ola2f289", "h8rb23wf", "p3v5d88w", "obl5rxe4", "y6ecbctm"

ENTITY = "marta-benavente-vilas-university-of-southern-denmark"
PROJECT = "vqe-image-denoising"
RUN_ID_VQA_BATCHES = ["yza4lhfz", "2kj65eqh", "9i6147kh"]
RUN_ID_VQA_SEEDS = ["2kj65eqh", "3vgy3hul", "i8n54mib"]
RUN_ID_VQA_QUBITS = ["y6ecbctm", "ola2f289", "ipg51vvf", "dlk2udo3"]
RUN_ID_VQA_LAYERS = ["y6ecbctm", "pi05mtdl", "2kj65eqh", "vw5twt42", "a6a4s2f2"]
RUN_ID_VQA_NOISE = ["y6ecbctm", "obl5rxe4", "h8rb23wf", "p3v5d88w"]
RUN_ID_CL = ["53r8gi4n", "y6ecbctm"] #"z01skhmq", "le8hlcmd", "hryp48k8"]
RUN_VQA = ["y6ecbctm", "ola2f289", "ipg51vvf", "dlk2udo3", "pi05mtdl", "2kj65eqh", "vw5twt42", "a6a4s2f2", "obl5rxe4", "h8rb23wf", "p3v5d88w"]


# plot_wandb_metric(
#     entity=ENTITY,
#     project=PROJECT,
#     run_ids=RUN_ID_VQA_BATCHES,
#     metric="train/loss",
#     title="Train Loss",
#     save_path="train_loss_VQA_batches.png"
# )

# plot_wandb_metric(
#     entity=ENTITY,
#     project=PROJECT,
#     run_ids=RUN_ID_VQA_SEEDS,
#     metric="train/loss",
#     title="Train Loss",
#     save_path="train_loss_VQA_seeds.png"
# )

# plot_wandb_metric(
#     entity=ENTITY,
#     project=PROJECT,
#     run_ids=RUN_ID_VQA_QUBITS,
#     metric="val/psnr",
#     title="Validation PSNR",
#     save_path="val_psnr_VQA_qubits.png"
# )

# plot_wandb_metric(
#     entity=ENTITY,
#     project=PROJECT,
#     run_ids=RUN_ID_VQA_QUBITS,
#     metric="val/ssim",
#     title="Validation SSIM",
#     save_path="val_ssin_VQA_qubits.png"
# )

# plot_wandb_metric(
#     entity=ENTITY,
#     project=PROJECT,
#     run_ids=RUN_ID_VQA_LAYERS,
#     metric="val/psnr",
#     title="Validation PSNR",
#     save_path="val_psnr_VQA_layers.png"
# )

# plot_wandb_metric(
#     entity=ENTITY,
#     project=PROJECT,
#     run_ids=RUN_ID_VQA_LAYERS,
#     metric="val/ssim",
#     title="Validation SSIM",
#     save_path="val_ssin_VQA_layers.png"
# )

# plot_wandb_metric(
#     entity=ENTITY,
#     project=PROJECT,
#     run_ids=RUN_ID_VQA_NOISE,
#     metric="val/psnr",
#     title="Validation PSNR",
#     save_path="val_psnr_VQA_noise.png"
# )

# plot_wandb_metric(
#     entity=ENTITY,
#     project=PROJECT,
#     run_ids=RUN_ID_VQA_NOISE,
#     metric="val/ssim",
#     title="Validation SSIM",
#     save_path="val_ssin_VQA_noise.png"
# )

plot_wandb_metric(
    entity=ENTITY,
    project=PROJECT,
    run_ids=RUN_ID_CL,
    metric="val/psnr",
    title="Validation PSNR",
    save_path="val_psnr_CL.png"
)

plot_wandb_metric(
    entity=ENTITY,
    project=PROJECT,
    run_ids=RUN_ID_CL,
    metric="val/ssim",
    title="Validation SSIM",
    save_path="val_ssin_CL.png"
)
