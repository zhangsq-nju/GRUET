from pathlib import Path

from omegaconf import OmegaConf

METHODS = (
    "ppl", "predictive_entropy", "semantic_entropy", "sentence_sar",
    "semantic_density", "p_true", "cot_uq_me", "verbalized_2s",
    "eig_val_laplacian", "deg_mat", "eccentricity", "cot_uq_sp",
    "topouq", "gut_up",
)
REASONING_METHODS = ("topouq", "cot_uq_me", "cot_uq_sp")
ESTIMATOR_CONFIGS = ("gut_up",)
GUTQ_REVISION = "7d012151c3d851bbd24cd2ef83ba259dbdaa5b0b"
CONF = Path(__file__).resolve().parents[1] / "conf"


def load_runs() -> list[dict]:
    return OmegaConf.to_container(
        OmegaConf.load(CONF / "release" / "experiments.yaml"), resolve=True
    )["runs"]


def load_estimator_config(name: str) -> dict:
    if name not in ESTIMATOR_CONFIGS:
        raise ValueError(f"unknown global configuration: {name}")
    return OmegaConf.to_container(
        OmegaConf.load(CONF / "estimator" / f"{name}.yaml"), resolve=True
    )


def load_models(path: Path | None = None) -> dict[str, dict]:
    models = OmegaConf.load(CONF / "release" / "models.yaml")
    if path is not None:
        models = OmegaConf.merge(models, OmegaConf.load(path))
    return OmegaConf.to_container(models, resolve=True)
