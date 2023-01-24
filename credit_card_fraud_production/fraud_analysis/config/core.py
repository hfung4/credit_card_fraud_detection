from pathlib import Path
from typing import List

from pydantic import BaseModel
from strictyaml import YAML, load

import fraud_analysis

# Project Directories
PACKAGE_ROOT = Path(fraud_analysis.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
RAW_DATA_DIR = PACKAGE_ROOT / "datasets/raw"
TEST_DATA_DIR = PACKAGE_ROOT / "datasets/test"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"
PIPELINE_DEBUG_DIR = PACKAGE_ROOT / "outputs/debug"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    raw_data_file: str
    saved_pipeline_filename: str
    # Debug
    DEBUG_PIPELINE: bool
    SAVED_COL_NAME_FILE: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    # general
    DV: str
    SELECTED_FEATURES: List[str]
    TEST_SIZE: float
    NFOLDS: int
    RANDOM_STATE: int
    SCORING: str

    # Pipeline hyerparameters
    DROP_NZV_TOL: float
    VARS_TO_STD: List[str]
    OVERSAMPLING_STRATEGY: float
    UNDERSAMPLING_STRATEGY: float

    CLF__N_ESTIMATORS: int
    CLF__MAX_FEATURES: str
    CLF__MIN_SAMPLES_SPLIT: int


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


# Functions ------------------------------------------------------


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path=None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(
                conf_file.read()
            )  # use the load function from strictyaml
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
