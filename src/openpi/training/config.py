"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro
import numpy as np

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
import openpi.policies.rainbow_policy as rainbow_policy
from openpi.policies.rainbow_policy import RainbowInputs8DOF, RainbowOutputs8DOF

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If true, will disable syncing the dataset from the Hugging Face Hub. Allows training on local-only datasets.
    local_files_only: bool = False


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # TODO(karl): comment this out once we have updated the Libero checkpoints to not use
        # the delta action transform
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotRainbowDataConfig(DataConfigFactory):
    """Data config for the Rainbow robot."""
    """ Assumes no Pi adaptation and no delta modelization of actions"""

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)  # Use action to match dataset

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[rainbow_policy.RainbowInputs(action_dim=model_config.action_dim)],
            outputs=[rainbow_policy.RainbowOutputs()],
        )
      
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
    

@dataclasses.dataclass(frozen=True)
class LeRobotRainbowDataConfigRotated(LeRobotRainbowDataConfig):
    """Data config for Rainbow robot with 180-degree rotated head camera."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[RainbowInputsWithRotation(action_dim=model_config.action_dim)],
            outputs=[rainbow_policy.RainbowOutputs()],
        )
        
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


# Define this class at module level, not inside a method
class RainbowInputsWithRotation(rainbow_policy.RainbowInputs):
    """Rainbow inputs with 180-degree rotation of the head camera image."""
    
    def __call__(self, data: dict) -> dict:
        # Get the head image before standard processing
        if "observation.image.head" in data:
            # Rotate image using NumPy (more consistent with codebase)
            img = np.asarray(data["observation.image.head"])
            # 180 degree rotation = flip both horizontally and vertically
            data["observation.image.head"] = np.flip(np.flip(img, axis=1), axis=2)
        
        # Call the parent method to do the standard processing
        return super().__call__(data)


@dataclasses.dataclass(frozen=True)
class LeRobotRainbowDataConfigRotated8DOF(LeRobotRainbowDataConfigRotated):
    """Data config for Rainbow robot with 180-degree rotated head camera and 8-DOF."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[RainbowInputs8DOF(action_dim=model_config.action_dim)],
            outputs=[RainbowOutputs8DOF()],
        )
        
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRainbowDataConfigRotated224(LeRobotRainbowDataConfig):
    """Data config for Rainbow robot with 180-degree rotated head camera and 224x224 input images."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[rainbow_policy.RainbowInputsRotated224(action_dim=model_config.action_dim)],
            outputs=[rainbow_policy.RainbowOutputs()],
        )
        
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 8
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 10000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 10000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")

_CONFIGS = [

# https://huggingface.co/datasets/HumanoidTeam/R2_VLA_merged_6tasks_100_episodes_v1_2025_05_22
TrainConfig(
    name="pi0_fast_finetune_r2_6skills_250t_32bz",
    exp_name="exp_r2_6skills_32bz",        
    model=pi0_fast.Pi0FASTConfig(
        action_dim=16,       
        action_horizon=50,
        max_token_len=250,
    ),
    data=LeRobotRainbowDataConfig(
        repo_id="HumanoidTeam/R2_VLA_merged_6tasks_100_episodes_v1_2025_05_22",
        base_config=DataConfig(
            local_files_only=False,
            prompt_from_task=True,          
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=500,
        peak_lr=5e-5,
        decay_steps=30000,
        decay_lr=5e-6,
    ),
    ema_decay=None,
    batch_size=32,          
    num_train_steps=120_000, 
    num_workers=8,
),
     # After Eight + Quality Street with 180-degree rotated head camera
TrainConfig(
    name="pi0_fast_rainbow_poc_aftereight_qs_rotated_250t_256bz",
    exp_name="exp_rotated_head_fix_32bz",  # Add an experiment name
    model=pi0_fast.Pi0FASTConfig(
        action_dim=16,
        action_horizon=50,
        max_token_len=250,
    ),
    data=LeRobotRainbowDataConfigRotated(  # Using the new rotated config
        repo_id="HumanoidTeam/after_eight_deea_and_quality_street_arjun",
        base_config=DataConfig(
            local_files_only=False,
            prompt_from_task=True,
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
    num_train_steps=120_000,
    batch_size=32,  # Using your tested batch size for H100
    num_workers=8,  # Increased for faster data loading
),


  # After Eight + Quality Street (rotated head-cam, original image size)
TrainConfig(
    name="pi0_fast_rainbow_poc_aftereight_qs_rotated_250t_384bz_40k",
    exp_name="exp_rotated_head_384bz_40k",
    model=pi0_fast.Pi0FASTConfig(
        action_dim=16,
        action_horizon=50,
        max_token_len=250,
    ),
    data=LeRobotRainbowDataConfigRotated(
        repo_id="HumanoidTeam/after_eight_deea_and_quality_street_arjun",
        base_config=DataConfig(
            local_files_only=False,
            prompt_from_task=True,
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_fast_base/params"
    ),

    # LR schedule scaled for 384 BZ
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1500,
        peak_lr=4.5e-4,
        decay_steps=28000,
        decay_lr=4e-5,
    ),
    ema_decay=None,

    batch_size=384,
    num_train_steps=40_000,
    num_workers=8,
)

    # https://huggingface.co/datasets/HumanoidTeam/VLA_merged_7tasks_100_episodes_v1_13052025
TrainConfig(
    name="pi0_fast_lora_aftereight_qs_rotated_250t_512bz",
    exp_name="exp_rotated_head_lora",          # new experiment name
    model=pi0_fast.Pi0FASTConfig(
        action_dim=16,
        action_horizon=50,
        max_token_len=250,
        paligemma_variant="gemma_2b_lora",     # ← LoRA enabled
    ),
    data=LeRobotRainbowDataConfigRotated(
        repo_id="HumanoidTeam/after_eight_deea_and_quality_street_arjun",
        base_config=DataConfig(
            local_files_only=False,
            prompt_from_task=True,
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_fast_base/params"
    ),

    # Freeze everything except the LoRA adapters & layer-norms
    freeze_filter=pi0_fast.Pi0FASTConfig(
        action_dim=16,
        action_horizon=50,
        max_token_len=250,
        paligemma_variant="gemma_2b_lora",
    ).get_freeze_filter(),

    # Same tuned cosine schedule you used in the 5-skills run
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=500,
        peak_lr=5e-5,
        decay_steps=30000,
        decay_lr=5e-6,
    ),
    ema_decay=None,           # keep EMA off for LoRA
    batch_size=512,
    num_train_steps=240_000,
    num_workers=8,
),

    # https://huggingface.co/datasets/HumanoidTeam/VLA_merged_7tasks_100_episodes_v1_13052025
TrainConfig(
    name="pi0_fast_finetune_7skills_250t_256bz_h100",
    exp_name="exp_pi0_fast_finetune_7skills_250t_256bz_h100",
    model=pi0_fast.Pi0FASTConfig(
        action_dim=16,
        action_horizon=50,
        max_token_len=250,
        # No paligemma_variant = LoRA disabled
    ),
    data=LeRobotRainbowDataConfig(
        repo_id="HumanoidTeam/VLA_merged_7tasks_100_episodes_v1_13052025",
        base_config=DataConfig(
            local_files_only=False,
            prompt_from_task=True,
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=500,
        peak_lr=5e-5,
        decay_steps=30000,
        decay_lr=5e-6,
    ),
    ema_decay=None,
    batch_size=256,
    num_train_steps=100_000,
    num_workers=8,
),

# https://huggingface.co/datasets/HumanoidTeam/five_tasks_08_05_25
TrainConfig(
    name="pi0_fast_lora_tuned_lr_5skills",
    exp_name="exp_pi0_fast_lora_tuned_lr_5skills",  
    model=pi0_fast.Pi0FASTConfig(
        action_dim=16,
        action_horizon=50,
        max_token_len=250,
        paligemma_variant="gemma_2b_lora",
    ),
    data=LeRobotRainbowDataConfig(
        repo_id="HumanoidTeam/five_tasks_08_05_25",
        base_config=DataConfig(
            local_files_only=False,
            prompt_from_task=True,
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
    freeze_filter=pi0_fast.Pi0FASTConfig(
        action_dim=16,
        action_horizon=50,
        max_token_len=250,
        paligemma_variant="gemma_2b_lora",
    ).get_freeze_filter(),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=500,
        peak_lr=5e-5,
        decay_steps=30000,
        decay_lr=5e-6,
    ),
    ema_decay=None,
    batch_size=256,
    num_train_steps=120_000,
    num_workers=8,
),

TrainConfig(
    name="pi0_fast_lora_tuned_lr_4skills",
    exp_name="exp_pi0_fast_lora_tuned_lr_4skills",  # ← Required or will raise ValueError
    model=pi0_fast.Pi0FASTConfig(
        action_dim=16,
        action_horizon=50,
        max_token_len=250,
        paligemma_variant="gemma_2b_lora",
    ),
    data=LeRobotRainbowDataConfig(
        repo_id="HumanoidTeam/six_skills_v2",
        base_config=DataConfig(
            local_files_only=False,
            prompt_from_task=True,
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
    freeze_filter=pi0_fast.Pi0FASTConfig(
        action_dim=16,
        action_horizon=50,
        max_token_len=250,
        paligemma_variant="gemma_2b_lora",
    ).get_freeze_filter(),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=500,
        peak_lr=5e-5,
        decay_steps=30000,
        decay_lr=5e-6,
    ),
    ema_decay=None,
    batch_size=256,
    num_train_steps=250_000,
    num_workers=8,
),
    # https://huggingface.co/datasets/HumanoidTeam/four_tasks_dataset_03_05_25
TrainConfig(
    name="pi0_fast_lora_multitask_4skills_250t_256bz_h100",
    model=pi0_fast.Pi0FASTConfig(
        action_dim=16,
        action_horizon=50,
        max_token_len=250,
        paligemma_variant="gemma_2b_lora",
    ),
    data=LeRobotRainbowDataConfig(
        repo_id="HumanoidTeam/four_tasks_dataset_03_05_25",
        base_config=DataConfig(
            local_files_only=False,
            prompt_from_task=True,  # <-- Read prompts from dataset files
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
    num_train_steps=60_000,
    batch_size=256,
    num_workers=8,  # Increased for faster data loading
    freeze_filter=pi0_fast.Pi0FASTConfig(
        action_dim=16,
        action_horizon=50,
        max_token_len=250,
        paligemma_variant="gemma_2b_lora",
    ).get_freeze_filter(),
    ema_decay=None,
),
    # https://huggingface.co/datasets/HumanoidTeam/your_4_task_dataset
TrainConfig(
    name="pi0_fast_multitask_4skills_250t_512bz_h200",
    model=pi0_fast.Pi0FASTConfig(
        action_dim=16,
        action_horizon=50,
        max_token_len=250,
    ),
    data=LeRobotRainbowDataConfig(
        repo_id="HumanoidTeam/four_tasks_dataset_03_05_25",
        base_config=DataConfig(
            local_files_only=False,
            prompt_from_task=True,  # <-- Read prompts from dataset files
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
    num_train_steps=60_000,
),

    # After Eight + Quality Street (128 batch, H200)
    TrainConfig(
        name="pi0_fast_rainbow_poc_aftereight_qs_deea_250t_128bz_h200",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=16,
            action_horizon=50,
            max_token_len=250,
        ),
        data=LeRobotRainbowDataConfig(
            repo_id="HumanoidTeam/after_eight_deea_and_quality_street_arjun",
            base_config=DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "s3://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
        num_train_steps=120_000,
        batch_size=384,
    ),

    # After Eight Slow (192 batch)
    TrainConfig(
        name="pi0_fast_rainbow_poc_aftereightslow_deea_250t_192bz",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=16,
            action_horizon=50,
            max_token_len=250,
        ),
        data=LeRobotRainbowDataConfig(
            repo_id="HumanoidTeam/AfterEightSlowDeea29042256",
            base_config=DataConfig(
                local_files_only=False,
                prompt_from_task=False,
            ),
            default_prompt="Pick up the After Eight box.",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "s3://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
        num_train_steps=120_000,
        batch_size=192,
    ),

    # Crumpets (384 batch)
    TrainConfig(
        name="pi0_fast_rainbow_poc_crumpets_deea_250t_384bz",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=16,
            action_horizon=50,
            max_token_len=250,
        ),
        data=LeRobotRainbowDataConfig(
            repo_id="HumanoidTeam/CrumpetsDeea24041939",
            base_config=DataConfig(
                local_files_only=False,
                prompt_from_task=False,
            ),
            default_prompt="Pick up the crumpets.",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "s3://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
        num_train_steps=120_000,
        batch_size=384,
    ),

    # Crumpets with LoRA on H100 (192 batch, paligemma variant)
    TrainConfig(
        name="pi0_fast_lora_crumpets_250t_192bz_h100",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=16,
            action_horizon=50,
            max_token_len=250,
            paligemma_variant="gemma_2b_lora",
        ),
        data=LeRobotRainbowDataConfig(
            repo_id="HumanoidTeam/CrumpetsDeea24041939",
            base_config=DataConfig(
                local_files_only=False,
                prompt_from_task=False,
            ),
            default_prompt="Pick up the crumpets.",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "s3://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
        num_train_steps=60_000,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=16,
            action_horizon=50,
            max_token_len=250,
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Quality Street
    TrainConfig(
        name="pi0_fast_rainbow_poc_qualitystreetcoder_arjun",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=16,
            action_horizon=50,
            max_token_len=250,
        ),
        data=LeRobotRainbowDataConfig(
            repo_id="HumanoidTeam/QualityStreetCoderArjun23041011",
            base_config=DataConfig(
                local_files_only=False,
                prompt_from_task=False,
            ),
            default_prompt="Pick up the colorful octagonal candy tin.",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "s3://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
        num_train_steps=120_000,
        batch_size=192,
    ),

    TrainConfig(
        name="pi0_aloha",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(


        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    # https://huggingface.co/datasets/HumanoidTeam/LargeWalkersArjun24042107
TrainConfig(
    name="pi0_fast_rainbow_poc_largewalkers_arjun",
    model=pi0_fast.Pi0FASTConfig(
        action_dim=16,  # Rainbow has 16 action dimensions
        action_horizon=10,
        max_token_len=180,  # Single-arm robot, so 180 should be sufficient
    ),
    data=LeRobotRainbowDataConfig(
        repo_id="HumanoidTeam/LargeWalkersArjun24042107",
        base_config=DataConfig(
            local_files_only=False,
            prompt_from_task=False,  # Use task field from dataset for prompts
        ),
        default_prompt="Pick up the bag of chips.",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
    num_train_steps=50_000,
),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Fine-tuning Aloha configs. Doritos test.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_doritos_pp",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="HumanoidTeam/Anastacia_1DoritosIn1box",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="Pick the blue Doritos chip bag and place it in the empty box.",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
# https://huggingface.co/datasets/HumanoidTeam/QuaversAnastacia24040802
TrainConfig(
    name="pi0_fast_rainbow_poc_quavers_anastacia_250t_128bz",
    model=pi0_fast.Pi0FASTConfig(
        action_dim=16,  # Rainbow has 16 action dimensions
        action_horizon=50,
        max_token_len=250,
    ),
    data=LeRobotRainbowDataConfig(
        repo_id="HumanoidTeam/QuaversAnastacia24040802",
        base_config=DataConfig(
            local_files_only=False,
            prompt_from_task=False,  # Use task field from dataset for prompts
        ),
        default_prompt="Pick up the bag of chips.",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
    num_train_steps=120_000,
    batch_size=128,
),
    # HumanoidTeam/cans_pick_one_amazon
    TrainConfig(
        name="pi0_toilet_pp",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="HumanoidTeam/cans_pick_one_amazon",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="Pick up the toilet paper and afterwards place it in the empty box.",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),

    # HumanoidTeam/stack
    TrainConfig(
        name="pi0_stack",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="HumanoidTeam/stack_2",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="Stack the blocks on top of each other. In this order: blue, purple, red and orange.",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
    ),

    # HumanoidTeam/stack
    TrainConfig(
        name="pi0_clean_plate",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="HumanoidTeam/clean_plate",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="Pick up a plate and clean it by dropping all the things into the box in front of you.",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),

    # https://huggingface.co/datasets/HumanoidTeam/AfterEightDeea23041956
TrainConfig(
    name="pi0_fast_rainbow_poc_aftereight_deea_250",
    model=pi0_fast.Pi0FASTConfig(
        action_dim=16,  # Rainbow has 16 action dimensions
        action_horizon=50,
        max_token_len=250,  # Single-arm robot, so 180 should be sufficient
    ),
    data=LeRobotRainbowDataConfig(
        repo_id="HumanoidTeam/AfterEightDeea23041956",
        base_config=DataConfig(
            local_files_only=False,
            prompt_from_task=False,  # Use task field from dataset for prompts
        ),
        default_prompt="Pick up the box.",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
    num_train_steps=120_000,
),
    # https://huggingface.co/datasets/HumanoidTeam/CrumpetsDeea24041939
TrainConfig(
    name="pi0_fast_rainbow_poc_crumpets_deea",
    model=pi0_fast.Pi0FASTConfig(
        action_dim=16,  # Rainbow has 16 action dimensions
        action_horizon=10,
        max_token_len=180,  # Single-arm robot, so 180 should be sufficient
    ),
    data=LeRobotRainbowDataConfig(
        repo_id="HumanoidTeam/CrumpetsDeea24041939",
        base_config=DataConfig(
            local_files_only=False,
            prompt_from_task=False,  # Use task field from dataset for prompts
        ),
        default_prompt="Pick up the bag of chips.",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
    num_train_steps=60_000,
),

    # https://huggingface.co/datasets/HumanoidTeam/QuaversDeea23041003
    TrainConfig(
        name="pi0_fast_rainbow_poc_quavers",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=16,  # Rainbow has 16 action dimensions
            action_horizon=10,
            max_token_len=180,  # Single-arm robot, so 180 should be sufficient
        ),
        data=LeRobotRainbowDataConfig(
            repo_id="HumanoidTeam/QuaversDeea23041003",
            base_config=DataConfig(
                local_files_only=False,
                prompt_from_task=False,  # Use task field from dataset for prompts
            ),
            default_prompt="Pick up the bag of chips.",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),

    # HumanoidTeam/CrumpetsKeti16041932
    TrainConfig(
        name="pi0_fast_rainbow_poc_crumpets",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=16,  # Rainbow has 16 action dimensions
            action_horizon=10,
            max_token_len=180,  # Single-arm robot, so 180 should be sufficient
        ),
        data=LeRobotRainbowDataConfig(
            repo_id="HumanoidTeam/CrumpetsKeti16041932",
            base_config=DataConfig(
                local_files_only=False,
                prompt_from_task=False,  # Use task field from dataset for prompts
            ),
            default_prompt="Pick up the crumpets.",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),

    # HumanoidTeam/POCPickCrispsFromShelfDiogo
    TrainConfig(
        name="pi0_fast_rainbow_poc",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=16,  # Rainbow has 16 action dimensions
            action_horizon=10,
            max_token_len=180,  # Single-arm robot, so 180 should be sufficient
        ),
        data=LeRobotRainbowDataConfig(
            repo_id="HumanoidTeam/POCPickCrispsFromShelfDiogo",
            base_config=DataConfig(
                local_files_only=False,
                prompt_from_task=False,  # Use task field from dataset for prompts
            ),
            default_prompt="Pick up the bag of chips.",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    # HumanoidTeam/cans_pick_one_amazon
    TrainConfig(
        name="pi0_aloha_can_pp",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="HumanoidTeam/cans_pick_one_amazon",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="Pick up a can and afterwards place it in the empty box.",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    # This config is used to demonstrate how to train on a simple simulated environment.
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    # Add this new config to the _CONFIGS list
    TrainConfig(
        name="pi0_fast_rainbow_poc_aftereight_qs_rotated_8dof_180t_256bz",
        exp_name="exp_rotated_head_8dof",  # New experiment name
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,  # Changed from 16 to 8 for single arm
            action_horizon=50,
            max_token_len=180,  # Reduced from 250 to 180 since we have fewer DOFs
        ),
        data=LeRobotRainbowDataConfigRotated8DOF(  # Use the new 8-DOF config class
            repo_id="HumanoidTeam/after_eight_deea_and_quality_street_arjun",
            base_config=DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "s3://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
        num_train_steps=120_000,
        batch_size=256,  # Keep the same batch size
        num_workers=8,  # Keep the same number of workers
    ),
    # After Eight + Quality Street with 180-degree rotated head camera (scaled learning rate for 256 batch)
    TrainConfig(
        name="pi0_fast_rainbow_poc_aftereight_qs_rotated_250t_256bz_scaled_lr",
        exp_name="exp_rotated_head_scaled_lr",  # New experiment name
        model=pi0_fast.Pi0FASTConfig(
            action_dim=16,
            action_horizon=50,
            max_token_len=250,
        ),
        data=LeRobotRainbowDataConfigRotated(  # Using the rotated config
            repo_id="HumanoidTeam/after_eight_deea_and_quality_street_arjun",
            base_config=DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "s3://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
        # Scaled learning rate for batch size 256 (8x increase from 32)
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,        # Increased warmup steps
            peak_lr=4e-4,            # Scaled up from 5e-5 (5e-5 * 8)
            decay_steps=30000,
            decay_lr=4e-5,           # Scaled up from 5e-6
        ),
        num_train_steps=120_000,
        batch_size=256,              # Target batch size
        num_workers=8,               # Keep same number of workers
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
