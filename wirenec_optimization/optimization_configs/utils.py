from pathlib import Path
from omegaconf import DictConfig, OmegaConf



def parse_config(config_path: Path, divide: bool = False) -> DictConfig | tuple[DictConfig, ...]:
    configs = OmegaConf.load(config_path)
    if divide:
        possible_configs = (
            "parametrization_hyperparams",
            "scattering_hyperparams",
            "optimization_hyperparams",
            "object_hyperparams",
        )
        return tuple([configs[name] for name in possible_configs if name in configs.keys()])
    else:
        return configs


if __name__ == "__main__":
    pth = Path("single_layer_config.yaml")
    config = parse_config(pth)

    print(config)
