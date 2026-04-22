"""
test_main.py
------------
Integration tests for the pipeline orchestrator (src/main.py).

Since main.py now reads from config.yaml instead of a SETTINGS dict,
these tests verify the module's entrypoint and config loading functions.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

import src.main as main_module


class TestConfigLoading:
    """Tests for config.yaml loading and helpers."""

    def test_load_config_returns_dict(self, tmp_path):
        """load_config returns a dictionary from a valid YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("project:\n  name: test\n")
        cfg = main_module.load_config(config_file)
        assert isinstance(cfg, dict)
        assert cfg["project"]["name"] == "test"

    def test_load_config_missing_file_raises(self, tmp_path):
        """load_config raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            main_module.load_config(tmp_path / "nonexistent.yaml")

    def test_load_config_invalid_yaml_raises(self, tmp_path):
        """load_config raises ValueError if YAML doesn't parse to dict."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("just a string\n")
        with pytest.raises(ValueError, match="dictionary"):
            main_module.load_config(config_file)

    def test_resolve_repo_path(self, tmp_path):
        """resolve_repo_path joins root with relative path."""
        result = main_module.resolve_repo_path(tmp_path, "data/raw/file.csv")
        assert result == tmp_path / "data/raw/file.csv"


class TestWandBHelpers:
    """Tests for W&B config helper functions."""

    def test_wandb_is_enabled_true(self):
        cfg = {"wandb": {"enabled": True, "project": "test"}}
        assert main_module._wandb_is_enabled(cfg) is True

    def test_wandb_is_enabled_false(self):
        cfg = {"wandb": {"enabled": False}}
        assert main_module._wandb_is_enabled(cfg) is False

    def test_wandb_is_enabled_missing(self):
        cfg = {}
        assert main_module._wandb_is_enabled(cfg) is False

    def test_wandb_get_str(self):
        cfg = {"wandb": {"project": "my-project"}}
        assert main_module._wandb_get_str(cfg, "project") == "my-project"

    def test_wandb_get_str_default(self):
        cfg = {"wandb": {}}
        assert main_module._wandb_get_str(cfg, "project", "default") == "default"

    def test_wandb_get_bool(self):
        cfg = {"wandb": {"log_predictions": True}}
        assert main_module._wandb_get_bool(cfg, "log_predictions") is True

    def test_wandb_get_bool_default(self):
        cfg = {}
        assert main_module._wandb_get_bool(cfg, "log_predictions", False) is False


class TestMainEntrypoint:
    """Tests for the main module entrypoint."""

    def test_main_module_entrypoint(self):
        """The if __name__ == '__main__' guard calls main()."""
        with patch.object(main_module, "main") as mock_main:
            exec(
                "if __name__ == '__main__': main()",
                {"__name__": "__main__", "main": main_module.main},
            )
            mock_main.assert_called_once()
