import pytest


class TestAppImport:
    def test_app_imports(self):
        """Verify app.py can be imported without errors."""
        import app
        assert hasattr(app, "demo")

    def test_app_has_expected_scholars(self):
        import app
        assert len(app.scholar_choices) >= 4

    def test_app_has_mode_selector(self):
        import app
        assert hasattr(app, "MODE_CHOICES")
        assert len(app.MODE_CHOICES) == 3
