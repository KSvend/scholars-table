import pytest


class TestAppImport:
    def test_app_imports(self):
        """Verify app.py can be imported without errors."""
        import app
        assert hasattr(app, "demo")

    def test_app_has_expected_scholars(self):
        import app
        assert len(app.scholar_choices) >= 4
