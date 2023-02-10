import unittest


class TestImports(unittest.TestCase):
    def test_imports(self):
        try:
            import shadow_scholar  # noqa: F401 type: ignore
        except ImportError:
            self.fail(
                "Could not import shadow_scholar; probably dependencies"
                " are not accurately marked as optional."
            )
