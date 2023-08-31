from . import test_pipeline, test_sequence


def load_tests(loader, suite, pattern):
    suite.addTests(loader.loadTestsFromModule(test_pipeline))
    suite.addTests(loader.loadTestsFromModule(test_sequence))
    return suite
