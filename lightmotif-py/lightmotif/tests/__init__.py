from . import test_pipeline, test_sequence, test_pvalue


def load_tests(loader, suite, pattern):
    suite.addTests(loader.loadTestsFromModule(test_pipeline))
    suite.addTests(loader.loadTestsFromModule(test_sequence))
    suite.addTests(loader.loadTestsFromModule(test_pvalue))
    return suite
