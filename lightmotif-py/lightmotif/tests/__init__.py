from . import (
    test_doctest, 
    test_pipeline,
    test_scanner, 
    test_sequence, 
    test_pvalue,
)


def load_tests(loader, suite, pattern):
    suite.addTests(loader.loadTestsFromModule(test_doctest))
    suite.addTests(loader.loadTestsFromModule(test_pipeline))
    suite.addTests(loader.loadTestsFromModule(test_pvalue))
    suite.addTests(loader.loadTestsFromModule(test_scanner))
    suite.addTests(loader.loadTestsFromModule(test_sequence))
    return suite
