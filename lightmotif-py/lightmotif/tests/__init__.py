from . import test_dna


def load_tests(loader, suite, pattern):
    suite.addTests(loader.loadTestsFromModule(test_dna))
    return suite
