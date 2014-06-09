#!/usr/bin/env python


def test_unsupervised():
    import experiments.mnist_classification.unsupervised as mdl
    mdl.run(dry_run=True)


def test_associative_memory():
    import experiments.mnist_classification.associative_memory as mdl
    mdl.run(dry_run=True)


def test_supervised_modulation():
    import experiments.mnist_classification.supervised_modulation as mdl
    mdl.run(dry_run=True)


def test_supervised_twoway():
    import experiments.mnist_classification.supervised_twoway as mdl
    mdl.run(dry_run=True)


def test_unsupervised_conv_coates():
    import experiments.coates_unsupervised as mdl
    mdl.run(dry_run=True)


def test_imputation():
    import experiments.mnist_classification.imputation as mdl
    mdl.run(dry_run=True)
