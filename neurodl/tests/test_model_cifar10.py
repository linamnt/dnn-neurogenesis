import neurodl.cnn as cnn
import neurodl
import numpy as np
import pytest
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture()
def model(request):
    mdl = cnn.NgnCnn(layer_size=10)
    mdl.to(device)

    def fin():
        print("Teardown model")

    request.addfinalizer(fin)
    return mdl


@pytest.fixture()
def model_ngn(request, model):
    p_new = 0.5
    replace = 0.6
    model.add_new(p_new=p_new, replace=replace)

    def fin():
        print("Teardown model")

    request.addfinalizer(fin)
    return model


@pytest.fixture()
def dataset(request):
    dt = cnn.Cifar10_data()

    def fin():
        print("Teardown dataset")

    request.addfinalizer(fin)
    return dt


def test_load_data_valid_split():
    split = 0.2
    batch_size = 4
    num_samples = 50000
    num_valid = int(50000 * split)
    print(num_valid)
    num_train = num_samples - int(num_samples * split)
    train, valid, test = cnn.load_data("validation", split=split)
    assert len(train) * batch_size == num_train
    assert len(valid) * batch_size == num_valid


def test_model_updates(model, dataset):
    before = list(model.parameters())[0].clone().detach().cpu().numpy()
    cnn.train_model(model, dataset, epochs=1)
    after = list(model.parameters())[0].clone().detach().cpu().numpy()
    for b, a in zip(before, after):
        # Make sure something changed.
        assert (b != a).any()


def test_neurogenesis_turnover(request, model):
    p_new = 5
    replace = 0.6
    added = int(p_new * (1 - replace))
    layer_size = model.layer_size

    before = model.fcs[1].weight.shape[0]
    model.add_new(p_new=p_new, replace=replace)
    after = model.fcs[1].weight.shape[0]

    assert (
        after - before
    ) == added, "Difference between layer sizes before and after neurogenesis does not equal net addition"
    assert after == (
        layer_size + added
    ), "Final size after neurogenesis not original size + net added"


def test_neurogenesis_kept_replacement(model):
    """
    Test whether
    """
    p_new = 5
    replace = 0.6
    #    removed = int(p_new * replace)

    before = model.fcs[1].weight.clone().cpu().data.numpy()
    idx = model.add_new(p_new=p_new, replace=replace, return_idx=True)
    after = model.fcs[1].weight.clone().cpu().data.numpy()
    before = np.delete(before, idx, axis=0)
    assert (before == after[:-p_new]).all()


def test_neurogenesis_kept_no_replacement(model):
    p_new = 5
    replace = 0

    before = model.fcs[1].weight.clone()
    model.add_new(p_new=p_new, replace=replace)
    after = model.fcs[1].weight.clone()

    assert (before == after[:-p_new]).all()


def test_model_updates_post_neurogenesis(model_ngn, dataset):
    before = list(model_ngn.parameters())[0].clone().detach().cpu().numpy()
    cnn.train_model(model_ngn, dataset, epochs=1)
    after = list(model_ngn.parameters())[0].clone().detach().cpu().numpy()
    for b, a in zip(before, after):
        # Make sure something changed.
        assert (b != a).any()


def test_targeted_threshold():
    dropout_rate = 0.5
    threshold = 0.75
    weights = torch.arange(100).reshape((10, 10))
    weights_out, mask = neurodl.targeted_neurogenesis(
        weights, dropout_rate=dropout_rate, targeted_portion=threshold, is_training=True
    )

    # targeted population must be below the 7th index
    assert torch.all(~mask[7:])


#    cnn.train_model(model, dataset, epochs=1, neurogenesis=5, frequency=None,
#                turnover=0.5)
