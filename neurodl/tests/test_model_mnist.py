from neurodl import mlp
import pytest
from torch import nn
from torch import optim
import torch 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@pytest.fixture()
def model(request):
    dt = mlp.load_data(20, './data/')
    mdl = mlp.NgnMlp(dt)
    mdl.to(device)

    def fin():
        print("Teardown model")
    request.addfinalizer(fin)

    return mdl


def test_data_split():
    print("Testing data splits")
    batchsize = 20
    num_data = 60000
    split = 0.6
    train_num = int(num_data*split)
    valid_num = int(num_data*(1-split))
    train, valid, full, test = mlp.load_data(batchsize, './data/', split=split)
    assert len(train)*batchsize == train_num
    assert len(valid)*batchsize == valid_num
    assert len(full)*batchsize == num_data
    assert len(train.sampler.indices) == train_num


def test_model_updates(model):
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    criterion = nn.NLLLoss()
    before = list(model.parameters())[0].clone()
    print(len(model.train_mnist.sampler.indices))
    print(len(model.valid_mnist.sampler.indices))

    mlp.train_model(model, optimizer, criterion, epochs=1)
    after = list(model.parameters())[0].clone()
    for b, a in zip(before, after):
        # Make sure something changed.
        assert (b != a).any()


def test_neurogenesis_turnover(model):
    pnew = 50
    replace = 0.6
    added = int(pnew * (1 - replace))
    layer_size = int(model.layer_size)

    before = model.fcs[0].weight.shape[0]
    model.add_new(pnew=50, replace=0.6)
    model.to(device)
    after = model.layer_size
    assert after == (layer_size + added)


def test_neurogenesis_kept_reset(model):
    pnew = 50
    replace = 0.6
    added = int(pnew * (1 - replace))
    removed = int(pnew * replace)

    before = model.fcs[0].weight.clone()
    model.add_new(pnew=pnew, replace=replace)
    model.to(device)
    after = model.fcs[0].weight.clone()

    assert before[:-removed].sum() == after[:-pnew].sum()
