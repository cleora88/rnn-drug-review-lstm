"""Check equations, padding invariance and gradients independently of training."""
import torch
from torch import nn
from models import Cell, Classifier

torch.manual_seed(42)
for kind in ['vanilla','cifg','wmc']:
    cell=Cell(5,7,kind).double()
    x=torch.randn(3,5,dtype=torch.double)
    h=torch.randn(3,7,dtype=torch.double)
    c=torch.randn(3,7,dtype=torch.double)
    gates=(cell.x(x)+cell.h(h)).chunk(3 if kind=='cifg' else 4,-1)
    if kind=='cifg':
        f,o,g=gates; f=torch.sigmoid(f); i=1-f
    else:
        i,f,o,g=gates
        if kind=='wmc': i=i+torch.tanh(cell.ci(c)); f=f+torch.tanh(cell.cf(c))
        i=torch.sigmoid(i); f=torch.sigmoid(f)
    expected_c=f*c+i*torch.tanh(g)
    if kind=='wmc': o=o+torch.tanh(cell.co(expected_c))
    expected_h=torch.sigmoid(o)*torch.tanh(expected_c)
    got_h,got_c=cell(x,(h,c))
    torch.testing.assert_close(got_h,expected_h); torch.testing.assert_close(got_c,expected_c)
    got_h.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cell.parameters())
    model=Classifier(30,kind).eval()
    a=torch.tensor([[2,3,4]])
    torch.testing.assert_close(model(a),model(torch.tensor([[2,3,4,0,0]])))
    assert not any(isinstance(m,nn.LSTM) for m in model.modules())
    print(kind,': equations, gradients and padding passed')
