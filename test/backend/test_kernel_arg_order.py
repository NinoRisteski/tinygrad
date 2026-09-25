import unittest, itertools
import numpy as np
from tinygrad import Tensor, UOp, TinyJit, Variable
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import KernelInfo
from test.helpers import assert_jit_cache_len

N = 4
X, Y = np.arange(1, N+1, dtype=np.int32), np.arange(10, 10*(N+1), 10, dtype=np.int32)
# every arg has a distinct role: swapping any two of o/x/y/a/b gives a different answer
def expected(a:int, b:int) -> np.ndarray: return X*a + Y - b

def call_kernel(order:str, **args) -> Tensor:
  # order is a permutation of the arg names: o/x/y (and u, unused) are buffers, a/b are bound Variables
  slot = {k:i for i,k in enumerate(order)}
  def fxn(*ph):
    o, x, y, a, b = [ph[slot[k]] for k in "oxyab"]
    r = UOp.range(N, 0)
    return o[r].store(x[r]*a + y[r] - b).end(r).sink(arg=KernelInfo(name=f"args_{order}"))
  outs = UOp.custom_kernel(*[v.uop if isinstance(v:=args[k], Tensor) else v for k in order], fxn=fxn)
  return Tensor(outs[slot["o"]])

def run(order:str, a:int, b:int) -> np.ndarray:
  return call_kernel(order, o=Tensor.empty(N, dtype=dtypes.int), x=Tensor(X), y=Tensor(Y), u=Tensor.full((N,), 99, dtype=dtypes.int),
                     a=Variable("a", 0, 100, dtypes.int).bind(a), b=Variable("b", 0, 100, dtypes.int).bind(b)).numpy()

class TestKernelArgOrder(unittest.TestCase):
  def test_buffers_any_order(self):
    for order in map("".join, itertools.permutations("oxy")):
      with self.subTest(order=order):
        slot = {k:i for i,k in enumerate(order)}
        def fxn(*ph):
          o, x, y = [ph[slot[k]] for k in "oxy"]
          r = UOp.range(N, 0)
          return o[r].store(x[r]*3 + y[r]).end(r).sink(arg=KernelInfo(name=f"bufs_{order}"))
        args = {"o": Tensor.empty(N, dtype=dtypes.int), "x": Tensor(X), "y": Tensor(Y)}
        outs = UOp.custom_kernel(*[args[k].uop for k in order], fxn=fxn)
        np.testing.assert_equal(Tensor(outs[slot["o"]]).numpy(), X*3 + Y)

  def test_buffers_and_vars_any_order(self):
    for order in map("".join, itertools.permutations("oxyab")):
      with self.subTest(order=order): np.testing.assert_equal(run(order, 3, 5), expected(3, 5))

  def test_unused_buffer(self):
    for order in ("oxyabu", "uaoxby", "xaubyo"):
      with self.subTest(order=order): np.testing.assert_equal(run(order, 3, 5), expected(3, 5))

  def test_rebind_values(self):
    for a, b in ((3, 5), (7, 1), (2, 9)):
      with self.subTest(a=a, b=b): np.testing.assert_equal(run("aoxby", a, b), expected(a, b))

  def test_jit(self):
    @TinyJit
    def f(x:Tensor, y:Tensor, a:UOp, b:UOp) -> Tensor: return call_kernel("xaoby", o=Tensor.empty(N, dtype=dtypes.int), x=x, y=y, a=a, b=b).realize()
    x, y = Tensor(X), Tensor(Y)
    for a, b in ((2, 1), (3, 4), (5, 6), (7, 8)):
      out = f(x, y, Variable("a", 0, 100, dtypes.int).bind(a), Variable("b", 0, 100, dtypes.int).bind(b))
      with self.subTest(a=a, b=b): np.testing.assert_equal(out.numpy(), expected(a, b))
    assert_jit_cache_len(f, 1)

if __name__ == "__main__": unittest.main()
