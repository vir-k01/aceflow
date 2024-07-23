from dataclasses import dataclass, field
from monty.json import MSONable


@dataclass
class FlowBBasisOrder(MSONable):
    name : str = 'BBasis Order'

@dataclass
class UnaryBBasisOrder(FlowBBasisOrder):
    order : int = 0
    name : str = 'UNARY'
    nradmax_by_orders: list = field(default_factory=lambda: [ 15, 6, 4, 3, 2, 2])
    lmax_by_orders: list = field(default_factory=lambda:[ 0, 3, 3, 2, 2, 1])

@dataclass
class BinaryBBasisOrder(FlowBBasisOrder):
    order : int = 1
    name : str = 'BINARY'
    nradmax_by_orders: list = field(default_factory=lambda: [ 15, 6, 3, 2, 2, 1])
    lmax_by_orders: list = field(default_factory=lambda:[ 0, 3, 2, 1, 1, 0])

@dataclass
class TernaryBBasisOrder(FlowBBasisOrder):
    order : int = 2
    name : str = 'TERNARY'
    nradmax_by_orders: list = field(default_factory=lambda: [ 15, 3, 3, 2, 1])
    lmax_by_orders: list = field(default_factory=lambda:[ 0, 2, 2, 1, 1])

@dataclass
class QuaternaryBBasisOrder(FlowBBasisOrder):
    order : int = 3
    name : str = 'QUATERNARY'
    nradmax_by_orders: list = field(default_factory=lambda: [ 15, 3, 2, 1])
    lmax_by_orders: list = field(default_factory=lambda:[ 0, 2, 2, 1])

@dataclass
class QuinaryBBasisOrder(FlowBBasisOrder):
    order : int = 4
    name : str = 'QUINARY'
    nradmax_by_orders: list = field(default_factory=lambda: [ 15, 3, 2, 1])
    lmax_by_orders: list = field(default_factory=lambda:[ 0, 2, 2, 1])

@dataclass
class AllBBasisOrder(FlowBBasisOrder):
    order : int = -1
    name : str = 'ALL'
    nradmax_by_orders: list = field(default_factory=lambda: [ 15, 3, 2, 1, 1])
    lmax_by_orders: list = field(default_factory=lambda:[ 0, 2, 2, 1, 1])

@dataclass
class BBasisBonds(MSONable):
    name : str = 'bonds'
    radbase : str = 'SBessel'
    radparameters : list = field(default_factory=lambda: [ 5.25 ])
    rcut : float = 5.0
    dcut : float = 0.01
    NameOfCutoffFunction : str = 'cos'


@dataclass
class BBasisEmbedding(MSONable):
    name : str = 'embeddings'
    npot : str = 'FinnisSinclairShiftedScaled'
    fs_parameters : list = field(default_factory=lambda: [ 1, 1, 1, 0.5, 1, 0.75, 1, 0.25, 1, 0.125, 1, 0.375, 1, 0.875, 1, 2])
    ndensity : int = 8