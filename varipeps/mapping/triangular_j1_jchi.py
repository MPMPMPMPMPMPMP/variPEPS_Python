import collections.abc
from dataclasses import dataclass, field
from functools import partial, reduce
from os import PathLike

import jax.numpy as jnp
import numpy as np
from jax import jit
import jax.util
import h5py
import jax
import logging
import time

from varipeps import varipeps_config
import varipeps.config
from varipeps.peps import PEPS_Tensor, PEPS_Unit_Cell
from varipeps.contractions import apply_contraction, apply_contraction_jitted
from varipeps.expectation.model import Expectation_Model
from varipeps.expectation.three_sites import (
    calc_three_sites_triangle_without_bottom_right_multiple_gates,
    calc_three_sites_triangle_without_top_left_multiple_gates,
    calc_three_sites_triangle_without_bottom_left_multiple_gates,
    calc_three_sites_triangle_without_top_right_multiple_gates,
)
from varipeps.expectation.spiral_helpers import apply_unitary
from varipeps.typing import Tensor
from varipeps.utils.random import PEPS_Random_Number_Generator
from varipeps.mapping import Map_To_PEPS_Model

from varipeps.utils.debug_print import debug_print

from typing import (
    Sequence,
    Union,
    List,
    Callable,
    TypeVar,
    Optional,
    Tuple,
    Type,
    Dict,
    Any,
)



logger = logging.getLogger("varipeps.expectation")

@jax.tree_util.register_dataclass
@dataclass
class Triangular_j1_jchi_model(Expectation_Model):
    r"""
    Expectation model for the triangular-lattice J1-Jchi Hamiltonian.
        --> y
       | +---+
       v |\  | < down triangle
       x | \ |
         +---+
            ^ up triangle

    Args:
    down_gates : Sequence[jnp.ndarray]
        Sequence of gates to be applied on down-pointing triangles
        The gate is applied in the order [top-left, top-right, bottom-right].
    
    up_gates : Sequence[jnp.ndarray]
        Sequence of gates to be applied on up-pointing triangles
        The gate is applied in the order [top-left, bottom-left, bottom-right].
    j1 : float
        Heisenberg exchange coupling.
    jchi : float
        Chiral three-spin coupling. In the provided constructor (from_j1_jchi)
        the sign of the chiral term is opposite for up- and down-pointing triangles.
    normalization_factor : int, optional
        Factor used to normalize the returned expectation values (e.g., when
        multiple physical sites are mapped to a single PEPS site). Default is 1.
    is_spiral_peps : bool, optional
        If True, the model treats tensors as spiral iPEPS and requires a
        spiral_unitary_operator to construct the spiral rotations.
    spiral_unitary_operator : Optional[jnp.ndarray], optional
        Operator used to generate the spiral rotations when is_spiral_peps is True.


    - The returned dtype will be real when all gates are Hermitian and complex
      otherwise.
    """

    up_gates: Sequence[jnp.ndarray]
    down_gates: Sequence[jnp.ndarray]
    j1 : float = field(metadata=dict(static=True))
    jchi : float = field(metadata=dict(static=True))
    _result_type: Type[jnp.number] = field(metadata=dict(static=True))
    normalization_factor: int = field(default=1, metadata=dict(static=True))
    is_spiral_peps: bool = field(default=False, metadata=dict(static=True))
    real_d: int = field(default=2, metadata=dict(static=True))
    spiral_unitary_operator: Optional[jnp.ndarray] = None
    _spiral_D: Optional[jnp.ndarray] = None
    _spiral_sigma: Optional[jnp.ndarray] = None

    @classmethod
    def from_j1_jchi(
        cls,
        j1: float,
        jchi: float,
        p: int = 2,
        normalization_factor: int = 1,
        is_spiral_peps: bool = False,
        spiral_unitary_operator: Optional[jnp.ndarray] = None,
    ):
        Id = jnp.eye(p)
        if p == 2:
            Sx = jnp.array([[0, 1], [1, 0]]) / 2
            Sy = jnp.array([[0, -1j], [1j, 0]]) / 2
            Sz = jnp.array([[1, 0], [0, -1]]) / 2
        else:
            raise NotImplementedError(f"Spin matrices for physical dimension p={p} not defined.")
        
        Svec = jnp.array([Sx, Sy, Sz])

        levicivit3 = jnp.zeros((3, 3, 3))
        levicivit3 = levicivit3.at[0, 1, 2].set(1).at[1, 2, 0].set(1).at[2, 0, 1].set(1)
        levicivit3 = levicivit3.at[0, 2, 1].set(-1).at[2, 1, 0].set(-1).at[1, 0, 2].set(-1)

        def kron(*args):
            return reduce(jnp.kron, args)

        SS_12 = kron(Sx, Sx, Id) + kron(Sy, Sy, Id) + kron(Sz, Sz, Id)
        SS_23 = kron(Id, Sx, Sx) + kron(Id, Sy, Sy) + kron(Id, Sz, Sz)
        SS_31 = kron(Sx, Id, Sx) + kron(Sy, Id, Sy) + kron(Sz, Id, Sz)

        SSS_chi = jnp.einsum('abc,aij,bkl,cmn->ikmjln', levicivit3, Svec, Svec, Svec).reshape(p**3, p**3)

        H_J1 = 1/2 * j1 * (SS_12 + SS_23 + SS_31)

        down_gates= H_J1 + jchi * SSS_chi
        down_gates = (down_gates,) if isinstance(down_gates, jnp.ndarray) else ValueError("down_gates must be a jnp.ndarray.")

        up_gates = H_J1 - jchi * SSS_chi
        up_gates = (up_gates,) if isinstance(up_gates, jnp.ndarray) else ValueError("up_gates must be a jnp.ndarray.")

        # print(list(jnp.allclose(g, g.T.conj()) for g in (trgl_down, trgl_up)))
        # _result_type = (jnp.float64 if all(jnp.allclose(g, g.T.conj()) for g in (trgl_down, trgl_up)) else jnp.complex128)
        # Determine result type
        all_gates = up_gates + down_gates
        _result_type = jnp.float64 if all(jnp.allclose(g, g.T.conj()) for g in all_gates) else jnp.complex128

        if spiral_unitary_operator is not None and not is_spiral_peps:
            raise ValueError(
            "spiral_unitary_operator should only be provided when is_spiral_peps is True."
            )
        if is_spiral_peps and spiral_unitary_operator is None:
            raise ValueError(
            "When is_spiral_peps is True, spiral_unitary_operator must be provided."
            )

        D, sigma = jnp.linalg.eigh(spiral_unitary_operator) if is_spiral_peps else (None, None)

        return cls(
            up_gates=up_gates,
            down_gates=down_gates,
            normalization_factor=normalization_factor,
            is_spiral_peps=is_spiral_peps,
            spiral_unitary_operator=spiral_unitary_operator,
            j1=j1,
            jchi=jchi,
            real_d=p,
            _result_type=_result_type,
            _spiral_D=D,
            _spiral_sigma=sigma,
        )



    @partial(jax.jit, static_argnames=("normalize_by_size", "only_unique", "return_single_gate_results"))
    def __call__(
        self,
        peps_tensors: Sequence[jnp.ndarray],
        unitcell: PEPS_Unit_Cell,
        spiral_vectors: Optional[Union[jnp.ndarray, Sequence[jnp.ndarray]]] = None,
        *,
        normalize_by_size: bool = True,
        only_unique: bool = True,
        return_single_gate_results: bool = False,
    ) -> Union[jnp.ndarray, List[jnp.ndarray]]:
        result = [
            jnp.array(0, dtype=self._result_type)
            for _ in range(len(self.up_gates))
        ]
        # print("compiling")
        if self.is_spiral_peps:
            if spiral_vectors is None:
                raise ValueError(
                    "When using spiral iPEPS, spiral_vectors must be provided."
                )
            # if not isinstance(spiral_vectors, collections.abc.Sequence):
            #     spiral_vectors = (spiral_vectors,) * 3


            #[top-left, top-right, bottom-right]
            working_down_gates = tuple(
                apply_unitary(
                    h,
                    tuple(jnp.array(ri) for ri in ((0,0), (0,1), (1,1))),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    3,
                    (0, 1, 2),
                    varipeps_config.spiral_wavevector_type,
                )
                for h in self.down_gates
            )
            #[top-left, bottom-left, bottom-right]
            working_up_gates = tuple(
                apply_unitary(
                    h,
                    tuple(jnp.array(ri) for ri in ((0,0), (1,0), (1,1))),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    3,
                    (0, 1, 2),
                    varipeps_config.spiral_wavevector_type,
                )
                for h in self.up_gates
            )
        else:
            working_up_gates = self.up_gates
            working_down_gates = self.down_gates

        for x, iter_rows in unitcell.iter_all_rows(only_unique=only_unique):
            for y, view in iter_rows:
                # Get all 4 tensors in the 2x2 view
                tensors_i = view.get_indices(
                    (slice(0, 2, None), slice(0, 2, None))
                )
                tensors = [
                    peps_tensors[i] for j in tensors_i for i in j
                ]
                tensor_objs = [t for tl in view[:2, :2] for t in tl]

                # print(tensor_objs)
                # print(tensors)

                #The gate is applied in the order [top-left, top-right, bottom-right].
                step_result_down = (
                    jax.checkpoint(calc_three_sites_triangle_without_bottom_left_multiple_gates)(
                        tensors,
                        tensor_objs,
                        working_down_gates,
                    )
                )
                #The gate is applied in the order [top-left, bottom-left, bottom-right].
                step_result_up = (
                    jax.checkpoint(calc_three_sites_triangle_without_top_right_multiple_gates)(
                        tensors,
                        tensor_objs,
                        working_up_gates,
                    )
                )

                for sr_i, (sr_down, sr_up) in enumerate(
                    zip(step_result_down, step_result_up, strict=True)
                ):
                    result[sr_i] += (sr_down.real if self._result_type == jnp.float64 else sr_down) + (sr_up.real if self._result_type == jnp.float64 else sr_up)

        if normalize_by_size:
            size = unitcell.get_len_unique_tensors() if only_unique else (unitcell.get_size()[0] * unitcell.get_size()[1])
            size = size * self.normalization_factor
            result = [r / size for r in result]

        if len(result) == 1:
            return result[0]
        else:
            return result



    def save_to_group(self, grp: h5py.Group):
        cls = type(self)
        grp.attrs["class"] = f"{cls.__module__}.{cls.__qualname__}"

        # Save only the parameters needed to reconstruct via from_j1_jchi
        grp.attrs["j1"] = float(self.j1)
        grp.attrs["jchi"] = float(self.jchi)
        grp.attrs["normalization_factor"] = int(self.normalization_factor)
        grp.attrs["is_spiral_peps"] = bool(self.is_spiral_peps)
        grp.attrs["p"] = int(self.real_d)

        if self.is_spiral_peps:
            if self.spiral_unitary_operator is None:
                raise ValueError("spiral_unitary_operator must be provided when is_spiral_peps is True.")
            grp.create_dataset(
                "spiral_unitary_operator",
                data=self.spiral_unitary_operator,
                compression="gzip",
                compression_opts=6,
            )

    @classmethod
    def load_from_group(cls, grp: h5py.Group):
        # if not grp.attrs["class"] == f"{cls.__module__}.{cls.__qualname__}":
        #     raise ValueError(
        #         "The HDF5 group suggests that this is not the right class to load data from it."
        #     )

        # Read parameters
        j1 = float(grp.attrs["j1"])
        jchi = float(grp.attrs["jchi"])
        normalization_factor = int(grp.attrs["normalization_factor"])
        is_spiral_peps = bool(grp.attrs["is_spiral_peps"])

        # Prefer explicitly saved physical dimension 'p'
        if "p" in grp.attrs:
            p = int(grp.attrs["p"])
        else:
            # Backward compatibility: infer p from saved gates if present
            p = 2
            if "gates" in grp and "len_up" in grp["gates"].attrs and grp["gates"].attrs["len_up"] > 0:
                g0 = jnp.asarray(grp["gates"]["up_gate_0"])
                # g0 shape is (p^3, p^3) -> infer p
                inferred = int(round(g0.shape[0] ** (1.0 / 3.0)))
                if inferred ** 3 == g0.shape[0]:
                    p = inferred

        spiral_unitary_operator = (
            jnp.asarray(grp["spiral_unitary_operator"]) if is_spiral_peps else None
        )

        # Reconstruct using the constructor that builds gates
        return cls.from_j1_jchi(
            j1=j1,
            jchi=jchi,
            p=p,
            normalization_factor=normalization_factor,
            is_spiral_peps=is_spiral_peps,
            spiral_unitary_operator=spiral_unitary_operator,
        )