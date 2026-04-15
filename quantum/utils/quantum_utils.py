"""
양자 컴퓨팅 공용 유틸리티

Pauli 행렬, 텐서곱, 상태 준비 등 공용 함수.
"""
import numpy as np
from functools import reduce

# === Pauli 행렬 ===
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI_MAP = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}


def tensor(*matrices):
    """행렬들의 텐서곱 (크로네커 곱)"""
    return reduce(np.kron, matrices)


def pauli_string_to_matrix(label):
    """Pauli 문자열 → 행렬.  예: 'XZI' → X⊗Z⊗I"""
    mats = [PAULI_MAP[c] for c in label.upper()]
    return tensor(*mats)


def commutator(A, B):
    """교환자 [A, B] = AB - BA"""
    return A @ B - B @ A


def anticommutator(A, B):
    """{A, B} = AB + BA"""
    return A @ B + B @ A


def partial_trace(rho, dims, keep):
    """부분 대각합.

    Parameters:
        rho: 밀도 행렬
        dims: 각 서브시스템 차원 리스트  예: [2, 2, 2]
        keep: 유지할 서브시스템 인덱스 리스트  예: [0]
    """
    n = len(dims)
    rho_t = rho.reshape(dims * 2)
    trace_axes = [i for i in range(n) if i not in keep]
    for ax in sorted(trace_axes, reverse=True):
        rho_t = np.trace(rho_t, axis1=ax, axis2=ax + n)
        n -= 1
        dims = [d for i, d in enumerate(dims) if i != ax]
    keep_dim = int(np.prod([dims[i] for i in range(len(dims))]))
    return rho_t.reshape(keep_dim, keep_dim)


def fidelity(rho, sigma):
    """상태 충실도 F(ρ, σ)"""
    from scipy.linalg import sqrtm
    sqrt_rho = sqrtm(rho)
    inner = sqrtm(sqrt_rho @ sigma @ sqrt_rho)
    return np.real(np.trace(inner)) ** 2


def entropy(rho):
    """폰 노이만 엔트로피 S(ρ) = -Tr(ρ log ρ)"""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]
    return -np.sum(eigvals * np.log2(eigvals))


def random_state(n_qubits):
    """Haar 랜덤 순수 상태 |ψ⟩"""
    dim = 2 ** n_qubits
    psi = np.random.randn(dim) + 1j * np.random.randn(dim)
    return psi / np.linalg.norm(psi)


def random_unitary(dim):
    """Haar 랜덤 유니터리 행렬"""
    from scipy.stats import unitary_group
    return unitary_group.rvs(dim)


def expectation(operator, state):
    """⟨ψ|O|ψ⟩"""
    return np.real(state.conj() @ operator @ state)


def ry_gate(theta):
    """RY(θ) 게이트"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz_gate(theta):
    """RZ(θ) 게이트"""
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


def cnot():
    """CNOT 게이트 (4×4)"""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)


def hardware_efficient_ansatz(n_qubits, n_layers, params):
    """하드웨어 효율적 앤자츠 유니터리 행렬 구성.

    구조: [RY 레이어 → CNOT 사다리] × n_layers
    파라미터 수: n_qubits × n_layers

    Parameters:
        n_qubits: 큐비트 수
        n_layers: 레이어 깊이
        params: 1D 파라미터 배열 (길이 = n_qubits * n_layers)

    Returns:
        2^n × 2^n 유니터리 행렬
    """
    dim = 2 ** n_qubits
    U = np.eye(dim, dtype=complex)
    idx = 0

    for layer in range(n_layers):
        # RY 레이어
        for q in range(n_qubits):
            gate_list = [I2] * n_qubits
            gate_list[q] = ry_gate(params[idx])
            idx += 1
            U = tensor(*gate_list) @ U

        # CNOT 사다리 (인접 큐비트)
        for q in range(n_qubits - 1):
            # q번째와 q+1번째 큐비트에 CNOT 적용
            if n_qubits == 2:
                cnot_full = cnot()
            else:
                parts = []
                if q > 0:
                    parts.append(np.eye(2 ** q, dtype=complex))
                parts.append(cnot())
                if q + 2 < n_qubits:
                    parts.append(np.eye(2 ** (n_qubits - q - 2), dtype=complex))
                cnot_full = tensor(*parts)
            U = cnot_full @ U

    return U
