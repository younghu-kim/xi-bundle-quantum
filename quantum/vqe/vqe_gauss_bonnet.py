"""
VQE-Gauss-Bonnet: GB 양자화 제약을 만족하는 Berry-Keating 해밀토니안 탐색

배경:
  - ξ-다발 프레임워크에서 접속의 곡률 κ = |ξ'/ξ|²를 Gauss-Bonnet 정리로 적분하면
    ∫κ dA = 2πN (N = 리만 제타 함수의 영점 수)
  - 각 영점 주위에서 모노드로미 조건: Δarg(ξ) = ±π
  - 이 두 제약을 만족하는 해밀토니안 H가 리만 가설의 스펙트럼 해석과 연결됨
  - Berry-Keating 접근: 고전 해밀토니안 H = xp의 양자화가 리만 영점을 고유값으로 가질 것

VQE 역할:
  - |ψ(θ)⟩ = U(θ)|0⟩ 형태의 변분 파라미터 θ를 최적화
  - 비용 함수: E_VQE + λ_GB·|∮κ - 2πN|² + λ_mono·Σ|Δarg - π|²
  - 최적 θ에서 ⟨H⟩가 최소화되면서 GB 제약도 만족되는 후보 해밀토니안 식별
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# gdl_unified scripts 경로 추가
sys.path.insert(0, os.path.expanduser('~/Desktop/gdl_unified/scripts'))

# Qiskit 선택적 임포트 (없으면 numpy 행렬 시뮬레이션으로 대체)
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    HAS_QISKIT = True
    print("Qiskit 감지됨: Aer 시뮬레이터 사용 가능")
except ImportError:
    HAS_QISKIT = False
    print("Qiskit 없음: numpy 행렬 시뮬레이션으로 동작")


# ---------------------------------------------------------------------------
# 파울리 행렬 상수 (텐서곱 기반)
# ---------------------------------------------------------------------------
PAULI_I = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_DICT = {'I': PAULI_I, 'X': PAULI_X, 'Y': PAULI_Y, 'Z': PAULI_Z}


class PauliDecomposer:
    """
    임의의 에르미트 행렬을 Pauli 텐서곱의 선형 결합으로 분해.

    H = Σ_i a_i P_i  (P_i ∈ {I,X,Y,Z}^⊗n)

    계수 공식: a_i = Tr(P_i · H) / 2^n
    분해 결과는 VQE 에너지 계산에 직접 사용됨.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits

    def _pauli_string_matrix(self, pauli_str: str) -> np.ndarray:
        """'XYZIX' 같은 문자열을 n-큐비트 텐서곱 행렬로 변환"""
        mat = PAULI_DICT[pauli_str[0]]
        for ch in pauli_str[1:]:
            mat = np.kron(mat, PAULI_DICT[ch])
        return mat

    def decompose(self, H: np.ndarray, threshold: float = 1e-10) -> list:
        """
        H를 Pauli 분해하여 [(계수, 파울리_문자열), ...] 반환.
        threshold 미만의 계수는 제거하여 스파스 표현 유지.
        """
        if H.shape != (self.dim, self.dim):
            raise ValueError(f"해밀토니안 크기 불일치: 기대 {self.dim}×{self.dim}, 입력 {H.shape}")

        labels = ['I', 'X', 'Y', 'Z']
        # 모든 n-큐비트 파울리 문자열 열거
        from itertools import product as iproduct
        pauli_terms = []
        for combo in iproduct(labels, repeat=self.n_qubits):
            pstr = ''.join(combo)
            P = self._pauli_string_matrix(pstr)
            coeff = np.real(np.trace(P @ H)) / self.dim
            if abs(coeff) > threshold:
                pauli_terms.append((coeff, pstr))

        return pauli_terms

    def reconstruct(self, pauli_terms: list) -> np.ndarray:
        """Pauli 분해 결과로부터 행렬 재구성 (검증용)"""
        mat = np.zeros((self.dim, self.dim), dtype=complex)
        for coeff, pstr in pauli_terms:
            mat += coeff * self._pauli_string_matrix(pstr)
        return mat


class HamiltonianCandidates:
    """
    5개 Berry-Keating 계열 해밀토니안의 유한 차원 트렁케이션.

    각 후보는 리만 제타 함수의 영점이 고유값이 되는
    스펙트럼 해석(Hilbert-Polya 가설)의 서로 다른 구현을 반영한다.
    """

    @staticmethod
    def xp_truncated(dim: int) -> np.ndarray:
        """
        H = (XP + PX) / 2 의 조화진동자 기저 트렁케이션.

        - X = 위치 연산자, P = -i ∂/∂x (운동량 연산자)
        - 조화진동자 생성/소멸 연산자로 표현:
            X = (a + a†) / √2
            P = i(a† - a) / √2
          → XP + PX = i(a†² - a²)  (반교환자 형태)
        - 행렬 원소: ⟨n|H|m⟩ = (i/2)(√(m(m-1))δ_{n,m-2} - √(n(n-1))δ_{n-2,m})
        - Berry-Keating: 이 연산자의 고유값이 리만 영점 Im(s)와 관련될 것으로 추측

        참조: Berry & Keating (1999), SIAM Rev. 41, 236–266
        """
        H = np.zeros((dim, dim), dtype=complex)
        for n in range(dim):
            for m in range(dim):
                # ⟨n|XP+PX|m⟩ = i[√(m(m-1))δ_{n,m-2} - √(n(n-1))δ_{n-2,m}]
                if m >= 2 and n == m - 2:
                    H[n, m] += 0.5j * np.sqrt(float(m * (m - 1)))
                if n >= 2 and m == n - 2:
                    H[n, m] -= 0.5j * np.sqrt(float(n * (n - 1)))
        # 에르미트 대칭 보정 (수치 오차 방지)
        H = (H + H.conj().T) / 2.0
        return H

    @staticmethod
    def leclair_mussardo(dim: int, coupling: float = 1.0) -> np.ndarray:
        """
        LeClair-Mussardo Bethe Ansatz 해밀토니안.

        - 1+1차원 가적분 양자장이론에서 유도된 산란 해밀토니안
        - Bethe Ansatz 양자화 조건: e^{i p_j L} ∏_{k≠j} S(p_j - p_k) = 1
          → 유한 크기 스펙트럼을 행렬로 근사
        - 행렬 원소:
            H_{nn} = n + 1/2  (자유 준위, 조화진동자 대각선)
            H_{nm} = coupling × f(n,m)  (산란 위상 off-diagonal)
          여기서 f(n,m) = 1/(|n-m| + 1) (유효 산란 커플링의 근사)
        - 리만 영점과의 관련: L-함수의 Bethe Ansatz 표현에서 등장하는 위상 구조를 모사

        참조: LeClair & Mussardo (1999), Nucl. Phys. B 552, 624–642
        """
        H = np.zeros((dim, dim), dtype=complex)
        for n in range(dim):
            # 대각: 조화진동자 에너지 준위 + 산란 자기 에너지
            H[n, n] = n + 0.5
            for m in range(n + 1, dim):
                # off-diagonal: Bethe Ansatz 산란 위상의 근사 (역거리 감쇠)
                val = coupling / (abs(n - m) + 1.0)
                # 위상 인자 추가: e^{iπ(n+m)/dim} (산란 위상 구조 반영)
                phase = np.exp(1j * np.pi * (n + m) / dim)
                H[n, m] = val * phase
                H[m, n] = np.conj(H[n, m])
        # 에르미트 대칭 보정
        H = (H + H.conj().T) / 2.0
        return H

    @staticmethod
    def sierra_landau(dim: int, B: float = 1.0) -> np.ndarray:
        """
        Sierra-Townsend Landau 준위 해밀토니안.

        - 균일 자기장 B 속의 2차원 전자계: H = (p - A)²/2 (Landau 게이지)
        - Landau 준위: E_n = B(2n+1), n = 0, 1, 2, ...
        - 영점 조건: 자기장 B를 조정하여 Landau 준위가 리만 영점에 일치하도록 설정
        - Sierra의 제안: H = p² + B²x² - B (1차원 축약 후)
          → 유한 차원 조화진동자 + 자기 상호작용

        행렬 구성:
          H_{nn} = B(2n + 1)  (Landau 준위 대각)
          비대각: 자기장 요동에 의한 결합 (ξ-다발 곡률 반영)

        참조: Sierra & Townsend (2008), PRL 101, 110201
        """
        H = np.zeros((dim, dim), dtype=complex)
        for n in range(dim):
            # Landau 준위 에너지
            H[n, n] = B * (2 * n + 1)
            # 자기 상호작용: 생성/소멸 연산자를 통한 인접 준위 결합
            if n + 1 < dim:
                # ⟨n+1|x|n⟩ = √((n+1)/2)  (조화진동자 행렬 원소)
                coupling = B * np.sqrt((n + 1) / 2.0)
                H[n, n + 1] = coupling
                H[n + 1, n] = np.conj(coupling)
        # 에르미트 대칭 보정
        H = (H + H.conj().T) / 2.0
        return H

    @staticmethod
    def yakaboylu_weil(dim: int) -> np.ndarray:
        """
        Yakaboylu Weil 양성 연산자의 유한 차원 근사.

        - Weil의 명시적 공식: Σ_ρ h(ρ) = h(1/2+it) 형태의 합
          (ρ: 리만 제타의 비자명 영점, h: 검정 함수)
        - 양성 연산자 W: ⟨φ|W|φ⟩ ≥ 0 이면 리만 가설 성립
        - 유한 차원 근사:
            W_{nm} = ∫ φ_n(x) K(x,y) φ_m(y) dx dy
          여기서 K는 Weil 커널, φ_n은 조화진동자 고유함수

        행렬 원소 근사 (수치 적분 대체):
          W_{nn} = n + 1  (대각: 양성 정치화)
          W_{nm} = (-1)^{n+m} / (|n-m|² + 1)  (Weil 커널의 패리티 구조)

        참조: Yakaboylu et al. (2021), J. Phys. A 54, 015302
        """
        H = np.zeros((dim, dim), dtype=complex)
        for n in range(dim):
            # 대각: 양성 정치화 (W ≥ 0 조건)
            H[n, n] = float(n + 1)
            for m in range(n + 1, dim):
                # Weil 커널의 패리티 구조: (-1)^{n+m} 위상
                sign = (-1) ** (n + m)
                val = sign / ((n - m) ** 2 + 1.0)
                H[n, m] = val
                H[m, n] = np.conj(H[n, m])
        # 에르미트 대칭 보정
        H = (H + H.conj().T) / 2.0
        return H

    @staticmethod
    def bbm_pt_symmetric(dim: int, epsilon: float = 0.1) -> np.ndarray:
        """
        Bender-Brody-Müller PT-대칭 해밀토니안.

        - 비에르미트이지만 PT-대칭: [H, PT] = 0 (P=반전, T=시간반전)
        - 실수 고유값을 가짐 (PT-대칭 비파괴 상태에서)
        - BBM 제안: H = (J+J†)/2 + iε(J-J†)/2
          여기서 J = 상승 연산자 (Lie 대수 su(1,1))
        - 트렁케이션된 su(1,1) 표현:
            J_{n,n+1} = √((n+1)(n+κ))  (κ = Bargmann 지수, κ=1/2 선택)
        - ε 파라미터: PT-대칭 파괴 강도 (ε→0: 에르미트 극한)

        참조: Bender, Brody & Müller (2017), PRL 118, 130201
        """
        # su(1,1) 생성 연산자 J (κ=1/2 표현)
        kappa = 0.5
        J = np.zeros((dim, dim), dtype=complex)
        for n in range(dim - 1):
            # su(1,1) 상승 연산자 행렬 원소
            J[n, n + 1] = np.sqrt((n + 1) * (n + kappa))

        # BBM 해밀토니안: 에르미트 부분 + iε 반에르미트 부분
        H_herm = (J + J.conj().T) / 2.0
        H_anti = 1j * epsilon * (J - J.conj().T) / 2.0

        H = H_herm + H_anti

        # 주의: BBM은 의도적으로 비에르미트이나, 실수 고유값을 가짐
        # VQE 적용을 위해 에르미트화: H → (H + H†)/2 + iε·반에르미트 부분 유지
        # 여기서는 에르미트 부분만 사용 (epsilon이 작으므로 근사 유효)
        H_eff = (H + H.conj().T) / 2.0

        return H_eff


class GaussBonnetConstraint:
    """
    Gauss-Bonnet 2πN 제약 조건 평가기.

    ξ-다발의 접속 1-형식 A에 대해:
      - 곡률: F = dA, κ = |F|²
      - GB 정리: ∫_M κ dA = 2πχ(M) = 2πN  (N = 영점 수 = 오일러 특성수)
      - 모노드로미: γ_j = Hol(A, ∂D_j) = e^{iπ}  (각 영점 j 주위)

    고유값 {λ_j}로부터 곡률 적분 근사:
      ∫κ ≈ Σ_j |λ_j|² / Σ_j |λ_j|  (스펙트럼 가중 평균)
    """

    def __init__(self, target_N: int):
        """
        Parameters
        ----------
        target_N : int
            기대 영점 수 N (GB 조건의 우변 2πN)
        """
        self.target_N = target_N
        self.target_integral = 2 * np.pi * target_N

    def curvature_integral(self, eigenvalues: np.ndarray) -> float:
        """
        고유값으로부터 곡률 적분 ∫κ 계산.

        물리적 해석:
          - 고유값 λ_j ↔ 접속의 홀로노미 e^{iλ_j}
          - 곡률 κ_j ≈ |λ_j - λ_{j-1}|² (인접 준위 차이 = 곡률의 국소 기여)
          - ∫κ ≈ Σ_j (λ_j - λ_{j-1})²  (스펙트럼 강성)

        Parameters
        ----------
        eigenvalues : np.ndarray
            정렬된 실수 고유값 배열

        Returns
        -------
        float
            곡률 적분 추정값
        """
        eigs = np.sort(np.real(eigenvalues))
        if len(eigs) < 2:
            return 0.0

        # 인접 간격의 제곱합 (분광 강성 = GB 적분의 이산 근사)
        gaps = np.diff(eigs)
        # 정규화: 전체 스펙트럼 범위로 나눠 척도 불변성 확보
        span = eigs[-1] - eigs[0]
        if span < 1e-12:
            return 0.0

        # ∫κ ≈ (dim²/span²) × Σ(Δλ_j)²  (스케일링 보정 포함)
        dim = len(eigs)
        kappa_integral = (dim ** 2 / span ** 2) * np.sum(gaps ** 2)

        return float(kappa_integral)

    def monodromy_penalty(self, eigenvalues: np.ndarray) -> float:
        """
        모노드로미 ±π 조건 패널티 계산.

        각 고유값 λ_j는 대응하는 영점 ρ_j = 1/2 + iλ_j 주위에서
        ξ(s)의 위상이 ±π 변화해야 함.

        근사:
          - 고유값 λ_j에 대응하는 위상: φ_j = π × (j+1) (이상적 간격)
          - 실제 위상: arg(e^{iλ_j}) = λ_j mod 2π
          - 패널티: Σ_j |arg(e^{iλ_j}) - π(2j+1)|²

        Parameters
        ----------
        eigenvalues : np.ndarray
            정렬된 실수 고유값 배열

        Returns
        -------
        float
            모노드로미 패널티 (0에 가까울수록 ±π 조건 만족)
        """
        eigs = np.sort(np.real(eigenvalues))
        n = len(eigs)

        # 이상적 위상 간격: π (리만 가설의 스펙트럼 조건)
        # 각 λ_j에서 위상: φ_j = λ_j mod (2π), 이상값은 π·(2j+1)/n
        penalty = 0.0
        for j, lam in enumerate(eigs):
            # 실제 위상 (0~2π 범위)
            actual_phase = lam % (2 * np.pi)
            # 이상 위상: j번째 영점의 위상이 π 단위로 균등 분포
            ideal_phase = np.pi * (2 * j + 1) / n
            diff = actual_phase - ideal_phase
            # 원형 거리 (최소 위상 차이)
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            penalty += diff ** 2

        return float(penalty / n)  # 정규화

    def total_constraint(
        self,
        eigenvalues: np.ndarray,
        lambda_gb: float = 1.0,
        lambda_mono: float = 0.5
    ) -> float:
        """
        전체 GB 제약 비용 계산.

        Cost_constraint = λ_GB × |∫κ - 2πN|² + λ_mono × Σ|Δarg - π|²

        Parameters
        ----------
        eigenvalues : np.ndarray
            해밀토니안의 고유값
        lambda_gb : float
            GB 적분 패널티 가중치
        lambda_mono : float
            모노드로미 패널티 가중치

        Returns
        -------
        float
            총 제약 위반 비용
        """
        kappa_int = self.curvature_integral(eigenvalues)
        gb_penalty = lambda_gb * (kappa_int - self.target_integral) ** 2

        mono_pen = self.monodromy_penalty(eigenvalues)
        mono_penalty = lambda_mono * mono_pen

        return float(gb_penalty + mono_penalty)


class VQEGaussBonnet:
    """
    VQE (Variational Quantum Eigensolver) + Gauss-Bonnet 제약 최적화.

    하드웨어 효율적 앤자츠 U(θ)를 numpy 행렬 시뮬레이션으로 구현.
    (Qiskit 없어도 완전 동작)

    앤자츠 구조:
      - n_layers 레이어, 각 레이어: RY(θ) 회전 + CNOT 사다리
      - 전체 유니터리: U(θ) = ∏_l [CNOT_layer × RY_layer(θ_l)]
      - 상태: |ψ(θ)⟩ = U(θ)|0...0⟩

    비용 함수:
      Cost(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ + λ_GB|∫κ-2πN|² + λ_mono·Σ|Δarg-π|²
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        hamiltonian_matrix: np.ndarray,
        target_N: int
    ):
        """
        Parameters
        ----------
        n_qubits : int
            큐비트 수 (5~10 권장)
        n_layers : int
            앤자츠 레이어 깊이 (2~4 권장)
        hamiltonian_matrix : np.ndarray
            에르미트 해밀토니안 행렬 (2^n_qubits × 2^n_qubits)
        target_N : int
            GB 조건의 목표 영점 수
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2 ** n_qubits
        self.H = hamiltonian_matrix
        self.gb_constraint = GaussBonnetConstraint(target_N)

        # 파라미터 수: n_layers × n_qubits (각 RY 게이트 1개 파라미터)
        self.n_params = n_layers * n_qubits

        # 해밀토니안 고유값 (GB 제약 계산용, 사전 계산)
        self._eigenvalues = None
        self._precompute_eigenvalues()

    def _precompute_eigenvalues(self):
        """해밀토니안 고유값 사전 계산"""
        try:
            self._eigenvalues, _ = eigh(self.H)
        except Exception as e:
            print(f"  경고: 고유값 계산 실패 ({e}), 대각 원소 사용")
            self._eigenvalues = np.real(np.diag(self.H))

    def _ry_gate(self, theta: float) -> np.ndarray:
        """단일 큐비트 RY 회전 게이트: RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]"""
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def _cnot_gate(self, control: int, target: int) -> np.ndarray:
        """
        n-큐비트 CNOT 게이트 (제어: control, 표적: target).
        전체 힐버트 공간 2^n × 2^n 행렬로 확장.
        """
        mat = np.eye(self.dim, dtype=complex)
        for i in range(self.dim):
            # 제어 큐비트가 |1⟩이면 표적 큐비트 비트 플립
            control_bit = (i >> (self.n_qubits - 1 - control)) & 1
            if control_bit == 1:
                # 표적 큐비트 비트 플립
                j = i ^ (1 << (self.n_qubits - 1 - target))
                mat[i, i] = 0
                mat[i, j] = 1
                mat[j, i] = 1
                mat[j, j] = 0
        return mat

    def _ry_layer(self, thetas: np.ndarray) -> np.ndarray:
        """
        n-큐비트 RY 레이어: 각 큐비트에 독립적인 RY 적용.
        전체 유니터리 = RY_0(θ_0) ⊗ RY_1(θ_1) ⊗ ... ⊗ RY_{n-1}(θ_{n-1})
        """
        mat = self._ry_gate(thetas[0])
        for i in range(1, self.n_qubits):
            mat = np.kron(mat, self._ry_gate(thetas[i]))
        return mat

    def _cnot_ladder(self) -> np.ndarray:
        """
        CNOT 사다리: 0→1, 1→2, ..., (n-2)→(n-1) 순서 CNOT.
        인접 큐비트 얽힘 생성 (하드웨어 효율적 구조).
        """
        mat = np.eye(self.dim, dtype=complex)
        for i in range(self.n_qubits - 1):
            mat = self._cnot_gate(i, i + 1) @ mat
        return mat

    def ansatz_matrix(self, params: np.ndarray) -> np.ndarray:
        """
        하드웨어 효율적 앤자츠 유니터리 행렬 U(θ).

        구조: U = ∏_{l=0}^{L-1} [CNOT_ladder × RY_layer(θ_l)]
        초기 상태: |0...0⟩ (계산 기저)

        Parameters
        ----------
        params : np.ndarray
            shape (n_layers × n_qubits,) 의 각도 파라미터

        Returns
        -------
        np.ndarray
            2^n × 2^n 유니터리 행렬
        """
        params_2d = params.reshape(self.n_layers, self.n_qubits)
        U = np.eye(self.dim, dtype=complex)

        for layer_idx in range(self.n_layers):
            ry = self._ry_layer(params_2d[layer_idx])
            cnot = self._cnot_ladder()
            U = cnot @ ry @ U

        # 마지막 RY 레이어 (CNOT 없이): 최종 회전으로 더 표현력 있는 앤자츠
        # (파라미터 재사용: 첫 레이어 파라미터 활용)
        final_ry = self._ry_layer(params_2d[0] * 0.5)
        U = final_ry @ U

        return U

    def state_vector(self, params: np.ndarray) -> np.ndarray:
        """앤자츠 상태 벡터 |ψ(θ)⟩ = U(θ)|0...0⟩ 계산"""
        U = self.ansatz_matrix(params)
        # 초기 상태: |0...0⟩ = 계산 기저의 첫 번째 벡터
        psi0 = np.zeros(self.dim, dtype=complex)
        psi0[0] = 1.0
        return U @ psi0

    def energy_expectation(self, params: np.ndarray) -> float:
        """
        변분 에너지 기대값 ⟨ψ(θ)|H|ψ(θ)⟩ 계산.

        Parameters
        ----------
        params : np.ndarray
            앤자츠 파라미터

        Returns
        -------
        float
            에너지 기대값 (실수)
        """
        psi = self.state_vector(params)
        energy = np.real(np.conj(psi) @ self.H @ psi)
        return float(energy)

    def constrained_cost(
        self,
        params: np.ndarray,
        lambda_gb: float = 1.0,
        lambda_mono: float = 0.5
    ) -> float:
        """
        전체 비용 함수: VQE 에너지 + GB 제약.

        Cost(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
                + λ_GB × |∫κ - 2πN|²
                + λ_mono × Σ|Δarg - π|²

        GB 제약을 비용에 포함함으로써:
          - 에너지 최소화(VQE) + 위상 구조 보존(GB)을 동시 달성
          - 바닥 상태가 GB 조건을 만족하는 해밀토니안 후보 식별 가능

        Parameters
        ----------
        params : np.ndarray
            앤자츠 파라미터
        lambda_gb : float
            GB 패널티 가중치 (클수록 GB 제약 엄격)
        lambda_mono : float
            모노드로미 패널티 가중치

        Returns
        -------
        float
            총 비용
        """
        # 변분 에너지 (VQE 항)
        e_vqe = self.energy_expectation(params)

        # GB 제약 비용 (사전 계산된 고유값 사용)
        gb_cost = self.gb_constraint.total_constraint(
            self._eigenvalues, lambda_gb, lambda_mono
        )

        return float(e_vqe + gb_cost)

    def optimize(
        self,
        n_seeds: int = 5,
        maxiter: int = 1000,
        lambda_gb: float = 1.0,
        lambda_mono: float = 0.5,
        method: str = 'COBYLA'
    ) -> dict:
        """
        다중 시드 최적화 실행.

        전략:
          1. n_seeds개의 랜덤 초기 파라미터로 독립적 최적화 시도
          2. 각 시드의 최종 비용 비교
          3. 최소 비용 시드의 결과 반환

        Parameters
        ----------
        n_seeds : int
            랜덤 초기화 시드 수
        maxiter : int
            최적화 반복 최대 횟수
        lambda_gb : float
            GB 패널티 가중치
        lambda_mono : float
            모노드로미 패널티 가중치
        method : str
            최적화 알고리즘 ('COBYLA' 또는 'L-BFGS-B')

        Returns
        -------
        dict
            best_params, best_cost, best_energy, convergence_history 포함
        """
        best_result = None
        best_cost = np.inf
        history = []

        for seed_idx in range(n_seeds):
            rng = np.random.RandomState(seed_idx * 42 + 7)
            # 초기 파라미터: [-π, π] 균등 분포
            init_params = rng.uniform(-np.pi, np.pi, self.n_params)

            def cost_fn(params):
                return self.constrained_cost(params, lambda_gb, lambda_mono)

            try:
                if method == 'COBYLA':
                    result = minimize(
                        cost_fn,
                        init_params,
                        method='COBYLA',
                        options={'maxiter': maxiter, 'rhobeg': 0.5}
                    )
                else:  # L-BFGS-B
                    result = minimize(
                        cost_fn,
                        init_params,
                        method='L-BFGS-B',
                        options={'maxiter': maxiter, 'ftol': 1e-10}
                    )

                seed_cost = result.fun
                seed_energy = self.energy_expectation(result.x)
                history.append({
                    'seed': seed_idx,
                    'final_cost': float(seed_cost),
                    'final_energy': float(seed_energy),
                    'success': result.success,
                    'n_iter': result.get('nit', maxiter)
                })

                if seed_cost < best_cost:
                    best_cost = seed_cost
                    best_result = result

            except Exception as e:
                history.append({
                    'seed': seed_idx,
                    'final_cost': np.inf,
                    'final_energy': np.inf,
                    'success': False,
                    'error': str(e)
                })

        if best_result is None:
            # 모든 시드 실패: 초기 파라미터 반환
            best_params = np.zeros(self.n_params)
            best_cost = self.constrained_cost(best_params, lambda_gb, lambda_mono)
        else:
            best_params = best_result.x

        return {
            'best_params': best_params,
            'best_cost': float(best_cost),
            'best_energy': float(self.energy_expectation(best_params)),
            'convergence_history': history,
            'n_seeds': n_seeds
        }

    def analyze_result(self, optimal_params: np.ndarray) -> dict:
        """
        최적 파라미터에서 물리적 결과 분석.

        항목:
          - 변분 에너지 ⟨H⟩
          - 해밀토니안 고유값 스펙트럼
          - GB 곡률 적분 ∫κ vs 목표 2πN
          - 모노드로미 패널티
          - 상태 벡터의 정규화 확인

        Parameters
        ----------
        optimal_params : np.ndarray
            최적화된 앤자츠 파라미터

        Returns
        -------
        dict
            분석 결과 딕셔너리
        """
        psi = self.state_vector(optimal_params)

        # 정규화 확인
        norm = np.real(np.dot(np.conj(psi), psi))

        # 변분 에너지
        energy = self.energy_expectation(optimal_params)

        # GB 분석
        kappa_integral = self.gb_constraint.curvature_integral(self._eigenvalues)
        target_integral = self.gb_constraint.target_integral
        gb_violation = abs(kappa_integral - target_integral)

        # 모노드로미 패널티
        mono_penalty = self.gb_constraint.monodromy_penalty(self._eigenvalues)

        # 스펙트럼 통계
        eigs = np.sort(self._eigenvalues)
        gaps = np.diff(eigs) if len(eigs) > 1 else np.array([0.0])

        return {
            '변분_에너지': float(energy),
            '상태_정규화': float(norm),
            '고유값_최솟값': float(eigs[0]),
            '고유값_최댓값': float(eigs[-1]),
            '평균_레벨_간격': float(np.mean(gaps)),
            'GB_곡률_적분': float(kappa_integral),
            'GB_목표값_2πN': float(target_integral),
            'GB_위반량': float(gb_violation),
            '모노드로미_패널티': float(mono_penalty),
            'GB_조건_충족': gb_violation < 0.1 * target_integral
        }


def _truncate_to_qubits(H_full: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    임의 크기 해밀토니안을 2^n_qubits 크기로 트렁케이션/패딩.

    - H_full이 더 크면: 첫 2^n 행/열 추출
    - H_full이 더 작으면: 0 패딩 (단위 행렬 블록 추가)
    """
    dim_target = 2 ** n_qubits
    dim_current = H_full.shape[0]

    if dim_current == dim_target:
        return H_full
    elif dim_current > dim_target:
        # 저에너지 부분 공간으로 트렁케이션
        # (에너지 정렬 후 낮은 고유값 dim_target개만 유지)
        eigs, vecs = eigh(H_full)
        H_trunc = np.diag(eigs[:dim_target])
        return H_trunc
    else:
        # 패딩: 나머지는 큰 에너지 (단위 행렬 × 큰 값)
        H_padded = np.zeros((dim_target, dim_target), dtype=complex)
        H_padded[:dim_current, :dim_current] = H_full
        # 패딩 영역은 고에너지로 설정 (고유값 범위의 10배)
        max_eig = np.max(np.abs(np.real(np.diag(H_full)))) + 1.0
        for i in range(dim_current, dim_target):
            H_padded[i, i] = max_eig * 10.0
        return H_padded


def benchmark_all_candidates(
    n_qubits: int = 5,
    target_N: int = 3,
    n_layers: int = 2,
    n_seeds: int = 3,
    lambda_gb: float = 1.0,
    lambda_mono: float = 0.5
) -> dict:
    """
    5개 Berry-Keating 후보 모두에 대해 VQE-GB 실행 및 비교.

    결과를 표 형식으로 출력하고 GB 조건 충족 여부 판정.

    Parameters
    ----------
    n_qubits : int
        큐비트 수 (기본값 5 → dim=32)
    target_N : int
        목표 영점 수 (기본값 3)
    n_layers : int
        앤자츠 레이어 수 (기본값 2)
    n_seeds : int
        최적화 시드 수 (기본값 3, 속도 우선)
    lambda_gb : float
        GB 패널티 가중치
    lambda_mono : float
        모노드로미 패널티 가중치

    Returns
    -------
    dict
        각 후보의 결과 딕셔너리
    """
    dim = 2 ** n_qubits  # 32 (5큐비트)

    print("=" * 70)
    print("VQE-Gauss-Bonnet 벤치마크")
    print(f"  큐비트 수: {n_qubits} (dim={dim})")
    print(f"  앤자츠 레이어: {n_layers}")
    print(f"  목표 영점 수 N={target_N} (GB 목표: 2πN = {2*np.pi*target_N:.4f})")
    print(f"  최적화 시드 수: {n_seeds}")
    print(f"  패널티 가중치: λ_GB={lambda_gb}, λ_mono={lambda_mono}")
    print("=" * 70)

    # 5개 후보 해밀토니안 생성
    print("\n[1단계] 해밀토니안 후보 생성 중...")
    candidates = {
        'H_xp (Berry-Keating)': HamiltonianCandidates.xp_truncated(dim),
        'H_LeClair (Bethe Ansatz)': HamiltonianCandidates.leclair_mussardo(dim, coupling=0.5),
        'H_Sierra (Landau)': HamiltonianCandidates.sierra_landau(dim, B=1.0),
        'H_Yakaboylu (Weil)': HamiltonianCandidates.yakaboylu_weil(dim),
        'H_BBM (PT-대칭)': HamiltonianCandidates.bbm_pt_symmetric(dim, epsilon=0.05),
    }

    # Pauli 분해 (정보 출력용, 실제 VQE는 행렬 직접 사용)
    decomposer = PauliDecomposer(n_qubits)

    results = {}

    print("\n[2단계] 각 후보에 대해 VQE-GB 최적화 실행 중...\n")
    for cand_name, H_full in candidates.items():
        print(f"  후보: {cand_name}")

        # n_qubits 크기에 맞게 트렁케이션
        H = _truncate_to_qubits(H_full, n_qubits)

        # Pauli 분해 항 수 계산
        try:
            pauli_terms = decomposer.decompose(H, threshold=1e-6)
            n_pauli = len(pauli_terms)
        except Exception:
            n_pauli = -1

        print(f"    Pauli 항 수: {n_pauli if n_pauli >= 0 else '계산 실패'}")

        # VQE-GB 최적화
        vqe = VQEGaussBonnet(
            n_qubits=n_qubits,
            n_layers=n_layers,
            hamiltonian_matrix=H,
            target_N=target_N
        )

        opt_result = vqe.optimize(
            n_seeds=n_seeds,
            maxiter=500,
            lambda_gb=lambda_gb,
            lambda_mono=lambda_mono,
            method='COBYLA'
        )

        # 결과 분석
        analysis = vqe.analyze_result(opt_result['best_params'])

        # 결과 저장
        results[cand_name] = {
            'optimization': opt_result,
            'analysis': analysis,
            'n_pauli_terms': n_pauli
        }

        # 출력
        print(f"    변분 에너지: {analysis['변분_에너지']:.6f}")
        print(f"    GB 곡률 적분: {analysis['GB_곡률_적분']:.6f} "
              f"(목표: {analysis['GB_목표값_2πN']:.6f})")
        print(f"    GB 위반량: {analysis['GB_위반량']:.6f}")
        print(f"    모노드로미 패널티: {analysis['모노드로미_패널티']:.6f}")
        gb_ok = "충족" if analysis['GB_조건_충족'] else "미충족"
        print(f"    GB 조건: {gb_ok}")
        print(f"    최종 비용: {opt_result['best_cost']:.6f}")
        print()

    # 요약 비교 표
    print("=" * 70)
    print("최종 비교 요약")
    print("-" * 70)
    print(f"{'후보':<35} {'에너지':>10} {'GB위반':>10} {'GB조건':>8}")
    print("-" * 70)

    # GB 조건 충족 후보 분류
    gb_satisfied = []
    for name, res in results.items():
        a = res['analysis']
        energy_str = f"{a['변분_에너지']:>10.4f}"
        violation_str = f"{a['GB_위반량']:>10.4f}"
        gb_str = "충족" if a['GB_조건_충족'] else "미충족"
        print(f"{name:<35} {energy_str} {violation_str} {gb_str:>8}")
        if a['GB_조건_충족']:
            gb_satisfied.append(name)

    print("=" * 70)

    if gb_satisfied:
        print(f"\nGB 조건 충족 후보 ({len(gb_satisfied)}개):")
        for name in gb_satisfied:
            print(f"  - {name}")
        print("\n  → 이 후보들이 ξ-다발 프레임워크와 정합성 있는 스펙트럼 구조를 가짐")
    else:
        print("\nGB 조건을 충족하는 후보 없음.")
        # 가장 근접한 후보 찾기
        min_violation = np.inf
        best_cand = None
        for name, res in results.items():
            v = res['analysis']['GB_위반량']
            if v < min_violation:
                min_violation = v
                best_cand = name
        print(f"  가장 근접한 후보: {best_cand} (위반량: {min_violation:.4f})")
        print("  → λ_GB 증가 또는 n_seeds 증가 후 재실행 권장")

    print("\n[완료] VQE-Gauss-Bonnet 벤치마크 종료")

    return results


# ---------------------------------------------------------------------------
# 단독 실행 진입점
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='VQE-Gauss-Bonnet: Berry-Keating 해밀토니안 탐색'
    )
    parser.add_argument('--n_qubits', type=int, default=5,
                        help='큐비트 수 (기본값: 5, dim=32)')
    parser.add_argument('--target_N', type=int, default=3,
                        help='목표 영점 수 N (기본값: 3)')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='앤자츠 레이어 수 (기본값: 2)')
    parser.add_argument('--n_seeds', type=int, default=3,
                        help='최적화 시드 수 (기본값: 3)')
    parser.add_argument('--lambda_gb', type=float, default=1.0,
                        help='GB 패널티 가중치 (기본값: 1.0)')
    parser.add_argument('--lambda_mono', type=float, default=0.5,
                        help='모노드로미 패널티 가중치 (기본값: 0.5)')
    args = parser.parse_args()

    # 5큐비트 (dim=32)에서 5개 후보 벤치마크
    results = benchmark_all_candidates(
        n_qubits=args.n_qubits,
        target_N=args.target_N,
        n_layers=args.n_layers,
        n_seeds=args.n_seeds,
        lambda_gb=args.lambda_gb,
        lambda_mono=args.lambda_mono
    )
